"""WujiSource: the Wuji SDK glove -> (21,3) MediaPipe keypoints.

Ported from ``prehensile_v1``'s ``prehensile/glove.py`` (connection with stale-
session retry) and ``prehensile/keypoints.py`` (drain-latest keypoint read),
folded into a single class that exposes EXACTLY the same seam as
``prehensile.udcap.UDCapSource``: a context manager with a synchronous
``poll() -> (21,3) float32 | None``. Downstream code (retargeting, teleop
loops) can therefore treat the UDP glove and the Wuji SDK glove
interchangeably -- construct either one, use it as a context manager, call
``poll()`` in a loop.

The v1 ``main()`` (see ``prehensile_v1/prehensile/teleop.py`` lines ~82-117)
used to own the ``SdkManager`` lifecycle (acquire the manager singleton, swap
to the default user so the built-in URDF drives ``hand_skeleton`` FK, connect
the glove, subscribe, and on the way out restore the previous user and
disconnect). That lifecycle now lives entirely inside ``WujiSource.__enter__``
/ ``.close()``, matching how ``UDCapSource`` owns its own socket lifecycle.

LAZY IMPORT: ``wuji_sdk`` is NOT imported at module import time. This module
must import cleanly even when ``wuji_sdk`` is not installed (e.g. on a dev
machine with no glove SDK, or in CI). Every ``wuji_sdk`` symbol
(``SdkManager``, ``ConnectOptions``, ``WujiGlove``) is imported inside the
function/method that actually needs it, so merely importing
``prehensile.wuji`` -- or calling the pure numpy helpers below -- never
requires the SDK to be present. Only entering a ``WujiSource`` (i.e. actually
talking to hardware) does.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

# --------------------------------------------------------------------------- #
# Pure keypoint helpers (numpy-only; ported verbatim from
# prehensile_v1/prehensile/keypoints.py). No wuji_sdk dependency.
# --------------------------------------------------------------------------- #


def read_latest_keypoints(skeleton_sub):
    """Drain the subscription; return the newest (21, 3) float32 frame, or None.

    The glove publishes ``hand_skeleton`` faster than a control loop consumes
    it, and ``recv()`` returns the OLDEST unread frame, so taking one per tick
    falls behind. Drain every queued frame and keep only the latest. (Pattern
    from wuji-sdk examples/python/retargeting/1.teleop_real.py.)

    The keypoints come out in standard MediaPipe order (index 0 = wrist;
    fingertips at 4/8/12/16/20), in meters, in the glove's wrist-local frame.
    """
    latest = None
    while True:
        frame = skeleton_sub.recv()
        if frame is None:
            break
        latest = frame
    if latest is None:
        return None
    return np.array([j.pose.position for j in latest.joints], dtype=np.float32)


def wrist_center(kp):
    """Subtract the wrist (landmark 0) so the keypoints are wrist-relative.

    dex_retargeting's vector optimizer only uses difference vectors, so this is
    not strictly required, but it matches RealHand's PICO front-end (which
    recenters on the wrist) and keeps values small.

    Kept as a helper only -- ``WujiSource.poll()`` deliberately does NOT apply
    it, to match the confirmed-working v1 behaviour exactly.
    """
    return kp - kp[0:1, :]


# --------------------------------------------------------------------------- #
# Connection helpers (ported from prehensile_v1/prehensile/glove.py). The
# glove allows only ONE session at a time. Connect fails with "Session already
# exists" if a previous run didn't disconnect cleanly (hard kill, or a quick
# restart before the device's heartbeat timeout), or if another process (Wuji
# Studio, another teleop/viz terminal) holds the session. We retry with a
# delay: a stale session frees itself once the heartbeat times out; a live one
# clears when you close it.
#
# NOTE the lazy import: ``wuji_sdk`` is imported HERE, inside the function that
# needs it, not at module top.
# --------------------------------------------------------------------------- #


def _scan_and_connect(manager):
    from wuji_sdk import ConnectOptions, WujiGlove

    no_bridge = ConnectOptions(enable_bridge=False)
    glove = None
    for d in manager.scan():
        dev = manager.connect(sn=d.sn, device_name=d.sn, options=no_bridge)
        if isinstance(dev, WujiGlove):
            glove = dev
    return glove


def _is_session_conflict(exc):
    msg = str(exc).lower()
    return "already exists" in msg or "another session" in msg


def connect_glove(manager, attempts=12, delay=4.0):
    """Return a connected WujiGlove, retrying past a stale/other session.

    Returns None if no glove is present, or if a conflicting session never
    clears within attempts*delay seconds (a clear message is printed in that
    case).
    """
    for attempt in range(1, attempts + 1):
        try:
            glove = _scan_and_connect(manager)
        except Exception as exc:
            if not _is_session_conflict(exc):
                raise
            if attempt == 1:
                print("Glove session already active -- is Wuji Studio or another "
                      "teleop/viz still connected? Close it.")
            with contextlib.suppress(Exception):
                manager.disconnect_all()  # clear our client-side view before retrying
            if attempt < attempts:
                print(f"  waiting for the old session to time out "
                      f"(retry {attempt}/{attempts} in {delay:.0f}s)...")
                time.sleep(delay)
            continue
        return glove  # may be None if no glove among devices; caller handles that
    print("Could not acquire the glove: a session is still holding it. Close Wuji "
          "Studio and any other teleop/viz terminal, then try again.")
    return None


# --------------------------------------------------------------------------- #
# The Wuji SDK glove source, matching UDCapSource's shape.
# --------------------------------------------------------------------------- #
class WujiSource:
    """Wuji SDK glove source behind the same seam as ``UDCapSource``.

    ``poll()`` returns the newest (21, 3) float32 MediaPipe-order keypoint
    frame (wrist-local metres, NOT wrist-centered -- see ``wrist_center``), or
    ``None`` if no fresh frame is queued. Use as a context manager::

        with WujiSource() as src:
            kp = src.poll()

    ``__enter__`` owns the ``SdkManager`` lifecycle that used to live in v1's
    ``main()``: acquire the singleton manager, remember + swap away from the
    caller's current user (the built-in default user gives reliable
    ``hand_skeleton`` FK), connect the glove (with stale-session retry via
    ``connect_glove``), and subscribe to its hand skeleton stream. ``close()``
    (called by ``__exit__``) restores the previous user and disconnects,
    mirroring v1's ``finally`` block.

    Unlike the UDP glove, the Wuji SDK glove is discovered by scanning
    connected hardware rather than by a JSON/protobuf hand-side field, so
    there is no left/right selection to make here -- whichever physical glove
    is plugged in is what gets connected. ``side`` is accepted and stored only
    for API symmetry with ``UDCapSource``.
    """

    def __init__(self, side: str = "left"):
        # Not used to select a glove (the hardware determines that); kept only
        # so callers can construct WujiSource(side=...) interchangeably with
        # UDCapSource(side=...).
        self.side = side
        self._manager = None
        self._previous_user = None
        self._sub = None

    # -- context manager ---------------------------------------------------- #
    def __enter__(self) -> "WujiSource":
        from wuji_sdk import SdkManager

        self._manager = SdkManager.instance()
        self._previous_user = self._manager.current_user()
        self._manager.switch_to_default_user()  # built-in URDF -> reliable hand_skeleton FK
        glove = connect_glove(self._manager)
        if glove is None:
            raise RuntimeError(
                "No Wuji glove found (scan returned no device, or a "
                "conflicting session never cleared -- see console output "
                "above for details)."
            )
        self._sub = glove.hand_skeleton().subscribe()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Restore the previous user and disconnect. Safe to call more than
        once, and safe to call even if ``__enter__`` was never (successfully)
        run."""
        if self._manager is None:
            return
        if self._previous_user is not None:
            with contextlib.suppress(Exception):
                self._manager.switch_user(self._previous_user["user_id"])
        with contextlib.suppress(Exception):
            self._manager.disconnect_all()
        self._manager = None
        self._previous_user = None
        self._sub = None

    # -- polling ------------------------------------------------------------ #
    def poll(self) -> np.ndarray | None:
        """Drain the subscription; return the newest (21,3) float32 frame, or None."""
        return read_latest_keypoints(self._sub)
