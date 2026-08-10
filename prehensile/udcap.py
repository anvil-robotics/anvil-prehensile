"""UDCapSource: the UDP glove source -> (21,3) MediaPipe keypoints (Branch B).

A drain-latest, non-blocking UDP receiver that turns whatever the UdexReal
HandDriver unicasts into the teleop seam value: ``(21,3) float32`` wrist-local
metres MediaPipe keypoints, or ``None``. It parses the glove's per-finger
QUATERNIONS and runs them through ``prehensile.fk.keypoints_from_quats(q,
fk.FK_MODE_BY_SIDE[side], side=side)``. Both LEFT and RIGHT gloves are wired
and validated (see ``prehensile/fk.py``'s per-side FK mode/chirality tables).

Wire is unconfirmed (the user is remote), so BOTH serializations are supported
and auto-detected per datagram, exactly like ``tools/probe_udp.py``:

  * first non-whitespace byte ``{``  -> per-glove JSON (``DeviceName`` / ``Bones``)
  * else                             -> protobuf ``TeleopDataQuat``

Only Branch B (quaternions) is implemented. A future live probe may reveal
protobuf POSE instead; a clearly-marked hook is left for that parser but it is
NOT implemented here.

Public surface (Tasks 7-8 depend on these):
  UDCapSource         drain-latest UDP source; context manager; ``.poll()``.
  iter_datagrams      replay a length-prefixed ``.bin`` fixture.
  parse_quat_datagram pure parse+FK of one datagram (no networking) -> (21,3)|None.

proto3 parses foreign bytes leniently, so every parse is content-validated
(LeftHand present + exactly 15 finite joint quats); malformed / foreign /
right-only datagrams yield ``None``, never an exception or a bogus keypoint
array. The module is self-contained: the vendored ``handdriver_teleop_pb2`` now
lives inside the package at ``prehensile/_vendor/udex_protobuf`` (a normal,
importable subpackage), so it is imported directly by dotted path -- the path
shim its generated flat cross-import needs is contained entirely in that
subpackage's own ``__init__.py``. If that pb2 import fails the JSON path
still works; protobuf attempts then return ``None`` with a one-shot warning.
"""

from __future__ import annotations

import errno
import json
import socket
import struct
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from prehensile import fk

# --------------------------------------------------------------------------- #
# Vendored pb2 import: prehensile._vendor.udex_protobuf is a normal importable
# subpackage (its __init__.py carries the path shim the generated module's
# flat cross-import needs), so no local sys.path hack is required here.
# --------------------------------------------------------------------------- #
try:
    from prehensile._vendor.udex_protobuf import handdriver_teleop_pb2 as _pb2
except Exception:  # ImportError, or a protobuf runtime version mismatch
    _pb2 = None

# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #
_SO_RCVBUF = 1 << 20  # 1 MB socket receive buffer (spec)
_RECV_BUFSIZE = 65535  # max UDP payload; one datagram per recv
_LEN = struct.Struct("<I")  # uint32 LE frame-length prefix
_N_JOINTS = 15  # 15 finger quats in JointsIndex_Quat order
# Poll this many times seeing datagrams-but-no-valid-left-frame before the
# one-shot diagnostic warning fires (avoids spamming during momentary noise).
_WARN_AFTER_POLLS = 30

# One-shot warning bookkeeping (keyed so each distinct hint prints at most once).
_warned: set[str] = set()

# Running count of datagrams rejected by _parse_datagram (malformed / non-JSON /
# unparseable / wrong-side content) across the process. Diagnostic only -- never
# consulted for control flow, so it adds no branching to the accept path.
_reject_count = 0


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"udcap: {message}", file=sys.stderr)


def _note_reject() -> None:
    """Bump the reject counter and, the first time only, warn that datagrams are
    being rejected (malformed / non-JSON / unparseable / wrong-side content) --
    previously silent. Mirrors ``_warn_once``'s one-shot style."""
    global _reject_count
    _reject_count += 1
    _warn_once(
        "reject",
        f"rejected a malformed/unparseable datagram (not valid JSON or a valid "
        f"protobuf quat frame for the requested side); {_reject_count} rejected "
        f"so far this run.",
    )


# --------------------------------------------------------------------------- #
# Per-datagram parsing (pure; no sockets). Auto-detect JSON vs protobuf-quat.
# --------------------------------------------------------------------------- #
def _extract_quats(data: bytes, side: str) -> tuple[np.ndarray, bool, bool] | None:
    """Extract the requested glove's ((15,4) XYZW quats, aButton, bButton), or None. Never raises."""
    # Auto-detect on the first NON-whitespace byte (matches tools/probe_udp.py).
    if data.lstrip()[:1] == b"{":
        return _extract_json(data, side)
    # TODO(Branch A): a TeleopDataPose parser slots in here if a live probe shows
    # HandDriver emitting protobuf POSE instead of quaternions. Unimplemented.
    return _extract_protobuf(data, side)


def _extract_json(data: bytes, side: str) -> tuple[np.ndarray, bool, bool] | None:
    """Per-glove JSON datagram -> ((15,4) XYZW quats, aButton, bButton) for ``side``, or None."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("DeviceName")
    want = "R" if side == "right" else "L"
    if not isinstance(name, str) or not name.endswith(want):
        return None
    bones = obj.get("Bones")
    if not isinstance(bones, list) or len(bones) < _N_JOINTS:
        return None
    try:
        quats = np.asarray(bones[:_N_JOINTS], dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if quats.shape != (_N_JOINTS, 4) or not np.all(np.isfinite(quats)):
        return None
    a_button = bool(obj.get("aButton", False))
    b_button = bool(obj.get("bButton", False))
    return quats, a_button, b_button


def _extract_protobuf(data: bytes, side: str) -> tuple[np.ndarray, bool, bool] | None:
    """protobuf ``TeleopDataQuat`` -> ((15,4) XYZW quats, aButton, bButton) for ``side``, or None."""
    if _pb2 is None:
        _warn_once(
            "pb2",
            "protobuf runtime unavailable; cannot parse binary datagrams (JSON "
            "still works). Regenerate vendor/udex_protobuf if you need the "
            "protobuf wire.",
        )
        return None
    msg = _pb2.TeleopDataQuat()
    try:
        # proto3 is lenient, but genuinely foreign bytes can still raise here.
        msg.ParseFromString(data)
    except Exception:
        return None
    field = "RightHand" if side == "right" else "LeftHand"
    if not msg.HasField(field):
        return None
    hand_msg = getattr(msg, field)
    joints = hand_msg.joints
    # Content validation: leniently-parsed foreign messages (e.g. TeleopDataAngle)
    # must be rejected rather than mis-decoded into a bogus keypoint array.
    if len(joints) != _N_JOINTS:
        return None
    try:
        quats = np.array([[j.x, j.y, j.z, j.w] for j in joints], dtype=np.float64)
    except Exception:
        return None
    if quats.shape != (_N_JOINTS, 4) or not np.all(np.isfinite(quats)):
        return None
    a_button = bool(getattr(hand_msg, "aButton", False))
    b_button = bool(getattr(hand_msg, "bButton", False))
    return quats, a_button, b_button


def _parse_datagram(data: bytes, side: str) -> tuple[np.ndarray, bool, bool] | None:
    """Parse one datagram to ((21,3) keypoints, aButton, bButton) for ``side``, or None. Never raises."""
    if not data:
        return None
    result = _extract_quats(data, side)
    if result is None:
        _note_reject()
        return None
    quats, a_button, b_button = result
    try:
        kp = fk.keypoints_from_quats(quats, fk.FK_MODE_BY_SIDE[side], side=side)
    except Exception:
        return None
    return kp, a_button, b_button


def parse_quat_datagram(data: bytes) -> np.ndarray | None:
    """One glove datagram (JSON or protobuf-quat) -> (21,3) float32 keypoints.

    Auto-detects the serialization, selects the LEFT glove, extracts its 15
    finger quaternions and runs FK. Returns ``None`` (never raises) for empty,
    malformed, foreign, or right-only datagrams. Pure function, separated from
    the socket so the parse+FK path is unit-testable without networking.

    (aButton/bButton are threaded internally through ``_parse_datagram`` for
    ``UDCapSource.poll()``'s benefit, but this public function's contract is
    unchanged: it returns only the keypoints.)
    """
    result = _parse_datagram(data, "left")
    if result is None:
        return None
    kp, _a_button, _b_button = result
    return kp


# --------------------------------------------------------------------------- #
# Fixture replay.
# --------------------------------------------------------------------------- #
def iter_datagrams(path) -> Iterator[bytes]:
    """Yield payloads from a ``[uint32 LE len][payload]`` fixture file.

    The file is a flat sequence of [uint32 little-endian payload length]
    [payload bytes] frames, repeated to EOF. Stops cleanly at EOF or on a
    truncated final frame (partial header or short payload).
    """
    with Path(path).open("rb") as f:
        while True:
            header = f.read(4)
            if len(header) < 4:
                return
            (n,) = _LEN.unpack(header)
            payload = f.read(n)
            if len(payload) < n:
                return  # truncated tail
            yield payload


# --------------------------------------------------------------------------- #
# The drain-latest UDP source.
# --------------------------------------------------------------------------- #
class UDCapSource:
    """Non-blocking, drain-latest UDP glove source.

    ``poll()`` drains ALL queued datagrams and returns the NEWEST that yields a
    valid ``side``-hand frame as ``(21,3) float32`` keypoints (else ``None``, and
    the caller holds its last pose). As a side effect, ``poll()`` also updates
    ``self.aButton`` and ``self.bButton`` (bool, default ``False``) from that
    same newest valid datagram -- they are left unchanged on a no-frame poll,
    never guessed. Use as a context manager::

        with UDCapSource(port=5555) as src:
            kp = src.poll()
            pressed = src.aButton

    Both ``side="left"`` and ``side="right"`` are wired and validated: ``"right"``
    selects the ``R`` JSON glove / ``RightHand`` protobuf field and decodes it
    with the right-side FK mode/chirality (``fk.FK_MODE_BY_SIDE["right"]``,
    ``fk.OUTPUT_X_FLIP_BY_SIDE["right"]``).
    """

    def __init__(self, port: int = 5555, side: str = "left", host: str = "0.0.0.0"):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.host = host
        self._no_frame_polls = 0
        # The glove's aButton/bButton, updated by poll() only when a valid
        # datagram is parsed (stale on no-frame polls, never guessed).
        self.aButton = False
        self.bButton = False
        # Per-side (aButton, bButton), updated by poll_sides() only for a side
        # that had a valid frame this drain (stale otherwise, same contract as
        # aButton/bButton above). Independent of poll()/self.side.
        self.buttons_by_side: dict[str, tuple[bool, bool]] = {
            "left": (False, False),
            "right": (False, False),
        }

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SO_RCVBUF)
        try:
            self._sock.bind((host, port))
        except OSError as exc:
            self._sock.close()
            self._sock = None
            if exc.errno == errno.EADDRINUSE:
                raise OSError(
                    errno.EADDRINUSE,
                    f"UDP port {port} is already in use (EADDRINUSE). Another "
                    f"receiver holds it (a stale probe/recorder, or a second "
                    f"UDCapSource). Find the owner with:  ss -ulpn 'sport = "
                    f":{port}'  then stop it, or use a different port.",
                ) from exc
            raise
        self._sock.setblocking(False)
        # Reflect the actual bound port (meaningful when port=0 was requested).
        self.port = self._sock.getsockname()[1]

    # -- context manager ---------------------------------------------------- #
    def __enter__(self) -> "UDCapSource":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # -- polling ------------------------------------------------------------ #
    def _drain(self) -> list[bytes]:
        """Read every currently-queued datagram (oldest first) until it blocks."""
        out: list[bytes] = []
        if self._sock is None:
            return out
        while True:
            try:
                data, _addr = self._sock.recvfrom(_RECV_BUFSIZE)
            except BlockingIOError:
                break
            except OSError:
                break
            out.append(data)
        return out

    def poll(self) -> np.ndarray | None:
        """Drain all queued datagrams; return the NEWEST valid frame, else None.

        Iterates newest -> oldest and returns the first datagram that parses and
        content-validates. If datagrams arrived but none yield a valid ``side``
        frame, a one-shot diagnostic warning fires after a few such polls.
        """
        datagrams = self._drain()
        for data in reversed(datagrams):
            result = _parse_datagram(data, self.side)
            if result is not None:
                kp, a_button, b_button = result
                self._no_frame_polls = 0
                self.aButton = a_button
                self.bButton = b_button
                return kp
        if datagrams:
            # Traffic is arriving but nothing decodes to a valid left frame.
            self._no_frame_polls += 1
            if self._no_frame_polls >= _WARN_AFTER_POLLS:
                _warn_once(
                    "no_frame",
                    f"received datagrams but none yielded a valid {self.side}-hand "
                    f"quaternion frame after {self._no_frame_polls} polls. Check "
                    f"that HandDriver is sending Content=Quater and the "
                    f"{self.side} glove is awake / calibrated.",
                )
        return None

    def poll_sides(self, sides) -> dict[str, np.ndarray | None]:
        """Drain all queued datagrams ONCE; return the newest valid frame per side.

        For driving BOTH gloves off a single shared socket (e.g. a combined
        two-hand teleop node): unlike ``poll()`` (which is scoped to
        ``self.side``), this demultiplexes one drain across several requested
        ``sides`` in one shot. For each side in ``sides``, iterates the
        drained datagrams newest -> oldest and takes the first that
        ``_parse_datagram(data, side)`` decodes; returns
        ``{side: (21,3) float32 | None}``.

        As a side effect, stashes ``self.buttons_by_side[side] = (aButton,
        bButton)`` from that same newest valid datagram for each side that had
        one this drain (left stale -- not reset to ``False`` -- for a side
        with no valid frame this drain, exactly like ``poll()``'s
        ``self.aButton``/``self.bButton``).

        Do not call this in the same tick as ``poll()``: both drain the same
        socket, and calling both would race/duplicate the drain. Use exactly
        one of the two per poll cycle.
        """
        datagrams = self._drain()
        result: dict[str, np.ndarray | None] = {}
        for side in sides:
            frame = None
            for data in reversed(datagrams):
                parsed = _parse_datagram(data, side)
                if parsed is not None:
                    kp, a_button, b_button = parsed
                    frame = kp
                    self.buttons_by_side[side] = (a_button, b_button)
                    break
            result[side] = frame
        return result
