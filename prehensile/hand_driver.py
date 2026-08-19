"""HandDriver: the output half of the hand seam (Phase 4b of the upstream migration).

Phase 4a (``prehensile.hand``) gave a non-L6 hand somewhere to describe itself
(channels, roles, home pose). This module gives it somewhere to plug in an
actual output device: a small ``HandDriver`` Protocol (open/close as a context
manager, set a uniform motor speed, send one frame of per-channel commands),
one real implementation (``L6Driver``, wrapping ``realhand``), and
``build_driver()``, which lazily resolves a ``HandDescriptor.output.driver``
("module:attr") reference into a live driver instance.

``realhand``/``can`` (the CAN SDK) are an optional extra (``[l6]``, see
pyproject.toml) -- a bare ``pip install prehensile`` does not have them. Every
import of either therefore stays FUNCTION-LOCAL, inside ``L6Driver``'s methods,
never at this module's top level: this whole file, ``HandDriver``/
``HandOutputError``/``L6Driver``/``build_driver`` included, must import cleanly
with neither package installed (verified in a lean venv; see the package's CI
"lean" job, which asserts ``realhand`` is NOT importable and still collects and
runs this module's tests).

``build_driver()`` is likewise lazy about the "module:attr" string itself: it
is resolved (``importlib.import_module``) only when a driver is actually being
built, never when a ``HandDescriptor`` is merely constructed or loaded from
YAML (``prehensile.hand`` / ``prehensile.hand_loader``) -- reading a hand's
config must never force a CAN import just because the config happened to name
a driver.

------------------------------------------------------------------------------
THE SAFETY CONTRACT -- why HandOutputError exists, and why it must never be
caught-and-swallowed inside this module.
------------------------------------------------------------------------------

The ROS consumer of the equivalent output loop (``hand_teleop_node``, which
drives up to two L6 hands from one node) used to let a CAN failure from
``set_angles`` propagate straight out of its per-frame timer callback. That
propagation killed the whole node process -- which also stopped that node's
``*_arm_enable`` keep-alive publish, so the arm side's staleness fail-safe
(watching for a stale/absent publish) latched BOTH arms disabled a fraction of
a second later. A flaky hand bricked the whole robot, observed repeatedly
2026-07-28. The fix was NOT to make the failure fatal in a "safer" way -- it
was to retire the failing hand and keep the node's timer loop running, so the
other hand keeps actuating and both arm-enable topics keep publishing.

The lesson generalizes to this driver: **propagation out of the consumer's
loop is the defect, not a fail-safe** -- but it is a defect the *consumer*
must be able to fix, which means the driver has to hand the consumer
something it can actually catch. So:

  * A send/open/speed/close failure here always raises the distinct
    ``HandOutputError``, with the original SDK exception chained as its
    ``__cause__`` (``raise HandOutputError(...) from exc``) -- never a raw
    ``CANError``/``can.CanError``/``OSError``/whatever else the SDK feels like
    throwing this week, and never silently swallowed-and-logged. A caller that
    cannot tell a hand failed cannot decide to retire it.
  * This driver does NOT decide what happens after that. Retiring a side,
    logging, retrying, aborting the whole process (as the CLI in
    ``prehensile.teleop`` does today, correctly, for its single hand): all of
    that is consumer policy, made by whoever calls ``send()``/``set_speed()``/
    opens the context manager, never by this module.
  * Internally this means catching broad ``Exception`` around each SDK call
    (we do not know, and should not need to enumerate, every exception type a
    third-party CAN SDK can raise) and immediately re-raising as
    ``HandOutputError`` -- which is NOT the same thing as ``except Exception:
    pass`` or ``except Exception: log.error(...)``: nothing is discarded, the
    original is always attached, and the caller ever sees only ONE type to
    catch. That is a strictly narrower, MORE catchable contract than the
    ``except Exception:  # noqa: BLE001`` a consumer would otherwise be forced
    to write at every call site (and did, before this module existed) -- do
    not "simplify" this back to a bare re-raise or a swallow; both defeat the
    reason this type exists.

------------------------------------------------------------------------------
THE FORCE-SENSOR POLL, AND WHY THIS DRIVER DROPS IT ON OPEN.
------------------------------------------------------------------------------

``L6.__init__`` auto-starts polling BOTH sensor sources -- ANGLE at 60 Hz and
FORCE_SENSOR at 30 Hz -- each on its own background thread, and the SDK's
force-sensor poll (``_send_sense_request``) emits one CAN frame PER SENSOR in a
tight loop: it is the burstiest writer in the process. Three uncoordinated
writers (that poll, the angle poll, and this driver's own ``send()``) against
SocketCAN's default 10-frame ``txqueuelen`` on a USB CAN adapter overflow the
queue -> ``ENOBUFS`` -> the SDK declares the bus fatal -> every subsequent call
raises. Observed repeatedly against real hardware. Nothing reads force-sensor
data today, so ``L6Driver.__enter__`` immediately re-calls ``start_polling()``
with ONLY the angle entry, which replaces (not adds to) the SDK's default set
and removes the force-sensor writer entirely. The angle poll stays: it is
cheap (one frame, not one per sensor) and readback (e.g. ``/hand/joint_states``)
needs it.
"""
from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol

from prehensile.hand import HandDescriptor


class HandOutputError(Exception):
    """A HandDriver's open/set_speed/send/close failed against the real device.

    Always wraps the underlying SDK exception as ``__cause__``. See this
    module's docstring for why this type exists and why it must never be
    swallowed instead of raised.
    """


class HandDriver(Protocol):
    """What the teleop output loop needs from a hand, real or fake.

    Deliberately just three verbs plus a channel count -- exactly what
    ``prehensile.teleop`` and the ROS hand-teleop consumer's output loop use
    today (open/close as a context manager, set one uniform motor speed,
    send one frame of per-channel commands). Nothing here invents a
    capability neither of those two call sites needs: in particular, no
    per-channel speed (both consumers only ever set one speed for every
    channel) and no readback (neither consumer reads angles/force back
    through this seam).
    """

    #: How many channels one ``send()`` frame must carry, checked by
    #: ``build_driver()`` against the bound ``HandDescriptor`` -- see there.
    n_channels: int

    def __enter__(self) -> "HandDriver":
        """Open the device. Raises ``HandOutputError`` on failure."""
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the device. Raises ``HandOutputError`` on failure."""
        ...

    def set_speed(self, speed: float) -> None:
        """Set every channel's motor speed to ``speed`` (0-100, device units)."""
        ...

    def send(self, channels: Sequence[float]) -> None:
        """Command one frame: ``len(channels) == n_channels`` values, in the
        driver's own channel order."""
        ...


class L6Driver:
    """``HandDriver`` over a real RealHand L6, via the ``realhand`` SDK.

    ``realhand``/``can`` are imported function-locally, inside ``__enter__``,
    ``set_speed`` and ``send`` -- see the module docstring -- never at class
    or module scope, so importing this class costs nothing without the
    ``[l6]`` extra installed.

    ``_l6_cls`` is a test-only escape hatch (mirrors ``prehensile.l6_discovery
    .resolve_hands``'s own ``_probe`` parameter): the real ``realhand.L6``
    class by default, or a fake standing in for it in tests, so the whole
    open/speed/send/close contract -- including the HandOutputError wrapping --
    can be exercised with no CAN hardware and without ``realhand`` installed
    at all.
    """

    n_channels = 6  # thumb_flex, thumb_abd, index, middle, ring, pinky

    def __init__(self, side: str, interface_name: str, *, _l6_cls=None) -> None:
        self._side = side
        self._interface_name = interface_name
        self._l6_cls = _l6_cls
        self._l6 = None  # set on successful __enter__; None means "not open"

    def __enter__(self) -> "L6Driver":
        if self._l6_cls is not None:
            # Test double: no realhand enum to build a poll-intervals dict
            # from, and the fake doesn't care what its keys are -- see
            # _FakeL6 in tests/test_hand_driver.py.
            l6_cls = self._l6_cls
            angle_only = {"angle": 1.0 / 60.0}
        else:
            # realhand/CAN imported only here -- see the module docstring.
            from realhand import L6
            from realhand.hand.l6 import SensorSource

            l6_cls = L6
            angle_only = {SensorSource.ANGLE: 1.0 / 60.0}

        try:
            l6 = l6_cls(side=self._side, interface_name=self._interface_name)
            l6.__enter__()
            # Drop the FORCE_SENSOR poll -- see the module docstring for why.
            l6.start_polling(angle_only)
        except Exception as exc:
            raise HandOutputError(
                f"opening L6 ({self._side}) on {self._interface_name!r} failed: {exc}"
            ) from exc

        self._l6 = l6
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        l6, self._l6 = self._l6, None
        if l6 is None:
            return
        try:
            l6.__exit__(exc_type, exc, tb)
        except Exception as close_exc:
            raise HandOutputError(f"closing L6 ({self._side}) failed: {close_exc}") from close_exc

    def set_speed(self, speed: float) -> None:
        self._require_open()
        try:
            self._l6.speed.set_speeds([speed] * self.n_channels)
        except Exception as exc:
            raise HandOutputError(f"setting L6 ({self._side}) speed failed: {exc}") from exc

    def send(self, channels: Sequence[float]) -> None:
        self._require_open()
        try:
            self._l6.angle.set_angles(channels)
        except Exception as exc:
            raise HandOutputError(f"L6 ({self._side}) send failed: {exc}") from exc

    def _require_open(self) -> None:
        if self._l6 is None:
            raise HandOutputError(
                f"L6 ({self._side}) driver used before __enter__ (or after __exit__)"
            )


def build_driver(descriptor: HandDescriptor, **kwargs) -> HandDriver:
    """Resolve ``descriptor.output.driver`` ("module:attr") and build a driver.

    Lazy: the referenced module is imported HERE, when a driver is actually
    being built -- never at descriptor-construction/load time (see
    ``prehensile.hand`` / ``prehensile.hand_loader``'s own docstrings; a bare
    YAML read must never force a CAN import). ``kwargs`` are forwarded
    verbatim to the resolved attribute (a class or factory callable) -- this
    function has no business knowing a given driver's construction arguments
    (``L6Driver`` wants ``side``/``interface_name``; another driver may want
    something else entirely).

    Raises ``ValueError`` (naming the hand) if ``output.driver`` is unset or
    not ``"module:attr"``-shaped, or if the built driver's channel COUNT
    disagrees with the descriptor's channel count. That last check catches
    exactly one failure mode -- the right number of channels wired to the
    wrong driver (or vice versa) -- and no more: two hands that both happen to
    have, say, 6 channels but disagree on ORDER pass this check silently,
    because a channel count alone cannot distinguish them.
    """
    ref = descriptor.output.driver
    if not ref:
        raise ValueError(f"hand {descriptor.name!r}: output.driver is not set; nothing to build")

    module_name, sep, attr_name = ref.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"hand {descriptor.name!r}: output.driver {ref!r} must be 'module:attr'")

    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"hand {descriptor.name!r}: output.driver {ref!r}: module {module_name!r} "
            f"has no attribute {attr_name!r}"
        ) from exc

    driver = factory(**kwargs)

    n_channels = len(descriptor.channels)
    if driver.n_channels != n_channels:
        raise ValueError(
            f"hand {descriptor.name!r}: driver {ref!r} has {driver.n_channels} channel(s) "
            f"but the descriptor has {n_channels}; a channel-COUNT mismatch is exactly the "
            "'right numbers, wrong finger order' failure mode this check exists to catch"
        )
    return driver
