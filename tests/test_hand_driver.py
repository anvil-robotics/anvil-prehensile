"""Unit tests for prehensile/hand_driver.py -- the HandDriver Protocol,
L6Driver, and build_driver().

FAKE DRIVER ONLY, no CAN, no hardware, no realhand import: this file must
collect and pass in CI's "lean" job, which asserts realhand is NOT importable
in its venv (see .github/workflows/ci.yml) and still runs this module's
tests. L6Driver itself is exercised through its `_l6_cls` test-only injection
point (mirrors prehensile.l6_discovery.resolve_hands's own `_probe` param):
a fake standing in for realhand.L6, never the real SDK.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import types
from types import SimpleNamespace

import pytest

from prehensile.hand import Channel, HandDescriptor, Output
from prehensile.hand_driver import HandOutputError, L6Driver, build_driver
from prehensile.hand_loader import load_hand_descriptor


# --------------------------------------------------------------------------- #
# Fakes: mimic just enough of realhand.L6's shape (constructor, context
# manager, start_polling, .speed.set_speeds, .angle.set_angles) to exercise
# L6Driver's wrapping contract, each call independently failable.
# --------------------------------------------------------------------------- #
class _FakeL6:
    def __init__(self, *, fail_open=False, fail_polling=False, fail_speed=False,
                 fail_send=False, fail_close=False):
        self._fail_open = fail_open
        self._fail_polling = fail_polling
        self._fail_speed = fail_speed
        self._fail_send = fail_send
        self._fail_close = fail_close
        self.poll_calls: list[dict] = []
        self.speed = SimpleNamespace(set_speeds=self._set_speeds)
        self.angle = SimpleNamespace(set_angles=self._set_angles)
        self.speed_calls: list[list[float]] = []
        self.angle_calls: list[list[float]] = []

    def __enter__(self):
        if self._fail_open:
            raise RuntimeError("fake open failure")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fail_close:
            raise RuntimeError("fake close failure")

    def start_polling(self, intervals):
        self.poll_calls.append(dict(intervals))
        if self._fail_polling:
            raise RuntimeError("fake polling failure")

    def _set_speeds(self, speeds):
        if self._fail_speed:
            raise RuntimeError("fake speed failure")
        self.speed_calls.append(list(speeds))

    def _set_angles(self, angles):
        if self._fail_send:
            raise RuntimeError("fake send failure")
        self.angle_calls.append(list(angles))


def _fake_l6_factory(fake=None, **fail_kwargs):
    """Build an `_l6_cls`-shaped factory: ignores the (side, interface_name)
    kwargs L6Driver calls it with and returns one pre-configured `_FakeL6`."""
    fake = fake if fake is not None else _FakeL6(**fail_kwargs)

    def factory(*, side, interface_name):
        del side, interface_name
        return fake

    factory.fake = fake
    return factory


def _driver(**fail_kwargs) -> L6Driver:
    return L6Driver(side="left", interface_name="hand_l", _l6_cls=_fake_l6_factory(**fail_kwargs))


# --------------------------------------------------------------------------- #
# 1. The safety contract: HandOutputError, chained, never swallowed.
# --------------------------------------------------------------------------- #
def test_send_failure_raises_hand_output_error_chained():
    driver = _driver(fail_send=True)
    with driver:
        with pytest.raises(HandOutputError) as excinfo:
            driver.send([0.0] * 6)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "fake send failure" in str(excinfo.value.__cause__)


def test_send_failure_is_observable_not_swallowed():
    """Pins the NEGATIVE half of the contract: a failing send() must be
    observable to the caller (an exception reaches them), never
    logged-and-ignored. A regression to catch-and-continue would make send()
    return normally here instead of raising, and this assert would fail."""
    driver = _driver(fail_send=True)
    with driver:
        raised = False
        try:
            driver.send([0.0] * 6)
        except HandOutputError:
            raised = True
        assert raised, "send() swallowed a failing SDK call instead of raising"


def test_open_failure_raises_hand_output_error_chained():
    driver = _driver(fail_open=True)
    with pytest.raises(HandOutputError) as excinfo:
        driver.__enter__()
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "fake open failure" in str(excinfo.value.__cause__)


def test_polling_failure_during_open_raises_hand_output_error_chained():
    """start_polling() (the force-sensor-drop call, see __enter__) is part of
    open -- a failure there must also surface as HandOutputError, not a raw
    SDK exception."""
    driver = _driver(fail_polling=True)
    with pytest.raises(HandOutputError) as excinfo:
        driver.__enter__()
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_close_failure_raises_hand_output_error_chained():
    driver = _driver(fail_close=True)
    driver.__enter__()
    with pytest.raises(HandOutputError) as excinfo:
        driver.__exit__(None, None, None)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "fake close failure" in str(excinfo.value.__cause__)


def test_set_speed_failure_raises_hand_output_error_chained():
    driver = _driver(fail_speed=True)
    with driver:
        with pytest.raises(HandOutputError) as excinfo:
            driver.set_speed(40.0)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "fake speed failure" in str(excinfo.value.__cause__)


def test_send_before_enter_raises_hand_output_error():
    """Using the driver outside its open/close bracket fails informatively
    rather than an opaque AttributeError on `None`."""
    driver = _driver()
    with pytest.raises(HandOutputError, match="before __enter__"):
        driver.send([0.0] * 6)


def test_send_after_exit_raises_hand_output_error():
    driver = _driver()
    with driver:
        pass
    with pytest.raises(HandOutputError, match="before __enter__"):
        driver.send([0.0] * 6)


# --------------------------------------------------------------------------- #
# Happy path: speed expands to every channel, send forwards channels verbatim,
# and __enter__ drops the FORCE_SENSOR poll (angle-only), per the module
# docstring's ENOBUFS story.
# --------------------------------------------------------------------------- #
def test_l6_driver_n_channels_is_six():
    assert L6Driver.n_channels == 6


def test_set_speed_expands_to_every_channel():
    driver = _driver()
    with driver:
        driver.set_speed(42.0)
    fake = driver._l6_cls.fake
    assert fake.speed_calls == [[42.0] * 6]


def test_send_forwards_channels_unchanged():
    driver = _driver()
    with driver:
        driver.send([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fake = driver._l6_cls.fake
    assert fake.angle_calls == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_enter_drops_force_sensor_poll_to_angle_only():
    """__enter__ must re-call start_polling with ONLY an angle entry (see the
    module docstring): three uncoordinated CAN writers overflow SocketCAN's
    default 10-frame txqueuelen on real hardware."""
    driver = _driver()
    with driver:
        pass
    fake = driver._l6_cls.fake
    assert len(fake.poll_calls) == 1
    assert list(fake.poll_calls[0].keys()) == ["angle"]


# --------------------------------------------------------------------------- #
# 2. Lazy resolution: loading a descriptor with an output.driver reference
#    must never import it -- only build_driver() does, and only when called.
# --------------------------------------------------------------------------- #
_MINIMAL_WITH_DRIVER = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: thumb_flex, role: thumb_flex, home: 100}
  - {name: thumb_abd,  role: thumb_abd,  home: 100}
  - {name: index,      role: index,      home: 0}
  - {name: middle,     role: middle,     home: 0}
  - {name: ring,       role: ring,       home: 0}
  - {name: pinky,      role: pinky,      home: 0}
output:
  driver: "definitely_not_a_real_module_xyz:Factory"
"""


def test_loading_descriptor_with_driver_ref_does_not_import_it(tmp_path):
    sys.modules.pop("definitely_not_a_real_module_xyz", None)
    path = tmp_path / "hand.yml"
    path.write_text(_MINIMAL_WITH_DRIVER)

    hand = load_hand_descriptor(path)

    assert hand.output.driver == "definitely_not_a_real_module_xyz:Factory"
    assert "definitely_not_a_real_module_xyz" not in sys.modules


def test_constructing_hand_descriptor_directly_does_not_import_driver():
    """Same guarantee, without going through YAML at all -- HandDescriptor's
    own construction (see prehensile/hand.py) never touches output.driver."""
    sys.modules.pop("also_not_a_real_module_xyz", None)
    HandDescriptor(
        name="direct",
        channels=(Channel(name="a", role="index", home=0.0),),
        pinch=None,
        group=(),
        output=Output(driver="also_not_a_real_module_xyz:Factory"),
    )
    assert "also_not_a_real_module_xyz" not in sys.modules


# --------------------------------------------------------------------------- #
# 3. build_driver(): resolves "module:attr" lazily, checks channel COUNT.
# --------------------------------------------------------------------------- #
def _minimal_descriptor(n_channels: int, driver_ref: str | None) -> HandDescriptor:
    """A pinch/group-free descriptor with `n_channels` identically-"index"-roled
    channels -- role reuse across channels is allowed (see test_hand.py), so
    this stays valid for any count without needing 6 distinct roles."""
    channels = tuple(Channel(name=f"c{i}", role="index", home=0.0) for i in range(n_channels))
    return HandDescriptor(name="synthetic", channels=channels, pinch=None, group=(),
                           output=Output(driver=driver_ref))


class _FakeDriver:
    """A minimal, standalone HandDriver double -- unrelated to L6Driver --
    used only to exercise build_driver()'s generic module:attr + channel-count
    contract."""

    def __init__(self, n_channels: int, **kwargs):
        self.n_channels = n_channels
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def set_speed(self, speed: float) -> None:
        pass

    def send(self, channels) -> None:
        pass


def _inject_fake_driver_module(monkeypatch, module_name: str, n_channels: int):
    mod = types.ModuleType(module_name)
    mod.make = lambda **kw: _FakeDriver(n_channels, **kw)
    monkeypatch.setitem(sys.modules, module_name, mod)


def test_build_driver_resolves_reference_and_forwards_kwargs(monkeypatch):
    _inject_fake_driver_module(monkeypatch, "fake_driver_mod_ok", n_channels=2)
    descriptor = _minimal_descriptor(2, "fake_driver_mod_ok:make")

    driver = build_driver(descriptor, side="left", interface_name="hand_l")

    assert isinstance(driver, _FakeDriver)
    assert driver.n_channels == 2
    assert driver.kwargs == {"side": "left", "interface_name": "hand_l"}


def test_build_driver_channel_count_mismatch_fails_loudly(monkeypatch):
    """The failure mode this check exists for: right numbers, wrong finger
    order (or, more directly testable: just the wrong count)."""
    _inject_fake_driver_module(monkeypatch, "fake_driver_mod_mismatch", n_channels=3)
    descriptor = _minimal_descriptor(2, "fake_driver_mod_mismatch:make")

    with pytest.raises(ValueError, match=r"has 3 channel\(s\) but the descriptor has 2"):
        build_driver(descriptor)


def test_build_driver_missing_output_driver_raises():
    descriptor = _minimal_descriptor(2, driver_ref=None)
    with pytest.raises(ValueError, match="output.driver is not set"):
        build_driver(descriptor)


def test_build_driver_malformed_reference_raises(monkeypatch):
    descriptor = _minimal_descriptor(2, driver_ref="not-a-module-colon-attr")
    with pytest.raises(ValueError, match="must be 'module:attr'"):
        build_driver(descriptor)


def test_build_driver_unknown_attribute_raises(monkeypatch):
    _inject_fake_driver_module(monkeypatch, "fake_driver_mod_no_attr", n_channels=2)
    descriptor = _minimal_descriptor(2, "fake_driver_mod_no_attr:nonexistent")
    with pytest.raises(ValueError, match="has no attribute 'nonexistent'"):
        build_driver(descriptor)


# --------------------------------------------------------------------------- #
# Import hygiene: the module must not pull in realhand at import time (run in
# a subprocess -- see tests/test_l6_discovery.py's identical rationale: other
# test modules in this session may already have imported realhand elsewhere).
# --------------------------------------------------------------------------- #
def test_importing_module_does_not_import_realhand():
    code = (
        "import sys\n"
        "import prehensile.hand_driver\n"
        "assert 'realhand' not in sys.modules, sorted(sys.modules)\n"
        "assert 'can' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "OK"


# --------------------------------------------------------------------------- #
# The shipped L6 descriptor names its own driver, and that reference resolves.
# Without this the descriptor->driver seam is real for a contributor's hand but
# decorative for the one hand this package actually ships, which is exactly the
# kind of gap that goes unnoticed until someone else's hand is the first to
# exercise it.
# --------------------------------------------------------------------------- #
def test_shipped_l6_descriptor_names_its_driver():
    from prehensile.curlmap import L6_HAND

    assert L6_HAND.output.driver == "prehensile.hand_driver:L6Driver"


def test_build_driver_resolves_the_shipped_l6_descriptor_end_to_end():
    """build_driver(L6_HAND, ...) must work, with no realhand installed: the
    reference resolves, the channel-count check passes against the descriptor's
    own 6 channels, and a fake stands in for the SDK."""
    from prehensile.curlmap import L6_HAND
    from prehensile.hand_driver import L6Driver, build_driver

    driver = build_driver(
        L6_HAND, side="left", interface_name="hand_l", _l6_cls=_FakeL6,
    )
    assert isinstance(driver, L6Driver)
    assert driver.n_channels == len(L6_HAND.channels) == 6


def test_importing_curlmap_does_not_import_the_driver_module():
    """curlmap.py names the driver as a STRING. Importing the mapper must not
    drag in hand_driver (and through it, on a machine that has it, realhand)."""
    code = (
        "import sys; import prehensile.curlmap;"
        "bad=[m for m in ('prehensile.hand_driver','realhand','can') if m in sys.modules];"
        "print('LEAKED:'+','.join(bad) if bad else 'CLEAN')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert "CLEAN" in out.stdout, out.stdout
