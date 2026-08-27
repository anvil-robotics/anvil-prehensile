"""TDD validators for tools/probe_wuji.py (Wuji glove readiness probe).

HARDWARE-FREE, SDK-FREE. Mirrors ``tests/test_wuji.py``'s policy: every test
here builds fakes in code and never imports ``wuji_sdk``. ``probe_wuji`` must
therefore keep its SDK import lazy (inside ``main()``), exactly like
``prehensile.wuji`` does -- test 1 pins that.

The numeric thresholds under test were MEASURED off the real glove
(WG1KA03260524030, firmware 0.11.4) on 2026-08-27: 120.0 Hz, wrist landmark at
the origin, wrist->middle-fingertip span 0.1838 m, bone lengths spanning
0.0185-0.0869 m, and a per-joint standard deviation of 3.6e-4 m while the glove
sat still. The synthetic hand below is built from those bone lengths so that a
"good" frame in these tests is the shape the hardware actually emits.
"""

import argparse

import numpy as np
import pytest

from tools.probe_wuji import (
    CHAINS,
    collect,
    format_row,
    keypoint_verdict,
    model_verdict,
    motion_verdict,
    read_model_path,
    render_report,
    side_verdict,
    verdict,
    _build_parser,
)


# --------------------------------------------------------------------------- #
# Fakes + a measured-realistic synthetic hand.
# --------------------------------------------------------------------------- #
# Measured off WG1KA03260524030 (see module docstring).
_MEASURED_BONES = {
    "thumb": [0.0319, 0.0350, 0.0350, 0.0312],
    "index": [0.0867, 0.0393, 0.0210, 0.0238],
    "middle": [0.0869, 0.0452, 0.0259, 0.0267],
    "ring": [0.0853, 0.0453, 0.0249, 0.0258],
    "pinky": [0.0811, 0.0317, 0.0185, 0.0249],
}


def _synthetic_hand():
    """(21,3) float32 frame with the real glove's bone lengths, wrist at origin.

    Each finger is laid out straight along +y, fanned along x so no two chains
    coincide. Not anatomically posed -- it only has to satisfy the same
    shape/finite/span/bone-length invariants the hardware frame does.
    """
    kp = np.zeros((21, 3), dtype=np.float32)
    for fan, (name, chain) in enumerate(CHAINS.items()):
        y = 0.0
        for i, bone in enumerate(_MEASURED_BONES[name]):
            y += bone
            kp[chain[i + 1]] = (0.02 * (fan - 2), y, 0.0)
    return kp


class _FakeResource:
    """Mimics the SDK's Resource wrapper: a ``.get()`` returning a value."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _RaisingResource:
    def __init__(self, exc):
        self._exc = exc

    def get(self):
        raise self._exc


class _FakeGlove:
    def __init__(self, model_path_resource):
        self._res = model_path_resource

    def hand_model_path(self):
        return self._res


class _FakeClock:
    """Monotonic clock advanced only by the sleeps the code under test makes."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


# --------------------------------------------------------------------------- #
# 1. Lazy SDK import (mirrors tests/test_wuji.py's import-without-SDK test).
# --------------------------------------------------------------------------- #
def test_module_imports_without_wuji_sdk(monkeypatch):
    """Importing the tool must not require wuji_sdk -- it is imported inside
    main(). Moving the import to module top would break this."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "wuji_sdk", None)  # poison the import
    mod = importlib.reload(importlib.import_module("tools.probe_wuji"))
    assert mod.keypoint_verdict(_synthetic_hand())[0] is True


# --------------------------------------------------------------------------- #
# 2. Argument parsing.
# --------------------------------------------------------------------------- #
def test_parser_defaults_match_probe_udp_shape():
    args = _build_parser().parse_args([])
    assert args.seconds == 5.0
    assert args.scan_only is False
    assert args.verbose is False  # SDK logs are quieted unless asked for


def test_scan_only_is_a_flag():
    assert _build_parser().parse_args(["--scan-only"]).scan_only is True


def test_there_is_no_hand_flag():
    """Handedness is DETECTED from the glove, never supplied. Re-adding a
    --hand flag would make this parse succeed."""
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["--hand", "right"])
    assert exc.value.code == 2  # argparse usage error


# --------------------------------------------------------------------------- #
# 3. keypoint_verdict: the shape/finite/scale invariants.
# --------------------------------------------------------------------------- #
def test_measured_hand_passes():
    ok, detail = keypoint_verdict(_synthetic_hand())
    assert ok, detail


def test_wrong_shape_is_rejected():
    ok, detail = keypoint_verdict(np.zeros((15, 3), dtype=np.float32))
    assert not ok
    assert "shape" in detail.lower()


def test_non_finite_is_rejected():
    kp = _synthetic_hand()
    kp[8] = np.nan
    ok, detail = keypoint_verdict(kp)
    assert not ok
    assert "finite" in detail.lower()


def test_collapsed_hand_is_rejected():
    """All-zero keypoints decode fine but mean the FK produced nothing."""
    ok, detail = keypoint_verdict(np.zeros((21, 3), dtype=np.float32))
    assert not ok
    assert "span" in detail.lower()


def test_millimetre_units_are_rejected():
    """A frame in mm rather than m has a 184 m span -- catches a unit change."""
    ok, detail = keypoint_verdict(_synthetic_hand() * 1000.0)
    assert not ok
    assert "span" in detail.lower()


def test_implausible_bone_length_is_rejected():
    """Perturb landmark 6, NOT a span endpoint: moving 0 or 12 would trip the
    span check first and this would silently stop testing bone lengths."""
    kp = _synthetic_hand()
    kp[6] = (0.0, 5.0, 0.0)  # index PIP flung 5 m from its MCP
    ok, detail = keypoint_verdict(kp)
    assert not ok
    assert "bone" in detail.lower()


# --------------------------------------------------------------------------- #
# 4. motion_verdict: a frozen stream is indistinguishable from a live one to
#    read_latest_keypoints, so the probe has to look across frames.
# --------------------------------------------------------------------------- #
def test_identical_frames_report_frozen():
    kp = _synthetic_hand()
    ok, detail = motion_verdict([kp.copy() for _ in range(10)])
    assert not ok
    assert "frozen" in detail.lower()


def test_jittering_frames_report_moving():
    """3.6e-4 m of per-joint sd is what the real glove shows sitting still."""
    rng = np.random.default_rng(0)
    base = _synthetic_hand()
    frames = [base + rng.normal(0, 3.6e-4, base.shape).astype(np.float32)
              for _ in range(10)]
    ok, detail = motion_verdict(frames)
    assert ok, detail


def test_single_frame_cannot_judge_motion():
    ok, _ = motion_verdict([_synthetic_hand()])
    assert ok  # not enough data is not a failure


# --------------------------------------------------------------------------- #
# 5. side_verdict: the glove reports its own handedness, so the probe detects
#    it rather than being told. Measured 2026-08-27: hand_side() -> 'right'.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["left", "right"])
def test_detected_side_names_the_teleop_flag_to_use(side):
    """The detected side is only useful if it tells you what to run next."""
    ok, detail = side_verdict(side)
    assert ok
    assert f"--hand {side}" in detail


def test_detected_side_is_case_insensitive():
    ok, detail = side_verdict("Right")
    assert ok
    assert "--hand right" in detail


def test_unrecognised_side_is_a_failure():
    """prehensile's HANDS only knows left/right; anything else cannot be run."""
    ok, detail = side_verdict("middle")
    assert not ok
    assert "middle" in detail


# --------------------------------------------------------------------------- #
# 6. model path: the SDK RAISES "Path not found" when no custom URDF is set,
#    rather than returning "". Measured on hardware 2026-08-27.
# --------------------------------------------------------------------------- #
def test_missing_model_path_reads_as_none():
    glove = _FakeGlove(_RaisingResource(
        RuntimeError("Path not found: calibration.hand_model_path")))
    assert read_model_path(glove) is None


def test_present_model_path_is_returned():
    glove = _FakeGlove(_FakeResource("/home/anvil/.wuji/hand.urdf"))
    assert read_model_path(glove) == "/home/anvil/.wuji/hand.urdf"


def test_unexpected_error_is_not_swallowed():
    """Only 'not found' means 'unset'; anything else is a real fault."""
    glove = _FakeGlove(_RaisingResource(RuntimeError("device timed out")))
    with pytest.raises(RuntimeError):
        read_model_path(glove)


def test_model_verdict_warns_on_builtin_fallback():
    ok, detail = model_verdict(None)
    assert not ok
    assert "built-in" in detail.lower()


def test_model_verdict_accepts_a_custom_urdf():
    ok, detail = model_verdict("/home/anvil/.wuji/hand.urdf")
    assert ok
    assert "hand.urdf" in detail


# --------------------------------------------------------------------------- #
# 7. collect: drains a subscription for a bounded wall-clock window.
# --------------------------------------------------------------------------- #
def test_collect_stops_at_the_deadline_and_keeps_frames():
    kp = _synthetic_hand()
    clock = _FakeClock()
    frames, elapsed = collect(lambda: kp, 0.05, monotonic=clock.monotonic,
                              sleep=clock.sleep, interval=0.01)
    assert elapsed >= 0.05
    assert len(frames) == 5


def test_collect_skips_none_reads():
    clock = _FakeClock()
    frames, _ = collect(lambda: None, 0.05, monotonic=clock.monotonic,
                        sleep=clock.sleep, interval=0.01)
    assert frames == []


# --------------------------------------------------------------------------- #
# 8. Buffered report. The SDK logs ~200 INFO lines to the console while
#    connecting and subscribing, so rows printed as they are computed end up
#    shredded across that flood. Every row is collected and rendered ONCE, after
#    teardown, which is what makes the result readable.
# --------------------------------------------------------------------------- #
def test_format_row_shows_status_label_and_detail():
    row = format_row("ok", "serial", "WG1KA03260524030")
    assert "[ok" in row
    assert "serial" in row
    assert "WG1KA03260524030" in row


def test_format_row_keeps_columns_aligned_across_statuses():
    """Ragged status columns are exactly what made the interleaved output hard
    to scan; a longer status must not shift the label column."""
    ok = format_row("ok", "serial", "x")
    fail = format_row("FAIL", "serial", "x")
    assert ok.index("serial") == fail.index("serial")


def test_render_report_preserves_section_and_row_order():
    out = render_report([
        ("identity", [("ok", "serial", "WG1K")]),
        ("readiness", [("ok", "hand side", "right"),
                       ("warn", "hand model", "built-in")]),
    ])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[0] == "identity:"
    assert "serial" in lines[1]
    assert lines[2] == "readiness:"
    assert "hand side" in lines[3]
    assert "hand model" in lines[4]


def test_render_report_marks_failing_rows():
    assert "[FAIL" in render_report([("s", [("FAIL", "stream", "no frames")])])


def test_verdict_is_ready_when_nothing_is_wrong():
    assert verdict([], []) == "READY"


def test_verdict_stays_ready_but_names_warnings():
    line = verdict([], ["built-in default URDF"])
    assert line.startswith("READY")
    assert "built-in default URDF" in line


def test_verdict_is_not_ready_when_anything_failed():
    line = verdict(["no keypoint frames"], ["built-in default URDF"])
    assert line.startswith("NOT READY")
    assert "no keypoint frames" in line
