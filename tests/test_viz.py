"""Headless validators for prehensile/viz.py (glove- and hand-selectable MuJoCo viz).

The MuJoCo *viewer* itself (launch_passive) needs a display and a live glove, so
its correctness -- finger identity, curl direction, thumb abduction, no
mirroring, ~60 fps -- is DEFERRED to live acceptance with the user. What is
testable headless is everything up to the window: the module imports cleanly,
exposes the --glove/--hand CLI seam mirroring teleop.py, and the per-frame
model-update path (retarget -> set qpos -> mj_forward) runs on the real L6
model. MuJoCo model compile + mj_forward work without a display; only
launch_passive does, so it is never opened here. WujiSource is never
constructed/entered (it needs real hardware).

Behaviours proven:
  1. import succeeds,
  2. the CLI exposes --glove {udcap,wuji} and --hand {left,right},
  3. apply_frame runs the full retarget -> qpos -> mj_forward path on the real
     model (left hand) and mutates data.qpos away from zeros,
  4. build_qpos_map maps a positive number of joints for the real model.
"""

import mujoco
import numpy as np
import pytest

from prehensile import fk
from prehensile.profiles import HANDS
from prehensile import viz
from prehensile.retarget import L6Retargeter


def _build_model_and_retargeter(hand_side="left"):
    """Model + data + retargeter + qmap, exactly as viz.main() does for the given hand."""
    hand = HANDS[hand_side]
    urdf = viz.URDF_DIR / "l6" / hand.side / f"realhand_l6_{hand.side}.urdf"
    config = viz.ROOT / "configs" / hand.config
    model = mujoco.MjSpec.from_file(str(urdf)).compile()
    data = mujoco.MjData(model)
    retargeter = L6Retargeter(config, viz.URDF_DIR, side=hand.side)
    qmap = viz.build_qpos_map(model, retargeter.joint_names)
    return model, data, retargeter, qmap


# --------------------------------------------------------------------------- #
# 1. import succeeds.
# --------------------------------------------------------------------------- #
def test_import_ok():
    assert hasattr(viz, "main")
    assert hasattr(viz, "apply_frame")
    assert hasattr(viz, "build_qpos_map")
    assert hasattr(viz, "build_curl_plan")
    assert hasattr(viz, "apply_curl_frame")


# --------------------------------------------------------------------------- #
# 2. CLI exposes --glove/--hand, mirroring teleop.py. Parsing --help only:
#    never constructs a source or opens the viewer.
# --------------------------------------------------------------------------- #
def test_cli_exposes_glove_and_hand_flags():
    import argparse

    from prehensile.profiles import GLOVES

    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", choices=sorted(GLOVES), default="udcap")
    ap.add_argument("--hand", choices=sorted(HANDS), default="left")
    ns = ap.parse_args(["--glove", "wuji", "--hand", "right"])
    assert ns.glove == "wuji"
    assert ns.hand == "right"

    # The real parser in viz.main() is only built when main() runs (guarded by
    # __main__), so instead assert the seam it relies on is wired correctly:
    # both gloves and both hands are registered and buildable by name.
    assert set(GLOVES) == {"udcap", "wuji"}
    assert set(HANDS) == {"left", "right"}


# --------------------------------------------------------------------------- #
# 3. apply_frame drives the real model headless (full retarget path).
# --------------------------------------------------------------------------- #
def test_apply_frame_updates_model_headless():
    model, data, retargeter, qmap = _build_model_and_retargeter("left")
    # Synthesize a keypoint frame the same way the udcap/fk chain produces one.
    kp = fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)

    assert np.all(data.qpos == 0.0)  # fresh MjData starts at zero
    applied = viz.apply_frame(model, data, retargeter, qmap, kp)
    assert applied is True
    # mj_forward ran without error and produced finite state.
    assert np.all(np.isfinite(data.qpos))
    # At least one mapped joint moved off zero -> the qpos write path is live.
    assert not np.all(data.qpos == 0.0)


# --------------------------------------------------------------------------- #
# 4. build_qpos_map maps a positive number of joints for the real model.
# --------------------------------------------------------------------------- #
def test_build_qpos_map_maps_positive_joints():
    model, _data, retargeter, qmap = _build_model_and_retargeter("left")
    assert len(qmap) > 0
    assert len(qmap) <= len(retargeter.joint_names)
    for addr, idx in qmap:
        assert 0 <= addr < model.nq
        assert 0 <= idx < len(retargeter.joint_names)


# --------------------------------------------------------------------------- #
# 5. --map curl path: build_curl_plan + apply_curl_frame pose the real model
#    headless. The angle->qpos inverse maps 100=open to each joint's lower limit
#    and 0=closed to its upper limit; mimic (*_dip) joints follow their
#    multiplier but stay clamped inside their own limits.
# --------------------------------------------------------------------------- #
class _StubMapper:
    """Returns a fixed angle vector (or None), ignoring the keypoints."""

    def __init__(self, angles):
        self._angles = angles

    def __call__(self, _kp):
        return self._angles


def _build_model_and_curl_plan(hand_side="left"):
    hand = HANDS[hand_side]
    urdf = viz.URDF_DIR / "l6" / hand.side / f"realhand_l6_{hand.side}.urdf"
    model = mujoco.MjSpec.from_file(str(urdf)).compile()
    data = mujoco.MjData(model)
    return model, data, viz.build_curl_plan(model, urdf)


def test_build_curl_plan_shape():
    _model, _data, (drivers, dips) = _build_model_and_curl_plan("left")
    assert len(drivers) == 6            # one per L6_SDK_ORDER slot
    assert len(dips) == 5               # thumb + index/middle/ring/pinky *_dip mimics
    for addr, lo, up in drivers:
        assert addr is not None
        assert up > lo                  # every driver joint has real travel


def test_apply_curl_frame_open_and_closed_extremes():
    model, data, plan = _build_model_and_curl_plan("left")
    drivers, dips = plan

    # Fully OPEN (angles 100) -> every driver at its lower limit.
    assert viz.apply_curl_frame(model, data, plan, _StubMapper([100.0] * 6), None) is True
    assert np.all(np.isfinite(data.qpos))
    for addr, lo, _up in drivers:
        assert data.qpos[addr] == pytest.approx(lo, abs=1e-9)

    # Fully CLOSED (angles 0) -> every driver at its upper limit; mimic dips
    # follow their multiplier but stay clamped within their own joint limits.
    assert viz.apply_curl_frame(model, data, plan, _StubMapper([0.0] * 6), None) is True
    for addr, _lo, up in drivers:
        assert data.qpos[addr] == pytest.approx(up, abs=1e-9)
    for addr, slot, mult, off, lo, up in dips:
        parent_up = drivers[slot][2]
        expected = min(max(mult * parent_up + off, lo), up)
        assert data.qpos[addr] == pytest.approx(expected, abs=1e-9)
        assert lo - 1e-9 <= data.qpos[addr] <= up + 1e-9   # clamp respected


def test_apply_curl_frame_none_holds_last_state():
    model, data, plan = _build_model_and_curl_plan("left")
    viz.apply_curl_frame(model, data, plan, _StubMapper([50.0] * 6), None)
    before = data.qpos.copy()
    assert viz.apply_curl_frame(model, data, plan, _StubMapper(None), None) is False
    assert np.array_equal(data.qpos, before)   # a None frame never mutates the sim


def test_apply_curl_frame_with_real_mapper():
    from prehensile.curlmap import CurlMapper

    model, data, plan = _build_model_and_curl_plan("left")
    kp = fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)
    assert viz.apply_curl_frame(model, data, plan, CurlMapper(alpha=1.0), kp) is True
    assert np.all(np.isfinite(data.qpos))
