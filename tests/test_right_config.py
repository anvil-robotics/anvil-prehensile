"""TDD validators for the right-hand config (Task 6).

Pure-synthetic, no hardware/glove/robot: proves ``configs/real_hand_right.yml``
loads and retargets on the right URDF, that it is a byte-for-byte copy of the
left config except for ``urdf_path``, and that ``UDCapSource`` constructs fine
for both ``side="right"`` and ``side="left"`` (right-hand UDCap now has a
validated right rest-pose table + FK mode/chirality, see ``prehensile/fk.py``).
"""

from pathlib import Path

import numpy as np
import yaml

from prehensile.command import qpos_index_map
from prehensile.udcap import UDCapSource
from prehensile.retarget import L6Retargeter

_REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = _REPO_ROOT / "assets" / "realhand_description"
CONFIG_LEFT = _REPO_ROOT / "configs" / "real_hand_left.yml"
CONFIG_RIGHT = _REPO_ROOT / "configs" / "real_hand_right.yml"


def _make_open_hand():
    """A rough flat open hand, (21,3) meters, MediaPipe order (fingers extended +y).

    Same shape as prehensile/offline_check.py's make_open_hand -- a plumbing-check
    synthetic frame, not a validated open/close pose.
    """
    kp = np.zeros((21, 3), dtype=np.float32)
    for base, x in [(1, -0.04), (5, -0.03), (9, -0.01), (13, 0.01), (17, 0.03)]:
        for k in range(4):
            kp[base + k] = [x, 0.03 * (k + 1), 0.0]
    kp[1] = [-0.03, 0.02, 0.01]  # thumb cmc nearer the palm
    return kp


def _make_curled_hand():
    """Open hand with the four fingers curled toward the palm (rough)."""
    kp = _make_open_hand()
    for base in (5, 9, 13, 17):
        kp[base + 1] = [kp[base + 1][0], 0.030, -0.010]
        kp[base + 2] = [kp[base + 2][0], 0.025, -0.030]
        kp[base + 3] = [kp[base + 3][0], 0.015, -0.050]
    return kp


# --------------------------------------------------------------------------- #
# 1. real_hand_right.yml loads and retargets on the right URDF.
# --------------------------------------------------------------------------- #
def test_right_config_loads_and_retargets():
    retargeter = L6Retargeter(CONFIG_RIGHT, URDF_DIR, side="right")

    for kp in (_make_open_hand(), _make_curled_hand()):
        qpos = retargeter.retarget(kp)
        assert qpos is not None
        qpos = np.asarray(qpos)
        assert qpos.ndim == 1
        assert qpos.shape[0] == len(retargeter.joint_names)
        assert np.all(np.isfinite(qpos))

    # command.py's by-name driver-joint map must resolve all 6 slots on the right
    # URDF's joint_names (proves the name-based command map works right-side too).
    index_map = qpos_index_map(retargeter.joint_names)
    assert len(index_map) == 6
    assert all(isinstance(i, (int, np.integer)) for i in index_map)


# --------------------------------------------------------------------------- #
# 2. right config differs from left ONLY in urdf_path.
# --------------------------------------------------------------------------- #
def test_right_config_differs_from_left_only_in_urdf_path():
    left_text = CONFIG_LEFT.read_text()
    right_text = CONFIG_RIGHT.read_text()
    assert left_text != right_text  # sanity: files aren't identical

    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    assert len(left_lines) == len(right_lines)

    diff_lines = [
        (a, b) for a, b in zip(left_lines, right_lines) if a != b
    ]
    assert len(diff_lines) == 1
    left_line, right_line = diff_lines[0]
    assert left_line.strip() == "urdf_path: l6/left/realhand_l6_left.urdf"
    assert right_line.strip() == "urdf_path: l6/right/realhand_l6_right.urdf"

    # Same check on the parsed structures: every key equal except urdf_path.
    left_cfg = yaml.safe_load(left_text)["retargeting"]
    right_cfg = yaml.safe_load(right_text)["retargeting"]
    assert set(left_cfg) == set(right_cfg)
    for key in left_cfg:
        if key == "urdf_path":
            continue
        assert left_cfg[key] == right_cfg[key], f"unexpected difference in {key!r}"
    assert left_cfg["urdf_path"] == "l6/left/realhand_l6_left.urdf"
    assert right_cfg["urdf_path"] == "l6/right/realhand_l6_right.urdf"


# --------------------------------------------------------------------------- #
# 3. UDCap-right is now supported: both sides construct fine.
# --------------------------------------------------------------------------- #
def test_udcap_right_constructs():
    src = UDCapSource(port=0, side="right")
    try:
        assert src.side == "right"
        assert callable(src.poll)
    finally:
        src.close()


def test_udcap_left_still_constructs():
    src = UDCapSource(port=0, side="left")
    try:
        assert src.side == "left"
        assert callable(src.poll)
    finally:
        src.close()
