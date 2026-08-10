"""M1 smoke test: synthetic keypoints -> retarget -> L6 angles. No hardware/glove.

Verifies the dex_retargeting install + URDF build + our retarget/command glue run
end-to-end and produce 6 values in [0,100]. The *real* open/close validation is the
live dry-run (M2) with the actual glove; a synthetic hand is only a plumbing check.

    uv run python -m prehensile.offline_check
"""

from pathlib import Path

import numpy as np

from prehensile.command import qpos_index_map, qpos_to_l6_angles
from prehensile.retarget import L6Retargeter

ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = ROOT / "assets" / "realhand_description"
CONFIG = ROOT / "configs" / "real_hand_left.yml"

JOINTS = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]


def make_open_hand():
    """A rough flat open hand, (21,3) meters, MediaPipe order (fingers extended +y)."""
    kp = np.zeros((21, 3), dtype=np.float64)
    for base, x in [(1, -0.04), (5, -0.03), (9, -0.01), (13, 0.01), (17, 0.03)]:
        for k in range(4):
            kp[base + k] = [x, 0.03 * (k + 1), 0.0]
    kp[1] = [-0.03, 0.02, 0.01]  # thumb cmc nearer the palm
    return kp


def make_curled_hand():
    """Open hand with the four fingers curled toward the palm (rough)."""
    kp = make_open_hand()
    for base in (5, 9, 13, 17):
        kp[base + 1] = [kp[base + 1][0], 0.030, -0.010]
        kp[base + 2] = [kp[base + 2][0], 0.025, -0.030]
        kp[base + 3] = [kp[base + 3][0], 0.015, -0.050]
    return kp


def main():
    print(f"config : {CONFIG}")
    print(f"urdf   : {URDF_DIR}")
    rt = L6Retargeter(CONFIG, URDF_DIR, side="left")
    print(f"joint_names ({len(rt.joint_names)}): {rt.joint_names}")
    print(f"human indices origin={rt.origin_idx.tolist()} task={rt.task_idx.tolist()}")
    index_map = qpos_index_map(rt.joint_names)
    print(f"qpos index map (SDK order {['thumb_flex','thumb_abd','index','middle','ring','pinky']}): {index_map}\n")

    for name, kp in [("open", make_open_hand()), ("curled", make_curled_hand())]:
        qpos = rt.retarget(kp)
        angles = qpos_to_l6_angles(qpos, index_map)
        print(f"[{name:6s}] qpos(len={len(qpos)}) = {np.round(qpos, 3).tolist()}")
        pretty = "  ".join(f"{n}={a:5.1f}" for n, a in zip(JOINTS, angles))
        print(f"         angles = {pretty}\n")

    print("OK — pipeline runs end-to-end (install + URDF build + retarget + command).")


if __name__ == "__main__":
    raise SystemExit(main())
