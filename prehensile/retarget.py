"""(21,3) MediaPipe keypoints -> L6 retargeted qpos (radians) via dex_retargeting.

Replicates RealHand's DexHandTracker.retarget on the *upstream* dex_retargeting lib:
  1. estimate a wrist frame from the keypoints (MANO convention),
  2. rotate the keypoints into it (@ OPERATOR2MANO_LEFT),
  3. build the origin->task difference vectors the vector optimizer expects
     (from the config's target_link_human_indices),
  4. optimize -> full-DOF qpos (11 for the L6 URDF), radians, pinocchio joint order.

`estimate_frame_from_hand_points` is copied verbatim from dexsuite/dex-retargeting's
single_hand_detector.py -- that module isn't shipped in the installed wheel (it pulls
in mediapipe), but the frame math itself is pure numpy.
"""

import numpy as np
from dex_retargeting.constants import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT
from dex_retargeting.retargeting_config import RetargetingConfig


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Wrist coordinate frame (orientation only) from (21,3) keypoints, MANO convention.

    Verbatim from dexsuite/dex-retargeting single_hand_detector.py. Uses landmarks
    [0, 5, 9] = wrist, index_mcp, middle_mcp.
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)
    normal = v[2, :]

    # Gram-Schmidt orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # Assume the vector from pinky to index aligns with +z in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


class L6Retargeter:
    """Vector retargeter for the REALHAND L6 (left or right)."""

    def __init__(self, config_path, urdf_dir, side="left"):
        # set_default_urdf_dir must be set before build() so the config's relative
        # urdf_path (l6/left/realhand_l6_left.urdf) resolves against the assets tree.
        RetargetingConfig.set_default_urdf_dir(str(urdf_dir))
        self.retargeting = RetargetingConfig.load_from_file(str(config_path)).build()
        self.operator2mano = OPERATOR2MANO_LEFT if side == "left" else OPERATOR2MANO_RIGHT

        # target_link_human_indices is (2, N): row 0 = origin landmarks, row 1 = task.
        idx = np.asarray(self.retargeting.optimizer.target_link_human_indices)
        self.origin_idx = idx[0]
        self.task_idx = idx[1]

    @property
    def joint_names(self):
        return self.retargeting.joint_names

    def retarget(self, keypoints):
        """(21,3) MediaPipe keypoints (meters) -> qpos (radians). None if malformed."""
        kp = np.asarray(keypoints, dtype=np.float64)
        if kp.shape != (21, 3):
            return None
        wrist_rot = estimate_frame_from_hand_points(kp)
        joint_pos = kp @ wrist_rot @ self.operator2mano
        ref_value = joint_pos[self.task_idx, :] - joint_pos[self.origin_idx, :]
        return self.retargeting.retarget(ref_value)
