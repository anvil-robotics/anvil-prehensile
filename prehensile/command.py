"""Map a retargeted qpos (radians) to a REALHAND L6 6-value command (0-100).

The dex_retargeting vector optimizer returns a full-DOF qpos in the L6 URDF's
pinocchio joint order (radians). We pick the 6 *driver* joints BY NAME (robust to
the pinocchio ordering, which differs from RealHand's reference) and normalize each
to 0-100 for ``hand.angle.set_angles`` (verified realhand==0.5.3 L6Angle order):

    [thumb_flex, thumb_abd, index, middle, ring, pinky]     100 = open, 0 = closed

Do NOT hardcode qpos indices: the installed URDF's joint order is
[index_mcp_pitch, index_dip, middle_mcp_pitch, middle_dip, pinky_mcp_pitch,
 pinky_dip, ring_mcp_pitch, ring_dip, thunb_cmc_roll, thumb_cmc_pitch, thumb_dip],
which does NOT match RealHand's reference indices. The *_dip joints are mimic and
are not commanded.
"""

import numpy as np

L6_OPEN = 100.0
# Flexion angle (rad) mapped to a fully-closed finger. RealHand uses pi/2.
L6_RETARGET_MAX_RAD = np.pi / 2.0

# realhand set_angles slot order:
L6_SDK_ORDER = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]

# our finger label -> the driver joint name in the L6 URDF (note the 'thunb' typo,
# which is present in both the URDF and the retarget config).
L6_DRIVER_JOINTS = {
    "thumb_flex": "thumb_cmc_pitch",
    "thumb_abd": "thunb_cmc_roll",
    "index": "index_mcp_pitch",
    "middle": "middle_mcp_pitch",
    "ring": "ring_mcp_pitch",
    "pinky": "pinky_mcp_pitch",
}


def qpos_index_map(joint_names):
    """Build the [qpos index per SDK slot] map from the retargeter's joint_names.

    joint_names comes from ``L6Retargeter.joint_names`` (pinocchio order). Returns a
    length-6 list of qpos indices in L6_SDK_ORDER.
    """
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    missing = [j for j in L6_DRIVER_JOINTS.values() if j not in name_to_idx]
    if missing:
        raise KeyError(f"driver joints not found in URDF joint_names: {missing}")
    return [name_to_idx[L6_DRIVER_JOINTS[slot]] for slot in L6_SDK_ORDER]


def qpos_to_l6_angles(qpos, index_map, invert_flex: bool = True):
    """qpos (radians) + index_map -> 6 L6 angles in [0,100], in SDK slot order.

    Different glove apps and the L6 can use OPPOSITE open/close conventions for the
    flexion channels (thumb_flex, index, middle, ring, pinky). Both senses have been
    confirmed on real hardware for the two gloves this project supports -- this is
    NOT a bug, it's a real per-glove difference that reduces to one boolean:

    - ``invert_flex=True`` (default, UDCap): the UDCap app reads 0=open .. 100=closed
      while the L6 command is 100=open .. 0=closed. Opening the operator's hand
      closed the robot on live hardware, so the flexion map is inverted to
      ``normalized*100`` to compensate.
    - ``invert_flex=False`` (Wuji): the Wuji glove's flexion sense already matches
      the L6 hand-native convention, so no inversion is needed and the flexion map
      is ``(1-normalized)*100``.

    thumb_abd (thunb_cmc_roll) is an ABDUCTION axis, not an open/close one, so it is
    UNAFFECTED by ``invert_flex`` in either mode: it is always the hand-native
    ``(1-normalized)*100``. Applying the flexion inversion to it reversed the thumb
    spread on live hardware (spreading the operator's thumb tucked the robot's), so
    it is always mapped back to the native sense regardless of glove.
    """
    qpos = np.asarray(qpos, dtype=float)
    flex_rad = np.array([qpos[i] for i in index_map])
    normalized = np.clip(flex_rad / L6_RETARGET_MAX_RAD, 0.0, 1.0)
    angles = normalized * L6_OPEN if invert_flex else (1.0 - normalized) * L6_OPEN
    # The abduction axis is independent of open/close and is always left on the
    # hand-native mapping, regardless of invert_flex.
    abd = L6_SDK_ORDER.index("thumb_abd")
    angles[abd] = (1.0 - normalized[abd]) * L6_OPEN
    return angles.tolist()
