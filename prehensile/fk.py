"""Forward-kinematics decode: glove per-finger quaternions -> MediaPipe keypoints.

Branch B of the UDCap glove -> RealHand L6 teleop port. The UDCap glove emits 15
per-finger joint quaternions (3 per finger, ``JointsIndex_Quat`` order:
index(0,1,2), middle(3,4,5), ring(6,7,8), pinky(9,10,11), thumb(12,13,14)); the 3
quats of a finger map to its three articulating joints ``_0`` (proximal), ``_1``
(intermediate), ``_2`` (distal). Combined with the rest skeleton from
``prehensile.rest_pose`` (fixed ``meta`` bones + rest bone lengths/orientations)
they define 21 MediaPipe keypoints (wrist-local, meters, MODEL right-handed
frame) that ``prehensile.retarget`` consumes.

Empirical unknown
-----------------
How the glove parameterizes its quaternions - absolute-global vs parent-local,
rest-referenced or not - is NOT knowable without a real recording (see
``select_mode``). Rather than guess, this module implements all four candidate
modes so each satisfies the two FK invariants, plus the discovery machinery to
pick the physically-correct one from a real open-hand + fist recording. Until
that recording exists, ``FK_MODE`` stays at a documented default and
``OUTPUT_X_FLIP`` stays ``False``.

Two invariants (proven in tests/test_fk.py, synthetic data only)
  1. Each mode's "no rotation" sentinel input reproduces the rest pose exactly.
  2. Every mode preserves all segment lengths for arbitrary rotations.

Public surface (later tasks depend on these):
  MODES                 the four candidate mode names.
  keypoints_from_quats  (15,4) XYZW quats + mode -> (21,3) float32 keypoints.
  identity_quats        a mode's "no rotation" sentinel (15,4) quat stream.
  select_mode           discovery tool: pick the mode from open/fist recordings.
  FK_MODE               the pending default mode (see caveat above).
  OUTPUT_X_FLIP         single documented handedness switch (default False).
"""

from __future__ import annotations

import numpy as np

from prehensile.rest_pose import (
    MEDIAPIPE_NODES,
    MEDIAPIPE_NODES_BY_SIDE,
    REST_POS_CM,
    REST_POS_CM_BY_SIDE,
    REST_QUAT,
    REST_QUAT_BY_SIDE,
)

# --------------------------------------------------------------------------- #
# Quaternion helpers (XYZW throughout, to match rest_pose). numpy-only so this
# module is self-contained (no scipy dependency). Verified against
# scipy.spatial.transform.Rotation as an independent oracle during development.
# --------------------------------------------------------------------------- #


def quat_normalize(q) -> np.ndarray:
    """Return ``q`` (XYZW) scaled to unit norm."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_conj(q) -> np.ndarray:
    """Conjugate of a quaternion (XYZW); equals the inverse for a unit quat."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.array([-x, -y, -z, w])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product ``a * b`` (XYZW); ``R(a*b) == R(a) @ R(b)``."""
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_to_mat(q) -> np.ndarray:
    """Unit-normalized quaternion (XYZW) -> (3,3) rotation matrix."""
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------- #
# Public constants / switches.
# --------------------------------------------------------------------------- #
MODES: list[str] = ["global_raw", "global_rest_ref", "local_chain", "local_rest_ref"]

# Per-side FK mode/handedness switch. Both LEFT and RIGHT are now CONFIRMED
# (see below).
#
# LEFT: CONFIRMED 2026-07-09 by tools/discover_fk_mode.py on real left-glove
# recordings (tests/fixtures/quat_{open,fist}_3s.bin): local_rest_ref was the SOLE
# mode passing both gates (open_err 0.0071 m decodes the open hand back to rest;
# fist_curl 0.432 genuinely curls the fingertips). global_rest_ref decoded the open
# hand well but failed to curl on a fist (0.986), so it was the wrong default.
#
# RIGHT: CONFIRMED 2026-07-23 by tools/discover_fk_mode.py --side right on real
# right-glove recordings (tests/fixtures/quat_{open,fist}_right_3s.bin):
# local_rest_ref passed both gates (open_err 0.0070 m, fist_curl 0.422); chirality
# right-correct so OUTPUT_X_FLIP=False.
FK_MODE_BY_SIDE: dict[str, str] = {
    "left": "local_rest_ref",
    "right": "local_rest_ref",
}

# Single documented handedness switch, per side. rest_pose is a right-handed MODEL
# frame and downstream retargeting wants right-handed, so we do NOT x-negate here
# (that is the glove's left-handed Unity WIRE frame).
#
# LEFT: CONFIRMED False 2026-07-09: the real open-hand recording decoded with a
# negative chirality sign (left-correct, not mirrored) under local_rest_ref.
# Chirality tests assert the sign for this value.
#
# RIGHT: CONFIRMED False 2026-07-23 by tools/discover_fk_mode.py --side right on
# real right-glove recordings (tests/fixtures/quat_{open,fist}_right_3s.bin): the
# mean open-hand recording decoded right-correct (unmirrored) under
# local_rest_ref, so no x-flip is needed.
OUTPUT_X_FLIP_BY_SIDE: dict[str, bool] = {
    "left": False,
    "right": False,
}

# Backward-compat scalar aliases (LEFT values, unchanged). Existing importers of
# the bare ``FK_MODE``/``OUTPUT_X_FLIP`` (prehensile/udcap.py, several tests) keep
# working exactly as before -- they only ever operated on the left glove.
FK_MODE: str = FK_MODE_BY_SIDE["left"]
OUTPUT_X_FLIP: bool = OUTPUT_X_FLIP_BY_SIDE["left"]

# --------------------------------------------------------------------------- #
# Kinematic chains + JointsIndex_Quat mapping. Each non-thumb finger is rooted at
# the wrist through its fixed ``meta`` bone; the thumb hangs directly off the
# wrist. The (i0,i1,i2) tuple is the finger's quat-stream indices for (_0,_1,_2).
#
# LEFT (``_FINGERS``) is UNCHANGED from before this module became side-aware.
# RIGHT (``_FINGERS_R``) is the ``_r``-suffixed analogue (same chain shape, same
# quat-slot indices -- the glove's JointsIndex_Quat layout is per-finger, not
# per-side). Both are exposed together via ``_FINGERS_BY_SIDE``.
# --------------------------------------------------------------------------- #
_FINGERS: list[tuple[list[str], tuple[int, int, int]]] = [
    (
        ["wrist_l", "finger_index_meta_l", "finger_index_0_l",
         "finger_index_1_l", "finger_index_2_l", "finger_index_l_end"],
        (0, 1, 2),
    ),
    (
        ["wrist_l", "finger_middle_meta_l", "finger_middle_0_l",
         "finger_middle_1_l", "finger_middle_2_l", "finger_middle_l_end"],
        (3, 4, 5),
    ),
    (
        ["wrist_l", "finger_ring_meta_l", "finger_ring_0_l",
         "finger_ring_1_l", "finger_ring_2_l", "finger_ring_l_end"],
        (6, 7, 8),
    ),
    (
        ["wrist_l", "finger_pinky_meta_l", "finger_pinky_0_l",
         "finger_pinky_1_l", "finger_pinky_2_l", "finger_pinky_l_end"],
        (9, 10, 11),
    ),
    (
        ["wrist_l", "finger_thumb_0_l", "finger_thumb_1_l",
         "finger_thumb_2_l", "finger_thumb_l_end"],
        (12, 13, 14),
    ),
]

_FINGERS_R: list[tuple[list[str], tuple[int, int, int]]] = [
    (
        ["wrist_r", "finger_index_meta_r", "finger_index_0_r",
         "finger_index_1_r", "finger_index_2_r", "finger_index_r_end"],
        (0, 1, 2),
    ),
    (
        ["wrist_r", "finger_middle_meta_r", "finger_middle_0_r",
         "finger_middle_1_r", "finger_middle_2_r", "finger_middle_r_end"],
        (3, 4, 5),
    ),
    (
        ["wrist_r", "finger_ring_meta_r", "finger_ring_0_r",
         "finger_ring_1_r", "finger_ring_2_r", "finger_ring_r_end"],
        (6, 7, 8),
    ),
    (
        ["wrist_r", "finger_pinky_meta_r", "finger_pinky_0_r",
         "finger_pinky_1_r", "finger_pinky_2_r", "finger_pinky_r_end"],
        (9, 10, 11),
    ),
    (
        ["wrist_r", "finger_thumb_0_r", "finger_thumb_1_r",
         "finger_thumb_2_r", "finger_thumb_r_end"],
        (12, 13, 14),
    ),
]

_FINGERS_BY_SIDE: dict[str, list[tuple[list[str], tuple[int, int, int]]]] = {
    "left": _FINGERS,
    "right": _FINGERS_R,
}

# side -> node-name suffix ("l"/"r"), used to build the wrist seed node name and
# the "_meta_{s}"/"_{s}_end" fixed-node checks in ``_current_orientation``.
_SUFFIX_BY_SIDE: dict[str, str] = {"left": "l", "right": "r"}

# node-name suffix -> articulating-joint slot (0=proximal, 1=intermediate, 2=distal)
_SLOT_BY_SUFFIX = {"_0_l": 0, "_1_l": 1, "_2_l": 2}
_SLOT_BY_SUFFIX_R = {"_0_r": 0, "_1_r": 1, "_2_r": 2}
_SLOT_BY_SUFFIX_BY_SIDE: dict[str, dict[str, int]] = {
    "left": _SLOT_BY_SUFFIX,
    "right": _SLOT_BY_SUFFIX_R,
}

# Precomputed rest rotation matrices, R_rest[node] = quat_to_mat(REST_QUAT[node]).
_R_REST: dict[str, np.ndarray] = {n: quat_to_mat(q) for n, q in REST_QUAT.items()}
_R_REST_R: dict[str, np.ndarray] = {n: quat_to_mat(q) for n, q in REST_QUAT_BY_SIDE["right"].items()}
_R_REST_BY_SIDE: dict[str, dict[str, np.ndarray]] = {
    "left": _R_REST,
    "right": _R_REST_R,
}


def _joint_slot(node: str, side: str = "left") -> int | None:
    """Articulating-joint slot for ``node`` (0/1/2), or None if not articulating."""
    for suffix, slot in _SLOT_BY_SUFFIX_BY_SIDE[side].items():
        if node.endswith(suffix):
            return slot
    return None


def _apply_x_flip(kp: np.ndarray) -> np.ndarray:
    """Negate the x component of every keypoint (right<->left handedness flip)."""
    kp[:, 0] *= -1.0
    return kp


# --------------------------------------------------------------------------- #
# Core FK.
# --------------------------------------------------------------------------- #


def _current_orientation(
    mode: str,
    child: str,
    parent: str,
    quats: np.ndarray,
    qidx: tuple[int, int, int],
    R_cur: dict[str, np.ndarray],
    side: str = "left",
) -> np.ndarray:
    """Current global orientation R_cur[child] for one node, per mode.

    ``meta`` and ``_end`` are fixed / inherited and do not consume a glove quat;
    the three articulating joints (_0,_1,_2) combine their quat Qi with the rest
    orientations per the mode definition. Assumes R_cur[parent] is already set.
    """
    suffix = _SUFFIX_BY_SIDE[side]
    R_rest = _R_REST_BY_SIDE[side]
    if child.endswith(f"_meta_{suffix}"):
        return R_rest[child]  # fixed metacarpal bone (no glove quat)
    if child.endswith(f"_{suffix}_end"):
        return R_cur[parent]  # fingertip inherits its parent (_2) orientation

    slot = _joint_slot(child, side)
    assert slot is not None, child
    Qi = quat_to_mat(quats[qidx[slot]])

    if mode == "global_raw":
        # quat is the joint's absolute global orientation
        return Qi
    if mode == "global_rest_ref":
        # quat is a global delta from the rest orientation (pre-multiplied)
        return Qi @ R_rest[child]
    if mode == "local_chain":
        # quat is the joint's parent-relative orientation, accumulated
        return R_cur[parent] @ Qi
    if mode == "local_rest_ref":
        # rest local transform, then the measured local delta
        rest_local = R_rest[parent].T @ R_rest[child]
        return R_cur[parent] @ rest_local @ Qi
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def keypoints_from_quats(joint_quats, mode: str, side: str = "left") -> np.ndarray:
    """(15,4) XYZW glove quats + mode + side -> (21,3) float32 MediaPipe keypoints.

    Output is wrist-local (wrist subtracted), meters (cm/100), MediaPipe landmark
    order, in the MODEL right-handed frame. If ``OUTPUT_X_FLIP_BY_SIDE[side]`` is
    True the x component is negated (the single documented handedness switch).

    The shared forward-position walk (identical across all four modes) is::

        p_cur[wrist] = p_rest[wrist]
        for each child down a chain (parent p):
            bone_local   = R_rest[p].T @ (p_rest[child] - p_rest[p])
            p_cur[child] = p_cur[parent] + R_cur[p] @ bone_local

    which reproduces the rest pose exactly when R_cur == R_rest everywhere
    (INVARIANT 1) and preserves every bone length for any rotations (INVARIANT 2).
    ``side`` selects which rest skeleton / chain tables / wrist node are used
    (default ``"left"``, matching all pre-existing call sites exactly).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    quats = np.asarray(joint_quats, dtype=np.float64)
    if quats.shape != (15, 4):
        raise ValueError(f"joint_quats must be (15,4) XYZW, got {quats.shape}")

    suffix = _SUFFIX_BY_SIDE[side]
    wrist_node = f"wrist_{suffix}"
    R_rest = _R_REST_BY_SIDE[side]
    rest_pos = REST_POS_CM_BY_SIDE[side]
    mp_nodes = MEDIAPIPE_NODES_BY_SIDE[side]

    R_cur: dict[str, np.ndarray] = {wrist_node: R_rest[wrist_node]}
    p_cur: dict[str, np.ndarray] = {wrist_node: rest_pos[wrist_node].copy()}

    for chain, qidx in _FINGERS_BY_SIDE[side]:
        for j in range(1, len(chain)):
            parent, child = chain[j - 1], chain[j]
            R_cur[child] = _current_orientation(mode, child, parent, quats, qidx, R_cur, side)
            bone_local = R_rest[parent].T @ (rest_pos[child] - rest_pos[parent])
            p_cur[child] = p_cur[parent] + R_cur[parent] @ bone_local

    out = np.array([p_cur[n] - p_cur[wrist_node] for n in mp_nodes]) / 100.0
    if OUTPUT_X_FLIP_BY_SIDE[side]:
        out = _apply_x_flip(out)
    return out.astype(np.float32)


def identity_quats(mode: str, side: str = "left") -> np.ndarray:
    """The mode's "no rotation" sentinel (15,4) XYZW stream, for ``side``.

    Feeding this to ``keypoints_from_quats(_, mode, side)`` reproduces the rest
    pose exactly (INVARIANT 1). It is also the neutral/open-hand input for that
    mode. ``side`` defaults to ``"left"``, matching all pre-existing call sites.

      * global_raw     -> the rest GLOBAL quats at each joint.
      * global_rest_ref-> identity (zero global delta).
      * local_chain    -> the rest LOCAL quats (R_rest[parent].T @ R_rest[joint]).
      * local_rest_ref -> identity (zero local delta).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    rest_quat = REST_QUAT_BY_SIDE[side]
    q = np.zeros((15, 4), dtype=np.float64)
    for chain, qidx in _FINGERS_BY_SIDE[side]:
        for j in range(1, len(chain)):
            parent, child = chain[j - 1], chain[j]
            slot = _joint_slot(child, side)
            if slot is None:
                continue
            if mode in ("global_rest_ref", "local_rest_ref"):
                q[qidx[slot]] = np.array([0.0, 0.0, 0.0, 1.0])
            elif mode == "global_raw":
                q[qidx[slot]] = rest_quat[child]
            else:  # local_chain: rest local quat = conj(q_parent) * q_child
                q[qidx[slot]] = quat_mul(quat_conj(rest_quat[parent]), rest_quat[child])
    return q


# --------------------------------------------------------------------------- #
# Discovery machinery.
# --------------------------------------------------------------------------- #

# Rest keypoints (wrist-local meters), the open-hand reference for select_mode,
# per side. LEFT (``_REST_KP``) is UNCHANGED; RIGHT (``_REST_KP_R``) is its
# right-hand analogue.
_REST_KP: np.ndarray = (
    np.array([REST_POS_CM[n] - REST_POS_CM["wrist_l"] for n in MEDIAPIPE_NODES]) / 100.0
)
_REST_KP_R: np.ndarray = (
    np.array(
        [REST_POS_CM_BY_SIDE["right"][n] - REST_POS_CM_BY_SIDE["right"]["wrist_r"]
         for n in MEDIAPIPE_NODES_BY_SIDE["right"]]
    ) / 100.0
)
_REST_KP_BY_SIDE: dict[str, np.ndarray] = {"left": _REST_KP, "right": _REST_KP_R}

# Fingertip landmark indices (thumb, index, middle, ring, pinky).
_TIP_INDICES = (4, 8, 12, 16, 20)


def _wrist_to_tip(kp: np.ndarray) -> np.ndarray:
    """Per-finger wrist->fingertip distances (m) for a (21,3) keypoint array."""
    return np.array([np.linalg.norm(kp[t] - kp[0]) for t in _TIP_INDICES])


def select_mode(open_frames, fist_frames, side: str = "left") -> tuple[str | None, dict]:
    """Pick the physically-correct FK mode from real open-hand + fist recordings.

    ``open_frames`` / ``fist_frames`` are lists of (15,4) XYZW quat arrays (an
    open hand held flat; a closed fist) for ``side`` (default ``"left"``,
    matching all pre-existing call sites). For each candidate mode:

      * open_err  - mean L2 distance (m) over the 21 landmarks between the
        mode's decode of the mean open frame and the rest pose. A correct mode
        decodes an open hand back to (near) the rest skeleton.
      * fist_curl - mean over fingers, over fist frames, of the ratio
        (wrist->tip in fist) / (wrist->tip in that mode's open decode). A correct
        mode curls the fingertips toward the wrist, so the ratio drops.

    Winner = the mode with open_err < 0.01 m AND the smallest fist_curl, which
    must be < 0.65. Returns ``(best_mode_or_None, per_mode_metrics)``.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    rest_kp = _REST_KP_BY_SIDE[side]
    open_list = [np.asarray(f, dtype=np.float64) for f in open_frames]
    fist_list = [np.asarray(f, dtype=np.float64) for f in fist_frames]
    mean_open = np.mean(open_list, axis=0)

    metrics: dict[str, dict[str, float]] = {}
    for mode in MODES:
        kp_open = keypoints_from_quats(mean_open, mode, side=side)
        open_err = float(np.mean(np.linalg.norm(kp_open - rest_kp, axis=1)))
        open_tips = _wrist_to_tip(kp_open)

        ratios = [
            float(np.mean(_wrist_to_tip(keypoints_from_quats(f, mode, side=side)) / open_tips))
            for f in fist_list
        ]
        fist_curl = float(np.mean(ratios)) if ratios else float("nan")
        metrics[mode] = {"open_err": open_err, "fist_curl": fist_curl}

    passing = {
        m: v for m, v in metrics.items()
        if v["open_err"] < 0.01 and v["fist_curl"] < 0.65
    }
    best = min(passing, key=lambda m: passing[m]["fist_curl"]) if passing else None
    return best, metrics
