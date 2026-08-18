"""TDD validators for prehensile/fk.py (Branch B forward kinematics).

SYNTHETIC-ONLY. The physically-correct interpretation of the glove quaternions
(mode + handedness) is an empirical unknown that can only be resolved with a real
glove recording (the user is remote; no glove available). These tests therefore
prove exactly what synthetic data can prove:

  * INVARIANT 1 - each mode's "no rotation" sentinel reproduces the rest pose,
  * INVARIANT 2 - every mode preserves all 20 segment lengths for any rotation,
  * the output contract (shape/dtype/wrist-origin/finite),
  * the chirality sign of the rest pose and that OUTPUT_X_FLIP inverts it,
  * single-joint isolation, and
  * the discovery machinery (select_mode) via a synthetic open/fist round-trip.

The real-glove fixture test is DEFERRED behind a skipif and documented below.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from prehensile import fk
from prehensile.fk import (
    FK_MODE,
    MODES,
    OUTPUT_X_FLIP,
    identity_quats,
    keypoints_from_quats,
    select_mode,
)
from prehensile.rest_pose import MEDIAPIPE_NODES, REST_POS_CM

# Rest-pose keypoints (wrist-local, meters) - the FK target for identity input.
REST_KP = (
    np.array([REST_POS_CM[n] - REST_POS_CM["wrist_l"] for n in MEDIAPIPE_NODES])
    / 100.0
)

# The 20 MediaPipe segments: wrist->mcp for each finger + consecutive intra-finger
# pairs (thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20).
MP_SEGMENTS: list[tuple[int, int]] = []
for _base in (1, 5, 9, 13, 17):
    MP_SEGMENTS.append((0, _base))
    MP_SEGMENTS += [(_base + k, _base + k + 1) for k in range(3)]

# JointsIndex_Quat proximal (_0) slot per finger: index, middle, ring, pinky, thumb.
_PROXIMAL_SLOTS = (0, 3, 6, 9, 12)


def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    """XYZW quaternion for a rotation of ``angle_rad`` about ``axis`` (any length)."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle_rad / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle_rad / 2.0)])


# --------------------------------------------------------------------------- #
# INVARIANT 1: each mode's sentinel input reproduces the rest pose exactly.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_identity_reproduces_rest_pose(mode):
    kp = keypoints_from_quats(identity_quats(mode), mode)
    np.testing.assert_allclose(kp, REST_KP, atol=1e-5)


# --------------------------------------------------------------------------- #
# INVARIANT 2: arbitrary rotations preserve every segment length (rigid bones).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_segment_lengths_preserved(mode):
    rng = np.random.default_rng(1234)
    q = rng.standard_normal((15, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    kp = keypoints_from_quats(q, mode)
    for a, b in MP_SEGMENTS:
        got = float(np.linalg.norm(kp[b] - kp[a]))
        want = float(np.linalg.norm(REST_KP[b] - REST_KP[a]))
        assert abs(got - want) < 1e-6, f"{mode} segment {a}->{b}: {got} vs {want}"


# --------------------------------------------------------------------------- #
# Output contract: (21,3) float32, wrist at origin, all finite.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
def test_output_contract(mode):
    kp = keypoints_from_quats(identity_quats(mode), mode)
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_array_equal(kp[0], np.zeros(3, dtype=np.float32))
    assert np.all(np.isfinite(kp))


# --------------------------------------------------------------------------- #
# Chirality: the rest pose has a fixed handedness sign; OUTPUT_X_FLIP inverts it.
# Observed sign (model right-handed frame, OUTPUT_X_FLIP=False) is NEGATIVE.
# --------------------------------------------------------------------------- #
def _chirality(kp) -> float:
    return float(np.dot(np.cross(kp[5] - kp[0], kp[9] - kp[0]), kp[4] - kp[0]))


def test_chirality_sign_and_flip():
    kp = keypoints_from_quats(identity_quats("global_rest_ref"), "global_rest_ref")
    c = _chirality(kp)
    assert c != 0.0
    # Guard: the hard-coded sign below is asserted for the default switch value.
    assert OUTPUT_X_FLIP is False
    assert c < 0.0, f"expected negative chirality for OUTPUT_X_FLIP=False, got {c}"
    # Toggling the single x-flip switch must invert the handedness sign.
    kp_flipped = fk._apply_x_flip(kp.copy())
    assert _chirality(kp_flipped) > 0.0


# --------------------------------------------------------------------------- #
# Synthetic single-joint bend: rotating only the index proximal joint moves the
# index fingertip but leaves the pinky fingertip at rest (joint isolation).
# --------------------------------------------------------------------------- #
def test_single_joint_bend_isolation():
    mode = "global_rest_ref"
    # A palm-normal-ish axis derived from the rest pose (wrist/index/pinky plane).
    normal = np.cross(REST_KP[5] - REST_KP[0], REST_KP[17] - REST_KP[0])
    q = identity_quats(mode)
    q[0] = _axis_angle_quat(normal, math.radians(45.0))  # index proximal (_0) only
    kp = keypoints_from_quats(q, mode)
    assert np.linalg.norm(kp[8] - REST_KP[8]) > 1e-2      # index tip displaced
    assert np.linalg.norm(kp[20] - REST_KP[20]) < 1e-6    # pinky tip unchanged


# --------------------------------------------------------------------------- #
# Discovery round-trip: synthesize an open + fist stream for a known ground-truth
# mode and prove select_mode recovers it. Ground truth = local_rest_ref because a
# genuine anatomical curl is expressed there as the SAME per-joint local flexion
# on every joint (identity sentinel + a fixed local delta) - which needs no FK
# internals to construct, so the test's fist is built independently of fk.py's
# forward walk. Fed to the other modes the same quats reinterpret to different
# poses, so only local_rest_ref passes both discovery gates.
# --------------------------------------------------------------------------- #
def test_select_mode_discovery_roundtrip():
    ground_truth = "local_rest_ref"
    open_stream = identity_quats(ground_truth)  # all-identity sentinel

    # ~75 deg flexion about each joint's local z, on proximal(_0)+intermediate(_1).
    flex = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(75.0))
    fist = open_stream.copy()
    for base in _PROXIMAL_SLOTS:
        fist[base] = flex          # _0 proximal
        fist[base + 1] = flex      # _1 intermediate

    best, metrics = select_mode([open_stream], [fist])

    assert set(metrics) == set(MODES)
    assert best == ground_truth
    assert metrics[ground_truth]["open_err"] < 0.01
    assert metrics[ground_truth]["fist_curl"] < 0.65
    # The winner must be a strict argmin among modes that pass the open gate.
    for m in MODES:
        if m != ground_truth and metrics[m]["open_err"] < 0.01:
            assert metrics[m]["fist_curl"] > metrics[ground_truth]["fist_curl"]


# --------------------------------------------------------------------------- #
# Sanity on the public switches/defaults (kept honest: FK_MODE is a documented
# PENDING default, not an empirically-resolved value).
# --------------------------------------------------------------------------- #
def test_public_defaults():
    assert FK_MODE in MODES
    assert OUTPUT_X_FLIP is False
    assert MODES == ["global_raw", "global_rest_ref", "local_chain", "local_rest_ref"]


# --------------------------------------------------------------------------- #
# DEFERRED real-glove validation. The recorder (tools/record_stream.py) writes
# length-prefixed raw protobuf datagrams; decoding them to (N,15,4) XYZW streams
# and running select_mode on a real open-hand + fist recording is what will
# actually resolve FK_MODE and OUTPUT_X_FLIP. Skipped until the recordings exist.
# --------------------------------------------------------------------------- #
_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE_OPEN = _FIXTURES / "quat_open_3s.bin"
_FIXTURE_FIST = _FIXTURES / "quat_fist_3s.bin"
_HAVE_FIXTURES = _FIXTURE_OPEN.exists() and _FIXTURE_FIST.exists()


@pytest.mark.skipif(
    not _HAVE_FIXTURES,
    reason="real glove recordings (tests/fixtures/quat_*_3s.bin) not available",
)
def test_live_fixture_discovery():
    """Regression guard on the committed real left-glove recordings.

    Confirmed 2026-07-09 via tools/discover_fk_mode.py: select_mode recovers
    local_rest_ref as the sole mode passing both gates, and the open hand has a
    left-correct (negative) chirality sign. The recordings are JSON datagrams;
    _extract_quats auto-detects the wire format (same parser teleop uses).
    """
    from prehensile.udcap import _extract_quats, iter_datagrams

    def frames(path):
        out = []
        for d in iter_datagrams(path):
            result = _extract_quats(d, "left")
            if result is not None:
                out.append(result[0])  # (quats, aButton) -> quats
        return out

    open_frames = frames(_FIXTURE_OPEN)
    fist_frames = frames(_FIXTURE_FIST)
    assert len(open_frames) > 100 and len(fist_frames) > 100

    best, metrics = select_mode(open_frames, fist_frames)
    assert best == "local_rest_ref", f"expected local_rest_ref, got {best!r}: {metrics}"
    assert metrics[best]["open_err"] < 0.01
    assert metrics[best]["fist_curl"] < 0.65

    # Mean open hand decodes left-correct (negative triple product) under the
    # winner, so OUTPUT_X_FLIP=False is right (no mirror).
    kp = keypoints_from_quats(np.mean(np.asarray(open_frames), axis=0), best)
    v5, v9, v4 = kp[5] - kp[0], kp[9] - kp[0], kp[4] - kp[0]
    assert float(np.dot(np.cross(v5, v9), v4)) < 0


_FIXTURE_OPEN_R = _FIXTURES / "quat_open_right_3s.bin"
_FIXTURE_FIST_R = _FIXTURES / "quat_fist_right_3s.bin"
_HAVE_FIXTURES_R = _FIXTURE_OPEN_R.exists() and _FIXTURE_FIST_R.exists()


@pytest.mark.skipif(
    not _HAVE_FIXTURES_R,
    reason="real glove recordings (tests/fixtures/quat_*_right_3s.bin) not available",
)
def test_live_fixture_discovery_right():
    """Regression guard on the committed real right-glove recordings.

    Confirmed 2026-07-23 via tools/discover_fk_mode.py --side right: select_mode
    recovers local_rest_ref as the sole mode passing both gates (open_err 0.0070
    m, fist_curl 0.422), and the open hand has a right-correct chirality sign
    (so OUTPUT_X_FLIP_BY_SIDE["right"]=False is right, no mirror). The
    recordings are JSON datagrams; _extract_quats auto-detects the wire format
    (same parser teleop uses).
    """
    from prehensile.udcap import _extract_quats, iter_datagrams

    def frames(path):
        out = []
        for d in iter_datagrams(path):
            result = _extract_quats(d, "right")
            if result is not None:
                out.append(result[0])  # (quats, aButton) -> quats
        return out

    open_frames = frames(_FIXTURE_OPEN_R)
    fist_frames = frames(_FIXTURE_FIST_R)
    assert len(open_frames) > 100 and len(fist_frames) > 100

    best, metrics = select_mode(open_frames, fist_frames, side="right")
    assert best == "local_rest_ref", f"expected local_rest_ref, got {best!r}: {metrics}"
    assert metrics[best]["open_err"] < 0.01
    assert metrics[best]["fist_curl"] < 0.65

    # Mean open hand decodes right-correct under the winner: its chirality sign
    # must match the RIGHT rest pose's own reference sign (left/right are
    # anatomical mirror images, so "correct" sign is opposite for each side --
    # see tools/discover_fk_mode.py's evaluate_and_report for the same logic).
    kp = keypoints_from_quats(
        np.mean(np.asarray(open_frames), axis=0), best, side="right"
    )
    v5, v9, v4 = kp[5] - kp[0], kp[9] - kp[0], kp[4] - kp[0]
    chir = float(np.dot(np.cross(v5, v9), v4))

    rest_kp = keypoints_from_quats(
        fk.identity_quats(best, side="right"), best, side="right"
    )
    rv5, rv9, rv4 = rest_kp[5] - rest_kp[0], rest_kp[9] - rest_kp[0], rest_kp[4] - rest_kp[0]
    rest_chir = float(np.dot(np.cross(rv5, rv9), rv4))
    assert (chir < 0) == (rest_chir < 0), (
        f"expected right-correct (unmirrored) chirality, got chir={chir:.3e} "
        f"vs rest reference={rest_chir:.3e}"
    )
