"""Validators for the UDCap left-hand rest-pose skeleton (prehensile/rest_pose.py).

These follow the reference's "validators" section: transcription cross-checks
against the vendor's wire example, rest-pose segment lengths, quaternion norms,
and structural completeness of the MediaPipe mapping / FK segments.
"""

import numpy as np
import pytest

from prehensile.rest_pose import (
    MEDIAPIPE_NODES,
    REST_POS_CM,
    REST_QUAT,
    SEGMENTS,
    to_wire_meters,
)


def test_mediapipe_nodes_length_and_unique():
    assert len(MEDIAPIPE_NODES) == 21
    assert len(set(MEDIAPIPE_NODES)) == 21


def test_all_mediapipe_nodes_present():
    for node in MEDIAPIPE_NODES:
        assert node in REST_POS_CM, f"{node} missing from REST_POS_CM"
        assert node in REST_QUAT, f"{node} missing from REST_QUAT"


def test_positions_and_quats_shapes():
    for node, pos in REST_POS_CM.items():
        assert pos.shape == (3,), f"{node} position not (3,)"
        assert pos.dtype == np.float64
    for node, quat in REST_QUAT.items():
        assert quat.shape == (4,), f"{node} quat not (4,)"
        assert quat.dtype == np.float64


def test_wire_cross_check_index_proximal():
    wire = to_wire_meters(REST_POS_CM["finger_index_0_l"], REST_POS_CM["wrist_l"])
    expected = np.array([-0.011040, 0.037389, 0.085647])
    np.testing.assert_allclose(wire, expected, atol=1e-4)


def test_wire_cross_check_thumb_proximal():
    wire = to_wire_meters(REST_POS_CM["finger_thumb_0_l"], REST_POS_CM["wrist_l"])
    expected = np.array([0.017912, 0.029178, 0.025298])
    np.testing.assert_allclose(wire, expected, atol=1e-4)


def test_wire_cross_check_middle_end_z_only():
    # Reference: x/y diverge (~1 cm) from the vendor's example packet; z matches.
    wire = to_wire_meters(
        REST_POS_CM["finger_middle_l_end"], REST_POS_CM["wrist_l"]
    )
    assert wire[2] == pytest.approx(0.18790, abs=1e-3)


def _seg_len_cm(parent: str, child: str) -> float:
    return float(np.linalg.norm(REST_POS_CM[child] - REST_POS_CM[parent]))


@pytest.mark.parametrize(
    "parent,child,expected",
    [
        ("finger_index_0_l", "finger_index_1_l", 4.33),
        ("finger_index_1_l", "finger_index_2_l", 2.83),
        ("finger_index_2_l", "finger_index_l_end", 2.28),
        ("finger_thumb_0_l", "finger_thumb_1_l", 4.04),
        ("finger_thumb_1_l", "finger_thumb_2_l", 3.25),
        ("finger_thumb_2_l", "finger_thumb_l_end", 3.05),
    ],
)
def test_segment_lengths_cm(parent, child, expected):
    assert _seg_len_cm(parent, child) == pytest.approx(expected, abs=0.05)


def test_wrist_to_middle_end_length():
    assert _seg_len_cm("wrist_l", "finger_middle_l_end") == pytest.approx(
        18.9, abs=0.1
    )


def test_quats_unit_norm():
    for node, quat in REST_QUAT.items():
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-3), node


def test_segments_endpoints_exist():
    assert len(SEGMENTS) > 0
    for parent, child in SEGMENTS:
        assert parent in REST_POS_CM, f"{parent} (segment parent) missing"
        assert child in REST_POS_CM, f"{child} (segment child) missing"
