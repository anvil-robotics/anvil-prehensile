"""Direction/mapping guards for prehensile/command.py (qpos -> L6 6-value command).

Hardware-free: builds a synthetic joint_names list (the six driver joints) and a
matching qpos, so every assertion is about the pure qpos->angle mapping, not the
retargeter or the glove.
"""

import numpy as np
import pytest

from prehensile.command import (
    L6_DRIVER_JOINTS,
    L6_SDK_ORDER,
    qpos_index_map,
    qpos_to_l6_angles,
)


def _synthetic():
    """A joint_names list holding exactly the six driver joints, + its index_map."""
    joint_names = list(L6_DRIVER_JOINTS.values())
    return joint_names, qpos_index_map(joint_names)


def test_flexion_channels_open_high_close_low():
    """Live-confirmed flexion sense: more flex angle -> higher (more-open) command.

    (0 rad extended -> 0; pi/2 flexed -> 100, per the current normalized*100 map.)
    """
    joint_names, imap = _synthetic()
    extended = np.zeros(len(joint_names))
    flexed = np.full(len(joint_names), np.pi / 2.0)
    a_ext = dict(zip(L6_SDK_ORDER, qpos_to_l6_angles(extended, imap)))
    a_flex = dict(zip(L6_SDK_ORDER, qpos_to_l6_angles(flexed, imap)))
    for f in ("thumb_flex", "index", "middle", "ring", "pinky"):
        assert a_flex[f] > a_ext[f], f"{f}: flex={a_flex[f]} ext={a_ext[f]}"


def test_thumb_abd_is_inverted_relative_to_flexion():
    """thumb_abd (thunb_cmc_roll) is an abduction axis, NOT an open/close axis.

    The global open/close inversion that fixed the five flexion channels reverses
    the thumb spread (live-confirmed: spreading the operator's thumb tucked the
    robot's). So the abduction channel must map OPPOSITE to the flexion channels:
    a larger roll angle must yield a *lower* command, not a higher one.
    """
    joint_names, imap = _synthetic()
    low_roll = np.zeros(len(joint_names))
    high_roll = np.full(len(joint_names), np.pi / 2.0)
    a_low = dict(zip(L6_SDK_ORDER, qpos_to_l6_angles(low_roll, imap)))
    a_high = dict(zip(L6_SDK_ORDER, qpos_to_l6_angles(high_roll, imap)))
    assert a_high["thumb_abd"] < a_low["thumb_abd"], (
        f"thumb_abd not inverted: high={a_high['thumb_abd']} low={a_low['thumb_abd']}"
    )


def test_invert_flex_false_complements_flexion_but_leaves_abd_unchanged():
    """Wuji (invert_flex=False) is the flexion-complement of UDCap (invert_flex=True).

    Both open/close senses are live-hardware-confirmed for their respective glove,
    reducing to a single boolean. For any given qpos, the flexion channels
    (thumb_flex, index, middle, ring, pinky) under invert_flex=False must equal
    100 - (the invert_flex=True value), while thumb_abd -- an abduction axis, not
    an open/close one -- must be IDENTICAL between the two modes.
    """
    joint_names, imap = _synthetic()
    rng = np.random.default_rng(0)
    qpos = rng.uniform(0.0, np.pi / 2.0, size=len(joint_names))

    angles_true = dict(zip(L6_SDK_ORDER, qpos_to_l6_angles(qpos, imap, invert_flex=True)))
    angles_false = dict(
        zip(L6_SDK_ORDER, qpos_to_l6_angles(qpos, imap, invert_flex=False))
    )

    for f in ("thumb_flex", "index", "middle", "ring", "pinky"):
        assert angles_false[f] == pytest.approx(100.0 - angles_true[f]), (
            f"{f}: invert_flex=False={angles_false[f]} "
            f"not complement of invert_flex=True={angles_true[f]}"
        )

    assert angles_false["thumb_abd"] == pytest.approx(angles_true["thumb_abd"]), (
        "thumb_abd must be unaffected by invert_flex: "
        f"False={angles_false['thumb_abd']} True={angles_true['thumb_abd']}"
    )

    for mode_angles in (angles_true, angles_false):
        for slot, value in mode_angles.items():
            assert 0.0 <= value <= 100.0, f"{slot}={value} out of [0,100] range"
