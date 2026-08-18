"""Proves the hand-descriptor seam (prehensile.hand.HandDescriptor +
CurlMapper(hand=...)) is REAL, not decorative, by driving CurlMapper with a
hand-authored descriptor for a hand that is genuinely NOT the L6:

  * different channel COUNT (7, not 6)
  * different channel NAMES (none of them match L6_SDK_ORDER's literals)
  * different channel ORDER (pinky-first, thumb last)
  * a channel COUNT that reuses a role (two "pinky" channels)
  * its own pinch wiring (driver="middle", not the L6's "index")
  * its own grasp group (("ring", "pinky"), not the L6's ("middle","ring","pinky"))
  * a REVERSED output convention (open=0/closed=1), to prove curlmap.py
    genuinely never reads ``hand.output`` (see curlmap.py's module docstring)

Also covers the ROS node's exact 5-kwarg CurlMapper(...) call (no ``hand=``)
and that ``hand`` is keyword-only/appended-last, per the migration's
non-negotiable constraints.
"""
import inspect
import math

import numpy as np
import pytest

from prehensile import fk
from prehensile.curlmap import CurlMapper, L6_HAND
from prehensile.command import L6_SDK_ORDER
from prehensile.hand import Channel, HandDescriptor, Output, Pinch

# JointsIndex_Quat proximal(_0) slot per finger, matching test_curlmap.py.
_SLOT_BY_FINGER = {"index": 0, "middle": 3, "ring": 6, "pinky": 9, "thumb": 12}


def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle_rad / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle_rad / 2.0)])


def _curl_kp_each(angles_by_finger: dict[str, float]) -> np.ndarray:
    q = fk.identity_quats(fk.FK_MODE)
    for finger, angle_deg in angles_by_finger.items():
        base = _SLOT_BY_FINGER[finger]
        flex = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(angle_deg))
        q[base] = flex
        q[base + 1] = flex
    return fk.keypoints_from_quats(q, fk.FK_MODE)


_KP_OPEN = _curl_kp_each({})
_KP_FIST = _curl_kp_each({"thumb": 75.0, "index": 75.0, "middle": 75.0, "ring": 75.0, "pinky": 75.0})

# -- OCTOGRIP: a genuinely non-L6 hand ---------------------------------------- #
#
#   channel     role         home    notes
#   j_pinky_a   pinky        10.0    reused role (with j_pinky_b)
#   j_pinky_b   pinky        10.0    reused role (with j_pinky_a)
#   j_ring      ring          0.0
#   j_middle    middle        0.0    the pinch DRIVER (not "index")
#   j_index     index         0.0    NOT the pinch driver on this hand
#   j_spread    thumb_abd   100.0
#   j_oppose    thumb_flex  100.0    the pinch DRIVEN channel
#
# grasp: pinch driver=middle/driven=thumb_flex; group=(ring, pinky) -- the
# group therefore covers j_ring, j_pinky_a AND j_pinky_b (3 channel indices
# over 2 distinct roles).
OCTOGRIP = HandDescriptor(
    name="octogrip",
    channels=(
        Channel(name="j_pinky_a", role="pinky", home=10.0),
        Channel(name="j_pinky_b", role="pinky", home=10.0),
        Channel(name="j_ring", role="ring", home=0.0),
        Channel(name="j_middle", role="middle", home=0.0),
        Channel(name="j_index", role="index", home=0.0),
        Channel(name="j_spread", role="thumb_abd", home=100.0),
        Channel(name="j_oppose", role="thumb_flex", home=100.0),
    ),
    pinch=Pinch(driver="middle", driven="thumb_flex"),
    group=("ring", "pinky"),
    # Deliberately the OPPOSITE convention from L6 (0=open, 1=closed, a
    # fractional unit rather than a percent) -- curlmap.py must not care.
    output=Output(units="fraction", open=0.0, closed=1.0),
)


def _by_name(hand: HandDescriptor, angles) -> dict:
    return dict(zip(hand.order, angles))


# -- shape / range ------------------------------------------------------------ #


def test_fictional_hand_has_the_declared_shape():
    assert len(OCTOGRIP.channels) == 7
    assert OCTOGRIP.order == (
        "j_pinky_a", "j_pinky_b", "j_ring", "j_middle", "j_index", "j_spread", "j_oppose",
    )
    assert set(OCTOGRIP.order).isdisjoint(L6_SDK_ORDER)  # genuinely different names
    assert OCTOGRIP.roles.count("pinky") == 2  # the reused role


def test_fictional_hand_produces_one_value_per_channel_in_declared_order():
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    angles = mapper(_KP_OPEN)
    assert angles is not None
    assert len(angles) == len(OCTOGRIP.channels) == 7


def test_fictional_hand_output_stays_in_the_internal_0_100_range_despite_reversed_output_convention():
    """THE KEY OUTPUT-IS-UNCONSUMED CHECK: OCTOGRIP.output declares open=0/
    closed=1 (the opposite of, and a different unit than, L6's 100/0), yet
    CurlMapper's arithmetic must be completely unaffected -- it always stays
    on the fixed dimensionless 0-100 "openness percent" scale, on every hand,
    per curlmap.py's module docstring. If curlmap.py ever started reading
    ``hand.output`` this would immediately fail (values would land near 0/1,
    not spread across 0-100)."""
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    for kp in (_KP_OPEN, _KP_FIST):
        angles = mapper(kp)
        assert angles is not None
        assert all(0.0 <= a <= 100.0 for a in angles)
    open_angles = _by_name(OCTOGRIP, CurlMapper(alpha=1.0, hand=OCTOGRIP)(_KP_OPEN))
    assert open_angles["j_index"] > 90.0  # open hand reads near-open on the internal scale


def test_fictional_hand_open_and_fist_are_sane_on_every_channel():
    mapper_open = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    mapper_fist = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    open_by = _by_name(OCTOGRIP, mapper_open(_KP_OPEN))
    fist_by = _by_name(OCTOGRIP, mapper_fist(_KP_FIST))
    # Flex-role channels (chord fingers + thumb_flex) read higher open than fist.
    for ch in ("j_pinky_a", "j_pinky_b", "j_ring", "j_middle", "j_index"):
        assert open_by[ch] > fist_by[ch], ch


# -- reused role: both channels track the same underlying metric ------------- #


def test_reused_role_channels_track_identically_with_no_per_channel_tuning():
    """j_pinky_a and j_pinky_b both carry role 'pinky' with no tuning
    overrides distinguishing them, so they must read IDENTICALLY on every
    frame -- proving role reuse actually broadcasts the same tracked metric,
    not two independent computations that happen to agree by luck."""
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    for kp in (_KP_OPEN, _KP_FIST, _curl_kp_each({"pinky": 40.0})):
        by = _by_name(OCTOGRIP, mapper(kp))
        assert by["j_pinky_a"] == pytest.approx(by["j_pinky_b"], abs=1e-9)


def test_reused_role_channels_can_diverge_under_per_channel_tuning():
    """...but they are still independent CHANNELS: per-channel tuning (keyed
    by channel NAME, exactly like the L6 path) can make them diverge."""
    mapper = CurlMapper(
        alpha=1.0, hand=OCTOGRIP,
        tuning={"j_pinky_a": {"flip": True}},
    )
    by = _by_name(OCTOGRIP, mapper(_curl_kp_each({"pinky": 40.0})))
    assert by["j_pinky_a"] == pytest.approx(100.0 - by["j_pinky_b"], abs=1e-6)


# -- pinch wiring: driver is genuinely "middle", not hardcoded "index" ------- #


def test_pinch_driven_follows_the_declared_driver_role_not_index():
    """The driven channel (thumb_flex role, 'j_oppose') must move with the
    DECLARED driver role (middle), and must be INDIFFERENT to a change in the
    'index' role -- proving the coupling is genuinely resolved from
    hand.pinch.driver rather than a hardcoded 'index' literal surviving
    somewhere in curlmap.py."""
    def driven(kp):
        m = CurlMapper(alpha=1.0, hand=OCTOGRIP, couple_thumb_index=True)
        m.locked = True
        return _by_name(OCTOGRIP, m(kp))["j_oppose"]

    # Varying INDEX only (middle held open) must not move the driven channel.
    a = driven(_curl_kp_each({"index": 20.0}))
    b = driven(_curl_kp_each({"index": 80.0}))
    assert a == pytest.approx(b, abs=1e-6)

    # Varying MIDDLE (the real driver) must move it.
    c = driven(_curl_kp_each({"middle": 60.0}))
    assert c != pytest.approx(a, abs=1.0)


def test_pinch_driven_matches_the_drivers_own_commanded_value_at_default_couple_low():
    """With no couple_low configured (defaults to 0), the driven channel must
    exactly mirror the driver channel's own commanded value -- the same
    formula the L6 path pins in test_couple_low_defaults_to_zero_when_unset."""
    m = CurlMapper(alpha=1.0, hand=OCTOGRIP, couple_thumb_index=True)
    m.locked = True
    by = _by_name(OCTOGRIP, m(_curl_kp_each({"middle": 35.0})))
    assert by["j_oppose"] == pytest.approx(by["j_middle"], abs=1e-6)


def test_pinch_inert_unless_locked_and_enabled():
    kp = _curl_kp_each({"middle": 60.0})
    plain = _by_name(OCTOGRIP, CurlMapper(alpha=1.0, hand=OCTOGRIP)(kp))
    m = CurlMapper(alpha=1.0, hand=OCTOGRIP, couple_thumb_index=True)  # not locked
    by = _by_name(OCTOGRIP, m(kp))
    assert by["j_oppose"] == pytest.approx(plain["j_oppose"], abs=1e-9)


# -- group wiring: ("ring", "pinky") covers 3 channels over 2 roles ---------- #


def test_group_covers_every_channel_whose_role_is_named_including_the_reused_one():
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    assert set(OCTOGRIP.order[i] for i in mapper._i_group) == {"j_ring", "j_pinky_a", "j_pinky_b"}


def test_group_value_is_the_pinky_reading_because_the_reused_role_outvotes_ring():
    """The group is {ring, pinky, pinky} (ring appears once, pinky's role is
    claimed by TWO channels) -- median(a, b, b) == b regardless of whether a
    is above or below b, so the group value must always land exactly on the
    (duplicated) pinky reading, never on ring's, whichever direction ring
    moves relative to pinky. This is the sharpest test that role-reuse-inside-
    a-group is honoured (not merely tolerated at load time)."""
    for ring_deg, pinky_deg in [(10.0, 60.0), (80.0, 60.0), (60.0, 60.0)]:
        kp = _curl_kp_each({"ring": ring_deg, "pinky": pinky_deg})
        m = CurlMapper(alpha=1.0, hand=OCTOGRIP)
        m.locked = True
        by = _by_name(OCTOGRIP, m(kp))
        reference = _by_name(OCTOGRIP, CurlMapper(alpha=1.0, hand=OCTOGRIP)(kp))
        assert by["j_pinky_a"] == pytest.approx(reference["j_pinky_a"], abs=1e-6)
        assert by["j_ring"] == pytest.approx(reference["j_pinky_a"], abs=1e-6), (ring_deg, pinky_deg)
        assert by["j_pinky_b"] == pytest.approx(by["j_ring"], abs=1e-9)


def test_group_is_bounded_by_its_members_tracked_values():
    kp = _curl_kp_each({"ring": 15.0, "pinky": 55.0})
    m = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    m.locked = True
    by = _by_name(OCTOGRIP, m(kp))
    reference = _by_name(OCTOGRIP, CurlMapper(alpha=1.0, hand=OCTOGRIP)(kp))
    values = [reference["j_ring"], reference["j_pinky_a"], reference["j_pinky_b"]]
    lo, hi = min(values), max(values)
    assert lo - 1e-9 <= by["j_ring"] <= hi + 1e-9


def test_group_does_not_touch_channels_outside_it():
    """j_index/j_middle/j_spread/j_oppose are outside group=(ring,pinky) and
    must be untouched by the lock's group behaviour."""
    kp = _curl_kp_each({"ring": 20.0, "pinky": 70.0, "index": 30.0})
    m = CurlMapper(alpha=1.0, hand=OCTOGRIP, couple_thumb_index=False)
    m.locked = True
    locked_by = _by_name(OCTOGRIP, m(kp))
    plain_by = _by_name(OCTOGRIP, CurlMapper(alpha=1.0, hand=OCTOGRIP, couple_thumb_index=False)(kp))
    for ch in ("j_index", "j_spread"):
        assert locked_by[ch] == pytest.approx(plain_by[ch], abs=1e-9)


# -- home_gesture: this hand's own per-channel home, not L6's ---------------- #


def test_home_gesture_uses_the_descriptors_own_per_channel_home_values():
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP)
    by = _by_name(OCTOGRIP, mapper.home_gesture())
    assert by == {
        "j_pinky_a": 10.0, "j_pinky_b": 10.0, "j_ring": 0.0,
        "j_middle": 0.0, "j_index": 0.0, "j_spread": 100.0, "j_oppose": 100.0,
    }


def test_home_gesture_tuning_override_still_works_by_channel_name():
    mapper = CurlMapper(alpha=1.0, hand=OCTOGRIP, tuning={"j_ring": {"home_gesture": 42.0}})
    by = _by_name(OCTOGRIP, mapper.home_gesture())
    assert by["j_ring"] == pytest.approx(42.0, abs=1e-9)
    assert by["j_pinky_a"] == pytest.approx(10.0, abs=1e-9)  # everyone else untouched


# -- construction / API surface ----------------------------------------------- #


def test_default_hand_is_l6_hand_identity_map():
    mapper = CurlMapper()
    assert mapper.hand is L6_HAND
    assert mapper.hand.order == tuple(L6_SDK_ORDER)
    assert mapper.hand.roles == tuple(L6_SDK_ORDER)  # identity role map


def test_hand_kwarg_is_keyword_only():
    sig = inspect.signature(CurlMapper.__init__)
    assert sig.parameters["hand"].kind == inspect.Parameter.KEYWORD_ONLY
    # ...and it is the LAST parameter (appended, not inserted).
    assert list(sig.parameters)[-1] == "hand"
    with pytest.raises(TypeError):
        CurlMapper("left", 0.4, None, None, None, False, None, 1.0, 1.0, None, None, True, OCTOGRIP)


def test_ros_node_five_kwarg_construction_still_works():
    """Pins the ROS node's EXACT call shape (hand_teleop_node.py:273-276):
    CurlMapper(side=..., alpha=..., abd_invert=..., tuning=...,
    couple_thumb_index=...) -- no `hand=` -- which must keep constructing
    (and default to the L6) with zero changes required on that call site."""
    sig = inspect.signature(CurlMapper.__init__)
    for name in ("side", "alpha", "abd_invert", "tuning", "couple_thumb_index"):
        assert name in sig.parameters

    mapper = CurlMapper(
        side="left", alpha=0.4, abd_invert=True, tuning=None, couple_thumb_index=True,
    )
    assert mapper.hand is L6_HAND
    assert mapper(_KP_OPEN) is not None
