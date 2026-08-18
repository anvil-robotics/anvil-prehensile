"""Unit tests for prehensile/hand.py -- the HandDescriptor dataclasses.

Pure-synthetic, no filesystem/YAML: constructs ``Channel``/``Pinch``/``Output``/
``HandDescriptor`` directly, so these tests pin the CONTRACT any loader (in
particular ``prehensile.hand_loader``) must uphold, independent of how a
descriptor was built. File-based load-failure tests (matching
``prehensile.tuning``'s loud-failure message style) live in
``tests/test_hand_loader.py``.
"""
import pytest

from prehensile.hand import Channel, HandDescriptor, Output, Pinch, ROLES


def _channel(name, role, home=50.0):
    return Channel(name=name, role=role, home=home)


def _hand(**kwargs):
    """A minimal valid 2-channel hand, overridable via kwargs.

    ``group``/``pinch`` default to ``()``/``None`` here (unlike
    ``HandDescriptor`` itself, whose own defaults are
    ``("middle", "ring", "pinky")``/``Pinch(driver="index", driven="thumb_flex")``)
    precisely so overriding ``channels`` to roles other than index/thumb_flex
    does not ALSO have to fight the unrelated pinch/group defaults -- tests
    that want the real default wiring build a full 6-channel HandDescriptor
    directly instead (see test_default_pinch_and_group_match_the_shipped_l6_wiring).
    """
    defaults = dict(
        name="mini",
        channels=(_channel("a", "index"), _channel("b", "thumb_flex")),
        group=(),
        pinch=None,
    )
    defaults.update(kwargs)
    return HandDescriptor(**defaults)


# -- construction basics -------------------------------------------------- #


def test_roles_is_exactly_the_six_curlmap_metrics():
    assert ROLES == {"thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"}


def test_minimal_valid_hand_constructs():
    hand = _hand()
    assert hand.order == ("a", "b")
    assert hand.roles == ("index", "thumb_flex")


def test_order_and_roles_are_aligned_and_derived_from_channels():
    hand = _hand(channels=(_channel("x", "pinky"), _channel("y", "ring"), _channel("z", "middle")))
    assert hand.order == ("x", "y", "z")
    assert hand.roles == ("pinky", "ring", "middle")


def test_no_channels_raises():
    with pytest.raises(ValueError, match="needs at least one channel"):
        HandDescriptor(name="empty", channels=())


# -- C4: role lives on the channel; an unmapped channel cannot exist ------- #


def test_omitted_role_raises_naming_the_channel():
    """Channel.role has no usable default (None/""), so a channel built
    without one fails to load -- there is no syntax for an unmapped channel."""
    with pytest.raises(ValueError, match="channel 'weird' has no role"):
        _hand(channels=(_channel("weird", None), _channel("b", "thumb_flex")))


def test_empty_string_role_raises_naming_the_channel():
    with pytest.raises(ValueError, match="channel 'weird' has no role"):
        _hand(channels=(_channel("weird", ""), _channel("b", "thumb_flex")))


def test_unknown_role_raises_naming_the_channel_and_role():
    with pytest.raises(ValueError, match="channel 'a' has unknown role 'wrist'"):
        _hand(channels=(_channel("a", "wrist"), _channel("b", "thumb_flex")))


def test_duplicate_channel_name_raises():
    with pytest.raises(ValueError, match=r"duplicate channel name\(s\) \['a'\]"):
        _hand(channels=(_channel("a", "index"), _channel("a", "middle")))


def test_role_reused_across_multiple_channels_is_allowed():
    """Unlike channel NAMES, a role may be claimed by more than one channel --
    e.g. two tendons/linkages both driven off the same tracked curl. This is
    the property the Phase-4a "fictional hand" fixture (test_curlmap_hand.py)
    depends on."""
    hand = _hand(channels=(_channel("a", "pinky"), _channel("b", "pinky"), _channel("c", "thumb_flex")))
    assert hand.roles == ("pinky", "pinky", "thumb_flex")
    assert hand.index_of_role("pinky") == 0  # first match


# -- C1: home is required, per channel, no default ------------------------- #


def test_omitted_home_raises_naming_the_channel():
    with pytest.raises(ValueError, match="channel 'a' has no 'home' value"):
        _hand(channels=(Channel(name="a", role="index"), _channel("b", "thumb_flex")))


def test_explicit_home_none_raises_naming_the_channel():
    with pytest.raises(ValueError, match="channel 'a' has no 'home' value"):
        _hand(channels=(Channel(name="a", role="index", home=None), _channel("b", "thumb_flex")))


def test_home_of_zero_is_not_treated_as_missing():
    """0.0 is a legitimate (falsy!) home value and must not be confused with
    'omitted' -- the check must be `is None`, never a truthiness check."""
    hand = _hand(channels=(_channel("a", "index", home=0.0), _channel("b", "thumb_flex", home=0.0)))
    assert hand.channels[0].home == 0.0


# -- C2: no per-channel limits concept exists here -------------------------- #


def test_no_limits_field_exists_on_channel_or_descriptor():
    """Per-channel limits are deliberately NOT a HandDescriptor concept (see
    the module docstring) -- they already live in the URDF and in tuning's
    couple_low. Pin the absence structurally so a future edit re-adding one
    fails a test instead of silently reintroducing the double-owned concept."""
    assert not hasattr(Channel(name="a", role="index", home=1.0), "limits")
    assert not hasattr(Channel(name="a", role="index", home=1.0), "min")
    assert not hasattr(Channel(name="a", role="index", home=1.0), "max")
    assert not hasattr(_hand(), "limits")


# -- C5: group membership is data; the reducer is not configurable here ---- #


def test_group_has_no_reducer_field():
    """The reducer (median) is hardcoded in curlmap.py, never a per-hand
    'reduce' knob -- see the module docstring's safety argument. Pin that no
    such field exists on the descriptor."""
    assert not hasattr(_hand(), "reduce")
    assert not hasattr(_hand(), "reducer")


# -- pinch/group role validation ------------------------------------------- #


def test_group_role_no_channel_carries_raises():
    with pytest.raises(ValueError, match="grasp group names role 'ring', which no channel carries"):
        _hand(group=("ring",))


def test_group_role_unknown_raises():
    with pytest.raises(ValueError, match="grasp group role 'wrist' unknown"):
        _hand(group=("wrist",))


def test_group_can_be_empty():
    """An empty grasp group is valid (a hand with no median-grouped channels)."""
    hand = _hand(group=())
    assert hand.group == ()


def test_pinch_driver_role_no_channel_carries_raises():
    with pytest.raises(ValueError, match="pinch driver names role 'middle', which no channel carries"):
        _hand(pinch=Pinch(driver="middle", driven="thumb_flex"))


def test_pinch_driven_role_no_channel_carries_raises():
    with pytest.raises(ValueError, match="pinch driven names role 'middle', which no channel carries"):
        _hand(pinch=Pinch(driver="index", driven="middle"))


def test_pinch_role_unknown_raises():
    with pytest.raises(ValueError, match="pinch driver role 'wrist' unknown"):
        _hand(pinch=Pinch(driver="wrist", driven="thumb_flex"))


def test_pinch_none_disables_the_pinch_entirely():
    hand = _hand(pinch=None)
    assert hand.pinch is None


def test_default_pinch_and_group_match_the_shipped_l6_wiring():
    hand = HandDescriptor(
        name="defaults-only",
        channels=tuple(_channel(n, n) for n in ("thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky")),
    )
    assert hand.pinch == Pinch(driver="index", driven="thumb_flex")
    assert hand.group == ("middle", "ring", "pinky")


# -- Output: driver-facing facts, unconsumed here --------------------------- #


def test_output_defaults_match_l6_convention():
    out = Output()
    assert (out.units, out.open, out.closed, out.driver) == ("percent", 100.0, 0.0, None)


def test_output_can_declare_a_reversed_convention():
    """Output is a free-form driver-facing record that HandDescriptor itself
    never interprets or validates against the mapper's internal space (see
    curlmap.py's module docstring) -- a hand may declare 0=open/1=closed
    without HandDescriptor objecting."""
    out = Output(units="fraction", open=0.0, closed=1.0, driver="pkg.mod:build")
    hand = _hand(output=out)
    assert hand.output is out


# -- index_of_role / channel_of_role --------------------------------------- #


def test_index_of_role_returns_none_when_absent():
    hand = _hand()
    assert hand.index_of_role("pinky") is None
    assert hand.channel_of_role("pinky") is None


def test_channel_of_role_returns_the_channel_name():
    hand = _hand()
    assert hand.channel_of_role("index") == "a"
    assert hand.channel_of_role("thumb_flex") == "b"
