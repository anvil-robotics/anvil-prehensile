"""Tests for prehensile/hand_loader.py -- the YAML hand-descriptor loader.

Filesystem-only (tmp_path), mirroring tests/test_tuning.py's conventions: every
rejection is pinned to its own message, naming the offending file/channel, so a
regression that drops a check silently cannot masquerade as some other one
still raising.
"""
import pytest

from prehensile.hand import Output, Pinch
from prehensile.hand_loader import load_hand_descriptor

_MINIMAL = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: thumb_flex, role: thumb_flex, home: 100}
  - {name: thumb_abd,  role: thumb_abd,  home: 100}
  - {name: index,      role: index,      home: 0}
  - {name: middle,     role: middle,     home: 0}
  - {name: ring,       role: ring,       home: 0}
  - {name: pinky,      role: pinky,      home: 0}
"""


def _write(tmp_path, text: str, name: str = "hand.yml"):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- happy path -------------------------------------------------------------- #


def test_load_minimal(tmp_path):
    hand = load_hand_descriptor(_write(tmp_path, _MINIMAL))
    assert hand.name == "testhand"
    assert hand.order == ("thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky")
    assert hand.roles == hand.order  # identity map
    assert [c.home for c in hand.channels] == [100.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    # unset sections fall back to HandDescriptor's own defaults
    assert hand.pinch == Pinch(driver="index", driven="thumb_flex")
    assert hand.group == ("middle", "ring", "pinky")
    assert hand.output == Output()
    assert hand.default_tuning is None
    assert hand.driver_joints == {}


def test_load_full_descriptor_with_every_section(tmp_path):
    text = _MINIMAL + """
grasp:
  pinch: {driver: middle, driven: thumb_flex}
  group: [index, ring]
output:
  units: percent
  open: 100.0
  closed: 0.0
  driver: "prehensile.drivers.realhand_l6:build"
tuning:
  default: configs/curl_tuning.yml
driver_joints:
  thumb_flex: thumb_cmc_pitch
"""
    hand = load_hand_descriptor(_write(tmp_path, text))
    assert hand.pinch == Pinch(driver="middle", driven="thumb_flex")
    assert hand.group == ("index", "ring")
    assert hand.output == Output(units="percent", open=100.0, closed=0.0,
                                  driver="prehensile.drivers.realhand_l6:build")
    assert hand.default_tuning == "configs/curl_tuning.yml"
    assert hand.driver_joints == {"thumb_flex": "thumb_cmc_pitch"}


def test_load_pinch_null_disables_pinch(tmp_path):
    text = _MINIMAL + "\ngrasp:\n  pinch: null\n"
    hand = load_hand_descriptor(_write(tmp_path, text))
    assert hand.pinch is None


def test_load_group_empty_list(tmp_path):
    text = _MINIMAL + "\ngrasp:\n  group: []\n"
    hand = load_hand_descriptor(_write(tmp_path, text))
    assert hand.group == ()


def test_load_a_genuinely_different_shaped_hand(tmp_path):
    """Different channel count, names, and order than the L6 -- proves the
    loader is not secretly L6-shaped."""
    text = """\
schema: prehensile.hand/1
name: octogrip
channels:
  - {name: j_pinky_a, role: pinky, home: 10}
  - {name: j_pinky_b, role: pinky, home: 10}
  - {name: j_ring,    role: ring,  home: 0}
  - {name: j_middle,  role: middle, home: 0}
  - {name: j_index,   role: index, home: 0}
  - {name: j_spread,  role: thumb_abd, home: 100}
  - {name: j_oppose,  role: thumb_flex, home: 100}
grasp:
  pinch: {driver: middle, driven: thumb_flex}
  group: [ring, pinky]
"""
    hand = load_hand_descriptor(_write(tmp_path, text))
    assert hand.name == "octogrip"
    assert len(hand.channels) == 7
    assert hand.order[0] == "j_pinky_a"
    assert hand.roles.count("pinky") == 2


# -- schema / top-level structure -------------------------------------------- #


def test_missing_schema_raises(tmp_path):
    text = _MINIMAL.replace("schema: prehensile.hand/1\n", "")
    with pytest.raises(ValueError, match="unsupported/missing schema"):
        load_hand_descriptor(_write(tmp_path, text))


def test_wrong_schema_raises(tmp_path):
    text = _MINIMAL.replace("prehensile.hand/1", "prehensile.hand/99")
    with pytest.raises(ValueError, match="unsupported/missing schema"):
        load_hand_descriptor(_write(tmp_path, text))


def test_non_mapping_top_level_raises(tmp_path):
    with pytest.raises(ValueError, match="expected a mapping at the top level"):
        load_hand_descriptor(_write(tmp_path, "- just\n- a\n- list\n"))


def test_missing_name_raises(tmp_path):
    text = _MINIMAL.replace("name: testhand\n", "")
    with pytest.raises(ValueError, match="missing required key 'name'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_missing_channels_raises(tmp_path):
    with pytest.raises(ValueError, match="missing required key 'channels'"):
        load_hand_descriptor(_write(tmp_path, "schema: prehensile.hand/1\nname: testhand\n"))


def test_empty_channels_list_raises(tmp_path):
    text = "schema: prehensile.hand/1\nname: testhand\nchannels: []\n"
    with pytest.raises(ValueError, match="'channels' must be a non-empty list"):
        load_hand_descriptor(_write(tmp_path, text))


# -- per-channel load failures (the required 5 rejections) ------------------- #


def test_channel_missing_name_raises(tmp_path):
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {role: index, home: 0}
"""
    with pytest.raises(ValueError, match="missing required key 'name'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_channel_omitted_role_raises_naming_the_channel(tmp_path):
    """One of the five required load-failure rejections: a channel with no
    'role' key at all (not merely a bad one) must fail to load, naming the
    channel."""
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: mystery, home: 50}
"""
    with pytest.raises(ValueError, match="channel 'mystery' is missing required key 'role'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_channel_omitted_home_raises_naming_the_channel(tmp_path):
    """The second required rejection: a channel with no 'home' key."""
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: mystery, role: index}
"""
    with pytest.raises(ValueError, match="channel 'mystery' is missing required key 'home'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_channel_unknown_role_raises(tmp_path):
    """The third required rejection: an unknown role, surfaced by
    HandDescriptor.__post_init__ through the loader."""
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: mystery, role: wrist, home: 50}
"""
    with pytest.raises(ValueError, match="channel 'mystery' has unknown role 'wrist'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_duplicate_channel_name_raises(tmp_path):
    """The fourth required rejection: two channels sharing a NAME (not to be
    confused with sharing a role, which is allowed -- see test_hand.py)."""
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: dup, role: index, home: 0}
  - {name: dup, role: middle, home: 0}
"""
    with pytest.raises(ValueError, match=r"duplicate channel name\(s\) \['dup'\]"):
        load_hand_descriptor(_write(tmp_path, text))


@pytest.mark.parametrize(
    "grasp_yaml",
    [
        "grasp:\n  group: [ring]\n",                             # group role uncarried
        "grasp:\n  pinch: {driver: ring, driven: thumb_flex}\n",  # pinch driver role uncarried
        "grasp:\n  pinch: {driver: index, driven: ring}\n",       # pinch driven role uncarried
    ],
)
def test_pinch_or_group_role_no_channel_carries_raises(tmp_path, grasp_yaml):
    """The fifth required rejection: a pinch/group role that no channel
    carries -- here the descriptor deliberately omits a 'ring' channel."""
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - {name: thumb_flex, role: thumb_flex, home: 100}
  - {name: index,      role: index,      home: 0}
""" + grasp_yaml
    with pytest.raises(ValueError, match="which no channel carries"):
        load_hand_descriptor(_write(tmp_path, text))


# -- malformed sections -------------------------------------------------------- #


def test_channel_entry_not_a_mapping_raises(tmp_path):
    text = """\
schema: prehensile.hand/1
name: testhand
channels:
  - "not a mapping"
"""
    with pytest.raises(ValueError, match=r"channels\[0\]: expected a mapping"):
        load_hand_descriptor(_write(tmp_path, text))


def test_grasp_not_a_mapping_raises(tmp_path):
    text = _MINIMAL + "\ngrasp: [1, 2]\n"
    with pytest.raises(ValueError, match="grasp: expected a mapping"):
        load_hand_descriptor(_write(tmp_path, text))


def test_pinch_missing_driver_raises(tmp_path):
    text = _MINIMAL + "\ngrasp:\n  pinch: {driven: thumb_flex}\n"
    with pytest.raises(ValueError, match="grasp.pinch: missing required key 'driver'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_pinch_missing_driven_raises(tmp_path):
    text = _MINIMAL + "\ngrasp:\n  pinch: {driver: index}\n"
    with pytest.raises(ValueError, match="grasp.pinch: missing required key 'driven'"):
        load_hand_descriptor(_write(tmp_path, text))


def test_output_not_a_mapping_raises(tmp_path):
    text = _MINIMAL + "\noutput: [1, 2]\n"
    with pytest.raises(ValueError, match="output: expected a mapping"):
        load_hand_descriptor(_write(tmp_path, text))


def test_tuning_not_a_mapping_raises(tmp_path):
    text = _MINIMAL + "\ntuning: [1, 2]\n"
    with pytest.raises(ValueError, match="tuning: expected a mapping"):
        load_hand_descriptor(_write(tmp_path, text))


def test_driver_joints_not_a_mapping_raises(tmp_path):
    text = _MINIMAL + "\ndriver_joints: [1, 2]\n"
    with pytest.raises(ValueError, match="driver_joints: expected a mapping"):
        load_hand_descriptor(_write(tmp_path, text))
