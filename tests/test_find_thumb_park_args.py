"""Tests for tools/find_thumb_park.py's argument parsing/validation.

Hardware-free: importing this module and calling ``_parse_args``/
``_build_parser`` never imports ``can``/``realhand`` (those stay lazy, inside
``main()``'s body -- see the module's own comment), so this exercises exactly
the post-parse ``--channel`` validation C3 calls out, with no hardware and no
``--live``/``set_angles``/``main()`` call involved.
"""
import argparse

import pytest

from prehensile.command import L6_SDK_ORDER
from tools.find_thumb_park import _build_parser, _parse_args


def test_build_parser_does_not_restrict_channel_choices():
    """`--channel` has no argparse `choices=` -- see _build_parser's docstring
    for why (a hand-descriptor-driven tool cannot fix its valid channels at
    parser-BUILD time)."""
    ap = _build_parser()
    channel_action = next(a for a in ap._actions if a.dest == "channel")
    assert channel_action.choices is None


def test_default_channel_is_thumb_abd():
    args = _parse_args([])
    assert args.channel == "thumb_abd"


@pytest.mark.parametrize("channel", L6_SDK_ORDER)
def test_every_l6_channel_is_accepted(channel):
    args = _parse_args(["--channel", channel])
    assert args.channel == channel


def test_invalid_channel_exits_like_argparses_own_choices_would(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--channel", "wrist"])
    assert exc_info.value.code == 2  # argparse's own usage-error exit code
    err = capsys.readouterr().err
    assert "invalid choice: 'wrist'" in err
    for ch in L6_SDK_ORDER:
        assert ch in err  # the valid-choices list is still surfaced


def test_other_flags_still_parse_normally():
    args = _parse_args(["--speed", "40", "--interface", "hand_x", "--start", "12.5"])
    assert (args.speed, args.interface, args.start) == (40.0, "hand_x", 12.5)


def test_parser_is_an_argparse_argument_parser():
    assert isinstance(_build_parser(), argparse.ArgumentParser)
