"""`-h` completeness for the two user-facing commands (teleop, viz).

These are the commands the README tells people to run, so their ``--help`` is
the primary documentation for anyone who has not opened the source. This pins
the STRUCTURAL properties that are easy to lose when a flag is added in a hurry:

  1. every flag says what it does
  2. every flag whose default matters says what that default is -- otherwise
     the reader has to read the source to find out what happens if they omit it
  3. every flag's help sits beside it rather than wrapping onto its own line
  4. the usage line shows how the command is actually invoked

Deliberately NOT tested: the wording of any help string. Grepping help text for
keywords breaks on innocent rewording while proving nothing about whether the
text is any good, so those assertions were dropped rather than maintained.

Hardware-free: building a parser touches no glove, no CAN and no display.
``teleop._build_parser`` already existed for exactly this reason (see its
docstring); ``viz._build_parser`` was split out of ``main()`` to match.
"""

import pytest

from prehensile.teleop import _build_parser as teleop_parser
from prehensile.viz import _build_parser as viz_parser

PARSERS = {"teleop": teleop_parser, "viz": viz_parser}


def _flags(parser):
    """Every real flag, excluding argparse's built-in -h."""
    return [a for a in parser._actions if a.dest != "help"]


def _flag(parser, dest):
    return next(a for a in _flags(parser) if a.dest == dest)


@pytest.mark.parametrize("command", sorted(PARSERS))
def test_every_flag_documents_itself(command):
    parser = PARSERS[command]()
    undocumented = [a.option_strings for a in _flags(parser) if not a.help]
    assert not undocumented, f"{command}: flags with no help: {undocumented}"


# --fps/--speed/--port are the flags whose behaviour silently changes with the
# value; --glove/--hand/--map pick a whole code path. For all of them "what do I
# get if I leave it out?" has to be answerable from --help alone.
@pytest.mark.parametrize("command,dest", [
    ("teleop", "glove"), ("teleop", "hand"), ("teleop", "map"),
    ("teleop", "fps"), ("teleop", "speed"), ("teleop", "port"),
    ("viz", "glove"), ("viz", "hand"), ("viz", "map"),
    ("viz", "fps"), ("viz", "port"),
])
def test_help_states_the_default(command, dest):
    action = _flag(PARSERS[command](), dest)
    assert str(action.default) in action.help, (
        f"{command} --{dest}: help does not mention its default "
        f"{action.default!r}")


@pytest.mark.parametrize("command", sorted(PARSERS))
def test_usage_shows_how_the_command_is_actually_run(command):
    """argparse defaults prog to 'teleop.py'/'viz.py', which is not runnable --
    both are only ever invoked as `python -m prehensile.<name>`."""
    assert PARSERS[command]().prog == f"python -m prehensile.{command}"


def test_teleop_has_no_interface_flag():
    """--interface was removed: the CAN interface always follows --hand
    (hand_l / hand_r, as named by tools/bring_up_hand.py), so an override was a
    second way to say the same thing. Re-adding it makes this parse succeed."""
    with pytest.raises(SystemExit) as exc:
        teleop_parser().parse_args(["--interface", "hand_r"])
    assert exc.value.code == 2  # argparse usage error


@pytest.mark.parametrize("command", sorted(PARSERS))
def test_no_flag_is_orphaned_from_its_help_text(command):
    """argparse drops a flag's help to the next line when the invocation is
    wider than the help column, which reads as a gap in the middle of the list.
    '--map {retarget,curl}' is 21 chars against argparse's default 20, so it
    wrapped while every shorter flag stayed inline. The column is widened to fit
    the longest flag instead."""
    parser = PARSERS[command]()
    lines = parser.format_help().splitlines()
    for action in _flags(parser):
        flag = action.option_strings[0]
        line = next(ln for ln in lines if ln.strip().startswith(flag))
        first_word = action.help.split()[0]
        assert first_word in line, (
            f"{command} {flag}: help starts on the next line, not beside it")
