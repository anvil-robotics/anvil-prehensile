"""`-h` completeness for `prehensile.teleop`.

teleop is one of the two commands the README tells people to run, so its
``--help`` is the primary documentation for anyone who has not opened the
source. This pins the STRUCTURAL properties that are easy to lose when a flag
is added in a hurry:

  1. every flag says what it does
  2. every flag whose default matters says what that default is -- otherwise
     the reader has to read the source to find out what happens if they omit it
  3. every flag's help sits beside it rather than wrapping onto its own line
  4. the usage line shows how the command is actually invoked

Deliberately NOT tested: the wording of any help string. Grepping help text for
keywords breaks on innocent rewording while proving nothing about whether the
text is any good, so those assertions were dropped rather than maintained.

viz gets the SAME four checks, in tests/test_cli_help_viz.py. Two files rather
than one parametrized across both commands, and the split is load-bearing:
``from prehensile.viz import ...`` drags in mujoco at COLLECTION time, and
pytest aborts the whole session on a collection error rather than failing the
one file -- so a single viz import here would take CI's lean job (the
pull-request gate) down with it, exactly as it did on 2026-08-27. Not solved
with ``importorskip``/``skipif`` on purpose: .github/scripts/
check_pytest_report.py treats ANY skip as a failure, by design, so a skip-based
fix would turn that job red instead of green. Same reasoning, and the same
shape, as the tests/test_teleop.py -> tests/test_teleop_retarget.py split.

The two helpers below are copied into the viz file rather than imported from
here: cross-importing one test module from another is the coupling this split
exists to remove -- each file must collect on its own.

Hardware-free: building a parser touches no glove, no CAN and no display.
``teleop._build_parser`` already existed for exactly this reason (see its
docstring).
"""

import pytest

from prehensile.teleop import _build_parser as teleop_parser


def _flags(parser):
    """Every real flag, excluding argparse's built-in -h."""
    return [a for a in parser._actions if a.dest != "help"]


def _flag(parser, dest):
    return next(a for a in _flags(parser) if a.dest == dest)


def test_every_flag_documents_itself():
    undocumented = [a.option_strings for a in _flags(teleop_parser()) if not a.help]
    assert not undocumented, f"teleop flags with no help: {undocumented}"


# --fps/--speed/--port are the flags whose behaviour silently changes with the
# value; --glove/--hand/--map pick a whole code path. For all of them "what do I
# get if I leave it out?" has to be answerable from --help alone.
@pytest.mark.parametrize("dest", ["glove", "hand", "map", "fps", "speed", "port"])
def test_help_states_the_default(dest):
    action = _flag(teleop_parser(), dest)
    assert str(action.default) in action.help, (
        f"teleop --{dest}: help does not mention its default {action.default!r}")


def test_usage_shows_how_the_command_is_actually_run():
    """argparse defaults prog to 'teleop.py', which is not runnable -- teleop is
    only ever invoked as `python -m prehensile.teleop`."""
    assert teleop_parser().prog == "python -m prehensile.teleop"


def test_teleop_has_no_interface_flag():
    """--interface was removed: the CAN interface always follows --hand
    (hand_l / hand_r, as named by tools/bring_up_hand.py), so an override was a
    second way to say the same thing. Re-adding it makes this parse succeed."""
    with pytest.raises(SystemExit) as exc:
        teleop_parser().parse_args(["--interface", "hand_r"])
    assert exc.value.code == 2  # argparse usage error


def test_no_flag_is_orphaned_from_its_help_text():
    """argparse drops a flag's help to the next line when the invocation is
    wider than the help column, which reads as a gap in the middle of the list.
    '--map {retarget,curl}' is 21 chars against argparse's default 20, so it
    wrapped while every shorter flag stayed inline. The column is widened to fit
    the longest flag instead."""
    parser = teleop_parser()
    lines = parser.format_help().splitlines()
    for action in _flags(parser):
        flag = action.option_strings[0]
        line = next(ln for ln in lines if ln.strip().startswith(flag))
        first_word = action.help.split()[0]
        assert first_word in line, (
            f"teleop {flag}: help starts on the next line, not beside it")
