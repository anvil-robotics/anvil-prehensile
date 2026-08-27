"""`-h` completeness for `prehensile.viz` -- the half that needs mujoco.

Split out of tests/test_cli_help.py deliberately, and the split is load-bearing
rather than cosmetic: ``from prehensile.viz import _build_parser`` drags in
mujoco at COLLECTION time, and pytest aborts the whole session on a collection
error rather than failing the one file. With these checks still sitting in
tests/test_cli_help.py, that single import made the entire module uncollectable
on a bare `pip install prehensile`, and CI's lean job (the pull-request gate)
died with 0 tests executed rather than merely losing this file. This file is the
one CI ignores; the teleop half now runs leanly.

Not solved with ``importorskip``/``skipif`` on purpose: .github/scripts/
check_pytest_report.py treats ANY skip as a failure, by design, so a skip-based
fix would turn the lean job red instead of green. Same shape as the
tests/test_teleop.py -> tests/test_teleop_retarget.py split.

The checks are the four structural ones from tests/test_cli_help.py (see there
for what each is guarding and why help WORDING is deliberately not asserted on),
applied to viz. ``--interface`` has no viz counterpart, so that fifth check
stays teleop-only. The two helpers are copies: cross-importing one test module
from another is exactly the coupling this split exists to remove.

Hardware-free despite the mujoco dependency: building a parser opens no window,
reads no glove and touches no CAN. ``viz._build_parser`` was split out of
``viz.main()`` for this, mirroring ``teleop._build_parser``.
"""

import pytest

from prehensile.viz import _build_parser as viz_parser


def _flags(parser):
    """Every real flag, excluding argparse's built-in -h."""
    return [a for a in parser._actions if a.dest != "help"]


def _flag(parser, dest):
    return next(a for a in _flags(parser) if a.dest == dest)


def test_every_flag_documents_itself():
    undocumented = [a.option_strings for a in _flags(viz_parser()) if not a.help]
    assert not undocumented, f"viz flags with no help: {undocumented}"


@pytest.mark.parametrize("dest", ["glove", "hand", "map", "fps", "port"])
def test_help_states_the_default(dest):
    action = _flag(viz_parser(), dest)
    assert str(action.default) in action.help, (
        f"viz --{dest}: help does not mention its default {action.default!r}")


def test_usage_shows_how_the_command_is_actually_run():
    """argparse defaults prog to 'viz.py', which is not runnable -- viz is only
    ever invoked as `python -m prehensile.viz`."""
    assert viz_parser().prog == "python -m prehensile.viz"


def test_no_flag_is_orphaned_from_its_help_text():
    """See the teleop copy: argparse drops help onto its own line once the
    invocation outgrows the help column, and '--map {retarget,curl}' is one
    character over argparse's default."""
    parser = viz_parser()
    lines = parser.format_help().splitlines()
    for action in _flags(parser):
        flag = action.option_strings[0]
        line = next(ln for ln in lines if ln.strip().startswith(flag))
        first_word = action.help.split()[0]
        assert first_word in line, (
            f"viz {flag}: help starts on the next line, not beside it")
