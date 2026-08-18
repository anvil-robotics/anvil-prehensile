#!/usr/bin/env python3
"""Assert that a pytest run actually EXECUTED the tests CI depends on.

Why this exists: tests/test_packaging.py carries a module-level
`pytest.mark.skipif(shutil.which("uv") is None, ...)`, and two of its fixtures
call `pytest.skip(...)` when the fresh-venv wheel install cannot reach the
network. Every one of those paths leaves pytest's exit status at 0, so a CI job
that checks only the exit code reports GREEN while the packaging regression
guard never ran at all. That guard is what stands between a wheel that quietly
stopped shipping prehensile/configs/curl_tuning.yml and tuning falling back to
defaults, where thumb_flex.couple_low becomes 0 and the robot thumb closes its
full travel into the operator's fingers. "Skipped" is therefore a FAILURE here,
not a neutral outcome.

Reads the JUnit XML that pytest wrote (--junit-xml=...) and exits non-zero if:

  * the report is missing, empty or unparseable;
  * ANY testcase was skipped -- neither CI job has an expected skip today, so a
    new one is a deliberate decision that should have to edit this call site;
  * a required test file contributed fewer executed-and-passing testcases than
    demanded (--require tests/test_packaging.py=3), or a specific pinned
    testcase did not pass (--require tests/test_packaging.py::test_foo) --
    the file-level count alone is satisfied by ANY N passing tests in that
    file, so it cannot by itself prove a particular load-bearing test ran;
  * the run executed fewer testcases in total than the known-good count
    (--min-tests), i.e. tests silently stopped being collected.

Failures and errors are left to pytest's own exit status, but they are reported
here too so one message explains the whole picture.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BAD_OUTCOMES = ("skipped", "failure", "error")


def _fail(message: str) -> None:
    print(f"::error::{message}" if _on_github() else f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _on_github() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _outcome(testcase: ET.Element) -> str:
    """Return "skipped"/"failure"/"error", or "passed" when nothing is attached."""
    for tag in BAD_OUTCOMES:
        if testcase.find(tag) is not None:
            return tag
    return "passed"


def _identity(testcase: ET.Element) -> str:
    """Human-readable "file::name" for a testcase element."""
    where = testcase.get("file") or testcase.get("classname") or "<unknown>"
    return f"{where}::{testcase.get('name', '<unnamed>')}"


def _matches_file(testcase: ET.Element, wanted: str) -> bool:
    """True if this testcase came from the source file `wanted`.

    pytest records both `file="tests/test_packaging.py"` and
    `classname="tests.test_packaging"` -- match either, so the check survives a
    junit_family change. In practice `classname` is the one doing the work: our
    pinned junit_family = "xunit2" (see pyproject.toml) does NOT emit a `file=`
    attribute on <testcase> at all (verified against this repo's own pytest
    9.1.1 output), so the `file` branch below is defensive only, in case a
    future pytest (or a junit_family change) starts emitting it again.
    """
    wanted_posix = Path(wanted).as_posix()
    if (testcase.get("file") or "") == wanted_posix:
        return True
    dotted = wanted_posix.removesuffix(".py").replace("/", ".")
    classname = testcase.get("classname") or ""
    return classname == dotted or classname.startswith(f"{dotted}.")


def _parse_require(value: str) -> tuple[str, str | None, int]:
    """Parse a --require value into (path, name, count).

    Two forms:
      * ``PATH=COUNT`` -- file-level: at least COUNT passing testcases from
        PATH, by any name. This is a headcount only -- it is satisfied by ANY
        COUNT passing tests in that file, so on its own it cannot tell a real
        guard from an unrelated test that happens to make up the number.
      * ``PATH::NAME`` -- testcase-level: the specific testcase PATH::NAME must
        be present and have passed. This is what actually pins a load-bearing
        test by identity rather than by count; `name` is returned non-None and
        `count` is implicitly 1 (a plain function testcase can only pass once
        per report).
    """
    if "::" in value:
        path, _, name = value.partition("::")
        if not path or not name:
            message = f"--require PATH::NAME expects both a path and a name, got {value!r}"
            raise argparse.ArgumentTypeError(message)
        return path, name, 1
    path, _, count = value.rpartition("=")
    if not path or not count.isdigit():
        message = (
            "--require expects PATH=COUNT (e.g. tests/test_packaging.py=3) or "
            f"PATH::NAME (e.g. tests/test_packaging.py::test_foo), got {value!r}"
        )
        raise argparse.ArgumentTypeError(message)
    return path, None, int(count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="path to the --junit-xml file pytest wrote")
    parser.add_argument(
        "--min-tests",
        type=int,
        default=0,
        help="fail if fewer than this many testcases were executed",
    )
    parser.add_argument(
        "--require",
        type=_parse_require,
        action="append",
        default=[],
        metavar="PATH=COUNT|PATH::NAME",
        help="fail unless PATH contributed at least COUNT passing testcases "
             "(PATH=COUNT), or unless the specific testcase PATH::NAME passed "
             "(PATH::NAME) -- COUNT alone is satisfied by any COUNT passing "
             "tests in that file, so pin load-bearing tests by NAME too",
    )
    args = parser.parse_args(argv)

    report: Path = args.report
    if not report.is_file():
        _fail(f"no pytest JUnit report at {report} -- the test step did not produce one")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as err:
        _fail(f"pytest JUnit report at {report} is not parseable XML: {err}")

    testcases = list(root.iter("testcase"))
    if not testcases:
        _fail(f"pytest JUnit report at {report} contains no testcases at all")

    outcomes = [(_identity(tc), _outcome(tc)) for tc in testcases]
    tally = {name: sum(1 for _, o in outcomes if o == name) for name in ("passed", *BAD_OUTCOMES)}
    print(
        f"{report}: {len(testcases)} testcases -- "
        + ", ".join(f"{count} {name}" for name, count in tally.items()),
    )

    problems: list[str] = []

    skipped = [ident for ident, outcome in outcomes if outcome == "skipped"]
    if skipped:
        problems.append(
            "these testcases were SKIPPED, and this workflow treats a skip as a failure "
            "(a skipped guard protects nothing): " + ", ".join(skipped),
        )

    broken = [f"{ident} ({outcome})" for ident, outcome in outcomes if outcome in ("failure", "error")]
    if broken:
        problems.append("these testcases did not pass: " + ", ".join(broken))

    for wanted, name, needed in args.require:
        if name is None:
            passing = sum(1 for tc in testcases if _matches_file(tc, wanted) and _outcome(tc) == "passed")
            label = wanted
        else:
            passing = sum(
                1 for tc in testcases
                if _matches_file(tc, wanted) and tc.get("name") == name and _outcome(tc) == "passed"
            )
            label = f"{wanted}::{name}"
        if passing < needed:
            problems.append(
                f"{label} contributed only {passing} passing testcases, expected at least "
                f"{needed} -- the guard it provides did not actually run",
            )

    if len(testcases) < args.min_tests:
        problems.append(
            f"only {len(testcases)} testcases were executed, expected at least {args.min_tests} "
            "-- tests stopped being collected",
        )

    if problems:
        _fail("; ".join(problems))
    print("pytest report check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
