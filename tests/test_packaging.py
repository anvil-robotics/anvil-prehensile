"""Regression guard for the packaging invariant this phase exists to protect:

    resolve_tuning(DEFAULT_TUNING_PATH) must return the REAL shipped tuning --
    not None -- after a normal (non-editable) `pip install` of a built wheel.

Why this needs a wheel-and-fresh-venv test rather than a unit test: the bug
this guards against (DEFAULT_TUNING_PATH resolving via `.parent.parent` out of
the package into a repo-root `configs/` that only exists for an editable
checkout) is invisible from an editable install or from this repo's own dev
venv -- `__file__` happens to land inside the repo either way, so the file is
"found" by accident. The only way to see the real failure mode is to build an
actual wheel and install it, non-editably, somewhere that has never seen this
checkout. A missing `thumb_flex.couple_low` is not a cosmetic default: it lets
the robot thumb close its full travel into a thumb parked across the palm --
into the operator's closing fingers (see prehensile/tuning.py's docstring).

This also (offline, no venv/network needed) locks in the Change 3 dependency
split: numpy/pyyaml/protobuf unconditional, realhand/dex-retargeting/mujoco/
torch/wuji-sdk gated behind `l6`/`research` extras.

Environment note: this sandbox has ROS 2 (jazzy) sourced, which sets
PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages globally. A `python -m
venv` still reports include-system-site-packages=false, but the ambient
PYTHONPATH is *not* cleared by venv creation and would still apply to anything
run inside it via plain `subprocess.run(..., env=os.environ)`. That ROS path
does not ship a `prehensile` package, so it happens not to shadow the installed
one here -- but the fresh-venv subprocess below strips PYTHONPATH explicitly
anyway, so this test's isolation does not depend on that being true forever.
"""

from __future__ import annotations

import email
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent

_UV = shutil.which("uv")

pytestmark = pytest.mark.skipif(
    _UV is None,
    reason="needs the `uv` build toolchain (uv build --wheel) on PATH",
)


def _clean_build_artifacts() -> None:
    """Remove this repo's own build/lib cache and *.egg-info.

    setuptools' setup.py-compatibility shim reuses <repo>/build/lib as an
    INCREMENTAL cache across separate `uv build` / `python -m build`
    invocations, in the repo's own working tree (not some isolated temp dir).
    Verified by hand while writing this test: after deliberately deleting
    [tool.setuptools.package-data] to check the test catches the regression,
    the very next build still produced a wheel containing the YAML, because
    build/lib/prehensile/configs/curl_tuning.yml was left over from a prior
    (correct) build and setuptools just copied it back out unconditionally.
    Skipping this cleanup would let a stale cache silently paper over exactly
    the pyproject.toml regression this test exists to catch.
    """
    for stale in (REPO_ROOT / "build", REPO_ROOT / "prehensile.egg-info"):
        shutil.rmtree(stale, ignore_errors=True)


def _venv_python(venv_dir: Path) -> Path:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return venv_dir / bin_dir / exe


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    """Build a wheel from THIS repo's current source tree into a throwaway dir."""
    _clean_build_artifacts()
    out_dir = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [_UV, "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    # Also clean up the build-cache side effect we may have just made in the
    # repo's own working tree, win or lose, so this is not left behind for the
    # next real build (editable or otherwise) to accidentally pick up.
    _clean_build_artifacts()
    assert result.returncode == 0, (
        f"`uv build --wheel` failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel in {out_dir}, got {wheels}"
    return wheels[0]


def test_wheel_ships_curl_tuning_yaml_with_real_content(built_wheel):
    """Offline, no venv/install needed: the wheel must contain
    prehensile/configs/curl_tuning.yml, and it must be the real shipped file
    (checked for the same couple_low=30 marker the end-to-end test asserts),
    not an empty or truncated stand-in. This is Change 2's own trap: setuptools
    ships .py files by default and silently drops everything else without an
    explicit [tool.setuptools.package-data] entry.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
        assert "prehensile/configs/curl_tuning.yml" in names, names
        content = zf.read("prehensile/configs/curl_tuning.yml").decode()
    assert "couple_low: 30" in content, content


def test_wheel_metadata_pins_realhand_torch_mujoco_wuji_sdk_to_extras(built_wheel):
    """Offline, no network: read METADATA out of the wheel zip directly and
    assert the Change 3 dependency-extras split -- numpy/pyyaml/protobuf
    unconditional; realhand (the `l6` extra) and dex-retargeting/mujoco/torch/
    wuji-sdk (the `research` extra) only ever appear with an `extra == ...`
    marker. A plain `pip install prehensile` must never drag in `realhand`
    (a git URL, and a numpy/scipy pin that conflicts with the downstream ROS
    deployment's own pins) or the research stack.
    """
    with zipfile.ZipFile(built_wheel) as zf:
        metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        metadata_text = zf.read(metadata_name).decode()

    msg = email.message_from_string(metadata_text)
    requires_dist = msg.get_all("Requires-Dist") or []
    # Parse with packaging.requirements rather than naive string splitting: a
    # naive `entry.split()[0] == pkg` mis-handles "protobuf>=5" (no space
    # before the specifier), silently reporting protobuf as absent.
    parsed = [Requirement(r) for r in requires_dist]

    def entries_for(pkg: str) -> list[Requirement]:
        return [r for r in parsed if r.name.lower() == pkg]

    for pkg in ("numpy", "pyyaml", "protobuf"):
        entries = entries_for(pkg)
        assert entries, f"{pkg} missing from Requires-Dist entirely: {requires_dist}"
        assert any(e.marker is None for e in entries), (
            f"{pkg} should have an unconditional Requires-Dist (no extra marker), "
            f"got: {[str(e) for e in entries]}"
        )

    for pkg, extra in (
        ("realhand", "l6"),
        ("dex-retargeting", "research"),
        ("mujoco", "research"),
        ("torch", "research"),
        ("wuji-sdk", "research"),
    ):
        entries = entries_for(pkg)
        assert entries, f"{pkg} missing from Requires-Dist entirely: {requires_dist}"
        assert all(e.marker is not None and f'extra == "{extra}"' in str(e.marker) for e in entries), (
            f"{pkg} must appear ONLY behind extra {extra!r}, got: {[str(e) for e in entries]}"
        )


@pytest.fixture(scope="module")
def wheel_venv_python(built_wheel, tmp_path_factory) -> Path:
    """A FRESH virtualenv -- never this repo's .venv, never editable -- with
    `built_wheel` installed non-editably. This is the whole point of the test:
    only a real separate-interpreter, separate-site-packages install exercises
    __file__ resolving into site-packages rather than the repo checkout, which
    is the exact blind spot an editable install (or this repo's own dev venv)
    papers over.
    """
    venv_dir = tmp_path_factory.mktemp("venv") / "venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True, text=True,
    )
    assert created.returncode == 0, (
        f"fresh venv creation failed:\nstdout={created.stdout}\nstderr={created.stderr}"
    )
    py = _venv_python(venv_dir)

    # Strip PYTHONPATH for anything run against this venv (see module
    # docstring): a venv does not clear an ambient PYTHONPATH on its own, and
    # this test's isolation should not depend on nothing on that path ever
    # shadowing `prehensile`. Also run with cwd=venv_dir, never REPO_ROOT: a
    # subprocess's cwd is NOT the same isolation knob as PYTHONPATH, but it
    # matters just as much here -- `python -c ...` (used below) puts '' (cwd)
    # first on sys.path, and cwd=REPO_ROOT would let Python resolve `import
    # prehensile` from the repo checkout itself before ever consulting the
    # venv's site-packages, silently defeating the entire point of this test.
    # Caught by hand while writing this test: it happened on the very first
    # run, well hidden, because it *looked* like a normal AssertionError.
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run_kwargs = dict(capture_output=True, text=True, env=clean_env, cwd=str(venv_dir))

    install = subprocess.run(
        [str(py), "-m", "pip", "install", "--no-cache-dir",
         "--disable-pip-version-check", str(built_wheel)],
        **run_kwargs,
    )
    if install.returncode != 0:
        # No network for numpy/protobuf et al in this sandbox. resolve_tuning
        # itself only needs pyyaml at import time, but it imports
        # prehensile.command (for L6_SDK_ORDER), which needs numpy -- fall
        # back to --no-deps plus an explicit install of just those two, rather
        # than silently weakening the test by skipping the failure outright.
        fallback = subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir",
             "--disable-pip-version-check", "--no-deps", str(built_wheel)],
            **run_kwargs,
        )
        if fallback.returncode != 0:
            pytest.skip(
                "fresh-venv wheel install failed even with --no-deps (no network and "
                f"no usable local wheel cache):\n{fallback.stderr}"
            )
        deps = subprocess.run(
            [str(py), "-m", "pip", "install", "--no-cache-dir",
             "--disable-pip-version-check", "pyyaml", "numpy"],
            **run_kwargs,
        )
        if deps.returncode != 0:
            pytest.skip(
                "could not install pyyaml/numpy for the --no-deps fallback -- fully "
                f"offline sandbox with no local cache:\n{deps.stderr}"
            )
    return py


def test_resolve_tuning_returns_real_tuning_after_fresh_non_editable_install(wheel_venv_python):
    """THE regression guard: resolve_tuning(DEFAULT_TUNING_PATH) must return
    the real shipped tuning -- not None, and not just "truthy" -- when
    prehensile is installed normally (non-editable) into a venv that has never
    seen this repo checkout.

    Runs as a subprocess IN the fresh venv (not via sys.path surgery from this
    process): only a real separate interpreter with its own site-packages
    exercises __file__ resolving under site-packages rather than the repo,
    which is the exact path the bug was hiding on.
    """
    script = textwrap.dedent(f"""
        from pathlib import Path
        from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning

        # Prove this is really the fresh venv's install, not the repo checkout --
        # a from-repo __file__ resolution sneaking in here would silently restore
        # the exact blind spot this test exists to close.
        repo_root = {str(REPO_ROOT)!r}
        assert not str(DEFAULT_TUNING_PATH).startswith(repo_root), DEFAULT_TUNING_PATH
        assert "site-packages" in str(DEFAULT_TUNING_PATH), DEFAULT_TUNING_PATH
        assert DEFAULT_TUNING_PATH.exists(), DEFAULT_TUNING_PATH

        tuning = resolve_tuning(DEFAULT_TUNING_PATH, side="left", require_couple_low=True)
        assert isinstance(tuning, dict), tuning
        assert tuning["thumb_flex"]["couple_low"] == 30, tuning["thumb_flex"]
        print("PACKAGING_CHECK_OK")
    """)
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # cwd MUST NOT be REPO_ROOT: `python -c` puts '' (cwd) first on sys.path,
    # so running from the repo checkout would resolve `import prehensile` from
    # the source tree before ever reaching the venv's site-packages -- see the
    # long comment in the wheel_venv_python fixture, where this was caught.
    venv_dir = wheel_venv_python.parent.parent
    check = subprocess.run(
        [str(wheel_venv_python), "-c", script],
        capture_output=True, text=True, env=clean_env, cwd=str(venv_dir),
    )
    assert check.returncode == 0, (
        f"resolve_tuning check failed in the fresh venv:\n"
        f"stdout={check.stdout}\nstderr={check.stderr}"
    )
    assert "PACKAGING_CHECK_OK" in check.stdout, check.stdout
