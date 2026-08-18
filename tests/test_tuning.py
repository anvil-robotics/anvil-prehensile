"""TDD validators for prehensile/tuning.py's resolve_tuning safety gate.

Pure-synthetic, filesystem-only (tmp_path): no glove/robot/hardware needed.
Covers the ``require_couple_low`` contract added when the CLI's --tune/
--no-tune/--calibrate flags were removed and configs/curl_tuning.yml became
the single source of truth -- a missing ``thumb_flex.couple_low`` used to
silently mean 0 (closing the thumb's full travel into a parked thumb_abd);
now the --map curl path refuses to run without it.
"""

import pytest

from prehensile.tuning import resolve_tuning


def _write(tmp_path, text: str):
    p = tmp_path / "curl_tuning.yml"
    p.write_text(text)
    return p


def test_require_couple_low_raises_when_thumb_flex_couple_low_is_absent(tmp_path):
    # A file that tunes other channels but never sets thumb_flex.couple_low.
    path = _write(tmp_path, "index: {gain: 1.2}\n")
    with pytest.raises(ValueError, match="couple_low"):
        resolve_tuning(path, side="left", require_couple_low=True)

    # Same failure mode when the file is entirely absent.
    missing = tmp_path / "does_not_exist.yml"
    with pytest.raises(ValueError):
        resolve_tuning(missing, side="left", require_couple_low=True)

    # Without the gate, both of the above are fine (existing lenient default).
    assert resolve_tuning(path, side="left") == {"index": {"gain": 1.2}}
    assert resolve_tuning(missing, side="left") is None


def test_require_couple_low_passes_when_present(tmp_path):
    path = _write(tmp_path, "thumb_flex: {couple_low: 30}\n")
    tuning = resolve_tuning(path, side="left", require_couple_low=True)
    assert tuning["thumb_flex"]["couple_low"] == 30.0

    # Also satisfied when couple_low only comes from a per-side section.
    side_path = _write(
        tmp_path,
        "right:\n  thumb_flex: {couple_low: 15}\n",
    )
    side_tuning = resolve_tuning(side_path, side="right", require_couple_low=True)
    assert side_tuning["thumb_flex"]["couple_low"] == 15.0
