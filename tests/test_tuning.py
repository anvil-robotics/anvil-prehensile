"""Tests for prehensile/tuning.py -- the curl-map tuning file loader/resolver.

Pure-synthetic, filesystem-only (tmp_path): no glove/robot/hardware needed.

Two things are covered here that grew up in different checkouts and are both
required:

  * the loader/resolver contract -- valid keys, per-side sections, the per-KEY
    side merge, and the channel-scoping of ``couple_low`` / non-scoping of
    ``home_gesture``;
  * the ``require_couple_low`` safety gate added when the CLI's --tune/
    --no-tune/--calibrate flags were removed and configs/curl_tuning.yml became
    the single source of truth -- a missing ``thumb_flex.couple_low`` used to
    silently mean 0 (closing the thumb's full travel into a parked thumb_abd);
    now the --map curl path refuses to run without it.
"""

import pytest

from prehensile.tuning import DEFAULT_TUNING_PATH, load_curl_tuning, resolve_tuning


def _write(tmp_path, text: str):
    p = tmp_path / "curl_tuning.yml"
    p.write_text(text)
    return p


# -- loader: channels, keys, coercion ----------------------------------------- #


def test_load_valid(tmp_path):
    p = _write(tmp_path, "thumb_flex: {gain: 1.5, pivot: 70}\nindex: {gain: 2.0}\n")
    t = load_curl_tuning(p)
    assert t["thumb_flex"] == {"gain": 1.5, "pivot": 70.0}
    assert t["index"] == {"gain": 2.0}


def test_load_all_six_channels(tmp_path):
    body = "\n".join(f"{c}: {{gain: 1.0, pivot: 50}}"
                     for c in ("thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"))
    t = load_curl_tuning(_write(tmp_path, body))
    assert set(t) == {"thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"}


def test_load_unknown_channel_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown channel"):
        load_curl_tuning(_write(tmp_path, "thumbflex: {gain: 1.5}\n"))


def test_load_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        load_curl_tuning(_write(tmp_path, "index: {slope: 2.0}\n"))


def test_load_flip_key_is_bool(tmp_path):
    p = _write(tmp_path, "thumb_flex: {flip: true}\n")
    t = load_curl_tuning(p)
    assert t["thumb_flex"]["flip"] is True


def test_load_park_key(tmp_path):
    p = _write(tmp_path, "thumb_flex: {park: 15}\n")
    t = load_curl_tuning(p)
    assert t["thumb_flex"]["park"] == 15.0
    assert isinstance(t["thumb_flex"]["park"], float)


def test_load_alpha_key(tmp_path):
    """``alpha`` is a per-channel EMA smoothing override, floated like the rest."""
    p = _write(tmp_path, "thumb_abd: {alpha: 0.1}\n")
    t = load_curl_tuning(p)
    assert t["thumb_abd"]["alpha"] == 0.1
    assert isinstance(t["thumb_abd"]["alpha"], float)


# -- resolver: missing file / per-side merge ---------------------------------- #


def test_resolve_none_path_and_no_overrides_is_none():
    assert resolve_tuning(None) is None


def test_resolve_missing_file_is_none(tmp_path):
    assert resolve_tuning(tmp_path / "nope.yml") is None


def test_load_side_section(tmp_path):
    p = _write(tmp_path, "right:\n  thumb_flex: {flip: true}\n")
    t = load_curl_tuning(p)
    assert t["right"]["thumb_flex"] == {"flip": True}


def test_load_unknown_channel_in_side_section_raises(tmp_path):
    p = _write(tmp_path, "right:\n  thumbflex: {flip: true}\n")
    with pytest.raises(ValueError, match="unknown channel"):
        load_curl_tuning(p)


def test_resolve_merges_side_over_shared(tmp_path):
    p = _write(tmp_path, "thumb_flex: {gain: 2.5, pivot: 80}\nright:\n  thumb_flex: {flip: true}\n")
    assert resolve_tuning(p, side="right")["thumb_flex"] == {"gain": 2.5, "pivot": 80.0, "flip": True}
    resolved_left = resolve_tuning(p, side="left")["thumb_flex"]
    assert resolved_left == {"gain": 2.5, "pivot": 80.0}
    assert "left" not in resolve_tuning(p, side="left")
    assert "right" not in resolve_tuning(p, side="left")


# -- couple_low: the channel-scoped key --------------------------------------- #


def test_load_couple_low_key(tmp_path):
    """couple_low is a valid per-channel key, floated like gain/pivot/park."""
    p = _write(tmp_path, "thumb_flex: {couple_low: 30}\n")
    t = load_curl_tuning(p)
    assert t["thumb_flex"]["couple_low"] == 30.0
    assert isinstance(t["thumb_flex"]["couple_low"], float)


def test_resolve_merges_couple_low_per_side(tmp_path):
    """A side section can override couple_low for one hand only, like any other key."""
    p = _write(tmp_path, "thumb_flex: {couple_low: 30}\nright:\n  thumb_flex: {couple_low: 45}\n")
    assert resolve_tuning(p, side="left")["thumb_flex"]["couple_low"] == 30.0
    assert resolve_tuning(p, side="right")["thumb_flex"]["couple_low"] == 45.0


def test_resolve_keeps_shared_couple_low_under_a_side_flip_override(tmp_path):
    """The side merge is per-KEY, so a side section that only sets `flip` must not
    drop the shared couple_low -- this is the shipped config's exact shape."""
    p = _write(tmp_path,
               "thumb_flex: {gain: 2.5, pivot: 80, couple_low: 30}\n"
               "right:\n  thumb_flex: {flip: false}\n")
    assert resolve_tuning(p, side="right")["thumb_flex"]["couple_low"] == 30.0


@pytest.mark.parametrize("channel", ["thumb_flex", "index"])
def test_couple_low_accepted_on_both_coupling_channels(tmp_path, channel):
    """couple_low is meaningful on thumb_flex (the driven channel's output floor)
    and on index (the driving finger's own command floor)."""
    p = _write(tmp_path, f"{channel}: {{couple_low: 20}}\n")
    assert load_curl_tuning(p)[channel]["couple_low"] == 20.0


@pytest.mark.parametrize("channel", ["thumb_abd", "middle", "ring", "pinky"])
def test_couple_low_rejected_on_channels_outside_the_coupling(tmp_path, channel):
    """couple_low means nothing on the other channels, so accepting it there would
    be a silent no-op -- exactly what this module's loud-failure contract forbids."""
    p = _write(tmp_path, f"{channel}: {{couple_low: 30}}\n")
    with pytest.raises(ValueError, match="couple_low"):
        load_curl_tuning(p)


def test_couple_low_rejected_on_other_channel_inside_a_side_section(tmp_path):
    """The same restriction applies inside a left:/right: section."""
    p = _write(tmp_path, "right:\n  pinky: {couple_low: 30}\n")
    with pytest.raises(ValueError, match="couple_low"):
        load_curl_tuning(p)


# -- home_gesture: valid on EVERY channel (not channel-scoped) ---------------- #


@pytest.mark.parametrize(
    "channel", ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]
)
def test_home_gesture_key_accepted_on_every_channel(tmp_path, channel):
    """home_gesture is valid on every channel (unlike couple_low, it is NOT
    channel-scoped) and is floated like gain/pivot/park."""
    p = _write(tmp_path, f"{channel}: {{home_gesture: 30}}\n")
    t = load_curl_tuning(p)
    assert t[channel]["home_gesture"] == 30.0
    assert isinstance(t[channel]["home_gesture"], float)


def test_shipped_config_carries_home_gesture_for_both_sides():
    """The shipped config still parses for both sides, and every channel now
    carries an explicit home_gesture value."""
    for side in ("left", "right"):
        tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=side)
        for channel in ("thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"):
            assert "home_gesture" in tuning[channel], f"{side}:{channel}"


# -- the shipped config + the require_couple_low safety gate ------------------ #


def test_shipped_config_parses():
    """The config that actually ships must load -- nothing else in the suite reads
    it, so a bad edit to it would otherwise only surface at runtime."""
    tuning = load_curl_tuning(DEFAULT_TUNING_PATH)
    assert "thumb_flex" in tuning
    assert resolve_tuning(DEFAULT_TUNING_PATH, side="right")["thumb_flex"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_shipped_config_satisfies_the_require_couple_low_gate(side):
    """The gate must still pass against the REAL shipped config for both hands --
    including now that every channel also carries a home_gesture key, which must
    not disturb the couple_low the gate looks for."""
    tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=side, require_couple_low=True)
    assert tuning["thumb_flex"]["couple_low"] == 30.0


def test_require_couple_low_raises_when_thumb_flex_couple_low_is_absent(tmp_path):
    """The file exists and loads fine, it just never sets thumb_flex.couple_low --
    this is the gate's third raise branch (C), distinct from the file being
    missing/unreadable (branches B/A below). Pinned to C's own message so a bug
    that removes this branch and lets the call fall through to some other raise
    cannot masquerade as this one still passing."""
    # A file that tunes other channels but never sets thumb_flex.couple_low.
    path = _write(tmp_path, "index: {gain: 1.2}\n")
    with pytest.raises(ValueError, match="missing required 'couple_low'"):
        resolve_tuning(path, side="left", require_couple_low=True)

    # Without the gate this is fine (existing lenient default).
    assert resolve_tuning(path, side="left") == {"index": {"gain": 1.2}}


def test_require_couple_low_raises_when_file_is_missing(tmp_path):
    """The gate's second raise branch (B): the tuning file does not exist at all.
    Pinned to B's own message -- a bare ``pytest.raises(ValueError)`` here would
    also pass if B were deleted and the call fell through to branch C's
    couple_low check (an empty `tuning` dict has no thumb_flex.couple_low
    either), which would hide B's removal entirely."""
    missing = tmp_path / "does_not_exist.yml"
    with pytest.raises(ValueError, match="file not found"):
        resolve_tuning(missing, side="left", require_couple_low=True)

    # Without the gate this is fine (existing lenient default).
    assert resolve_tuning(missing, side="left") is None


def test_require_couple_low_raises_when_file_is_unreadable(tmp_path):
    """The gate's first raise branch (A): the path exists but reading it raises
    OSError, which is promoted to ValueError naming the path. A directory is the
    clean way to trigger this: ``Path.exists()`` is True for it, but ``open()``
    raises ``IsADirectoryError``, an ``OSError`` subclass."""
    path = tmp_path / "curl_tuning.yml"
    path.mkdir()
    with pytest.raises(ValueError, match="could not be read") as exc_info:
        resolve_tuning(path, side="left", require_couple_low=True)
    assert str(path) in str(exc_info.value)


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
