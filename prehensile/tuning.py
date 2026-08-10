"""Load + resolve the curl-map response tuning file (configs/curl_tuning.yml).

A tiny YAML config of per-channel ``{gain, pivot, flip}`` for the ``--map curl``
path, so the CurlMapper response can be tuned by editing a file instead of
passing CLI flags every run. Kept out of ``curlmap.py`` so that module stays
pure-numpy and yaml-free (importable for the hardware-free tests).

Channel names are the six L6 slots (``command.L6_SDK_ORDER``); the valid keys
are ``gain``, ``pivot``, ``alpha``, ``flip``, ``park`` and ``couple_low``
(thumb_abd's invert sense is set in ``profiles.py``, not here). ``flip``
complements the channel's output (``100 - x``); it is the knob for a thumb
that is mounted/oriented backwards on one hand. ``park`` is a
native-convention 0-100 value a channel is forced to while the mapper is
locked (see ``CurlMapper.locked``/``set_park``), e.g. a grasp thumb posture.
``couple_low`` belongs to the thumb<-index coupling (see
``CurlMapper.couple_thumb_index``) and is valid on **only** its two channels --
``thumb_flex`` (the driven channel's output floor) and ``index`` (the driving
finger's own command floor) -- see ``CHANNEL_SCOPED_KEYS``. Top-level entries
may also be a ``left``/``right`` side section (itself a ``{channel: params}``
map) that ``resolve_tuning`` merges on top of the shared channels for that
side only; the merge is per-KEY, so a side section that sets only ``flip``
keeps the shared ``couple_low``. Unknown channels/keys -- and a
channel-scoped key on the wrong channel -- raise so typos fail loudly.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from prehensile.command import L6_SDK_ORDER

# Shipped tuning file, resolved relative to the installed package so both the CLI
# (which has its own ROOT) and the ROS hand node (which does not) can find it.
DEFAULT_TUNING_PATH = Path(__file__).resolve().parent.parent / "configs" / "curl_tuning.yml"

VALID_CHANNELS = set(L6_SDK_ORDER)
VALID_KEYS = {"gain", "pivot", "alpha", "flip", "park", "couple_low"}
VALID_SIDES = {"left", "right"}
# Keys that only mean something on SPECIFIC channels. Accepting them elsewhere
# would be a silent no-op -- exactly what this module's loud-failure contract
# exists to prevent -- so they are rejected at load time instead. ``couple_low``
# is valid on the coupling's two channels: on ``thumb_flex`` it is the driven
# channel's output floor, on ``index`` it is the driving finger's own command
# floor (see ``CurlMapper.couple_thumb_index``).
CHANNEL_SCOPED_KEYS = {"couple_low": frozenset({"thumb_flex", "index"})}


def _coerce_params(params, where: str, channel: str) -> dict[str, float | bool]:
    """Validate ``params``' keys against ``VALID_KEYS`` and coerce each value:
    ``gain``/``pivot``/``alpha``/``park``/``couple_low`` -> ``float``,
    ``flip`` -> ``bool`` (NOT floated, so it stays a real bool). ``channel`` is the owning channel
    name, used to enforce ``CHANNEL_SCOPED_KEYS``. ``where`` is a
    caller-supplied description of the params' location, used in the raised
    message."""
    if not isinstance(params, dict):
        raise ValueError(f"{where} must map to params, got {type(params).__name__}")
    bad = set(params) - VALID_KEYS
    if bad:
        raise ValueError(f"{where}: unknown key(s) {sorted(bad)}; valid: {sorted(VALID_KEYS)}")
    for key, only_on in CHANNEL_SCOPED_KEYS.items():
        if key in params and channel not in only_on:
            raise ValueError(
                f"{where}: {key!r} only applies to channel(s) {sorted(only_on)}, "
                f"not {channel!r}"
            )
    return {k: (bool(v) if k == "flip" else float(v)) for k, v in params.items()}


def load_curl_tuning(path) -> dict[str, dict]:
    """Parse a curl-tuning YAML into ``{channel: {"gain"|"pivot": float,
    "flip": bool}}``, with optional per-side sections.

    Each top-level key must be either a valid channel name (as above) or a
    side name (``left``/``right``) whose value is itself a ``{channel:
    params}`` mapping restricted to the same channels. Raises ``ValueError`` on
    an unknown top-level key, an unknown channel inside a side section, or an
    unknown param key, so a typo fails loudly instead of silently doing
    nothing."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping channel -> params, got {type(data).__name__}")
    out: dict[str, dict] = {}
    for key, value in data.items():
        if key in VALID_SIDES:
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: side {key!r} must map to {{channel: params}}, got {type(value).__name__}"
                )
            side_out: dict[str, dict] = {}
            for ch, params in value.items():
                if ch not in VALID_CHANNELS:
                    raise ValueError(
                        f"{path}: unknown channel {ch!r} in side {key!r}; valid: {sorted(VALID_CHANNELS)}"
                    )
                side_out[ch] = _coerce_params(
                    params, f"{path}: channel {ch!r} in side {key!r}", ch
                )
            out[key] = side_out
        elif key in VALID_CHANNELS:
            out[key] = _coerce_params(value, f"{path}: channel {key!r}", key)
        else:
            raise ValueError(f"{path}: unknown channel {key!r}; valid: {sorted(VALID_CHANNELS)}")
    return out


def resolve_tuning(path, *, side=None, require_couple_low=False) -> dict | None:
    """Load ``path`` (if it exists) and merge ``side``'s section on top.

    Precedence: ``side``'s section > the shared top-level channels > built-in
    defaults. Returns the merged ``{channel: {...}}`` dict -- channel keys
    only, never ``left``/``right``, since the mapper consumes a flat per-side
    dict -- or ``None`` when there is nothing to apply (no readable file and
    no side section).

    ``require_couple_low`` is the ``--map curl`` path's safety gate: a missing
    ``thumb_flex.couple_low`` silently defaults to 0, which closes the thumb's
    full travel into a parked (tucked) ``thumb_abd`` -- straight into the
    operator's closing fingers. When ``require_couple_low`` is True this
    function instead raises ``ValueError`` naming ``path`` and the missing
    key, and the tuning file being missing or unreadable is likewise promoted
    from a silent fallback to a ``ValueError`` (an unrecoverable condition on
    that path, not a default to quietly fall back from)."""
    tuning: dict[str, dict] = {}
    path_obj = Path(path) if path is not None else None
    if path_obj is not None and path_obj.exists():
        try:
            loaded = load_curl_tuning(path_obj)
        except OSError as exc:
            if require_couple_low:
                raise ValueError(
                    f"{path}: curl tuning file exists but could not be read ({exc})"
                ) from exc
            raise
        for ch in VALID_CHANNELS:
            if ch in loaded:
                tuning[ch] = dict(loaded[ch])
        if side is not None and side in loaded:
            for ch, params in loaded[side].items():
                tuning.setdefault(ch, {}).update(params)
    elif require_couple_low:
        raise ValueError(
            f"{path}: curl tuning file not found; it is required on the --map curl path "
            "because thumb_flex.couple_low has no safe silent default"
        )

    if require_couple_low and "couple_low" not in tuning.get("thumb_flex", {}):
        raise ValueError(
            f"{path}: missing required 'couple_low' on the 'thumb_flex' channel -- without it "
            "the thumb closes its full travel into a parked thumb_abd, into the operator's "
            "closing fingers"
        )
    return tuning or None
