"""Load a ``HandDescriptor`` from a YAML hand-descriptor file.

Kept out of ``prehensile/hand.py`` (stdlib-only dataclasses) and out of
``prehensile/curlmap.py`` (pure-numpy, no yaml) -- see both modules' docstrings.
Nothing here is imported by ``curlmap.py``; a caller that wants to drive a hand
other than the shipped default (``prehensile.curlmap.L6_HAND``) loads one with
``load_hand_descriptor`` and passes the result as ``CurlMapper(..., hand=...)``.

Schema (``schema: prehensile.hand/1``), matching ``prehensile.hand``'s
dataclasses field-for-field:

    schema: prehensile.hand/1
    name: my_hand
    channels:
      - {name: <sdk slot name>, role: <one of prehensile.hand.ROLES>, home: <0-100>}
      ...
    grasp:                      # optional; omit for the Pinch()/("middle","ring","pinky") defaults
      pinch: {driver: <role>, driven: <role>}   # or `pinch: null` for no pinch coupling
      group: [<role>, ...]
    output:                     # optional; every key optional; UNCONSUMED until Phase 4b
      units: percent
      open: 100.0
      closed: 0.0
      driver: "module.path:attr"
    tuning:
      default: <path, interpreted by the caller -- this module does not resolve it>
    driver_joints:               # optional
      <role or channel name>: <urdf joint name>

Every raise here names the offending file/channel, matching
``prehensile.tuning``'s loud-failure style: a missing/unknown key is a load
error, never a silent default -- ``HandDescriptor.__post_init__`` enforces the
same contract at the dataclass level for anyone constructing one directly
(without going through this loader) too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from prehensile.hand import Channel, HandDescriptor, Output, Pinch

SCHEMA = "prehensile.hand/1"

_UNSET = object()  # distinguishes an absent `grasp.pinch` key from an explicit `pinch: null`


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _load_channels(raw: Any, path) -> tuple[Channel, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: 'channels' must be a non-empty list")
    channels = []
    for i, entry in enumerate(raw):
        where = f"{path}: channels[{i}]"
        entry = _require_mapping(entry, where)
        name = entry.get("name")
        if not name:
            raise ValueError(f"{where}: missing required key 'name'")
        if "role" not in entry:
            raise ValueError(f"{path}: channel {name!r} is missing required key 'role'")
        if "home" not in entry:
            raise ValueError(f"{path}: channel {name!r} is missing required key 'home'")
        channels.append(Channel(name=name, role=entry["role"], home=float(entry["home"])))
    return tuple(channels)


def _load_pinch(grasp: dict, path) -> Pinch | None:
    raw = grasp.get("pinch", _UNSET)
    if raw is _UNSET:
        return Pinch()
    if raw is None:
        return None
    raw = _require_mapping(raw, f"{path}: grasp.pinch")
    driver = _require(raw, "driver", f"{path}: grasp.pinch")
    driven = _require(raw, "driven", f"{path}: grasp.pinch")
    return Pinch(driver=driver, driven=driven)


def _load_group(grasp: dict) -> tuple[str, ...]:
    raw = grasp.get("group", ("middle", "ring", "pinky"))
    return tuple(raw)


def _load_output(raw: Any, path) -> Output:
    if raw is None:
        return Output()
    raw = _require_mapping(raw, f"{path}: output")
    return Output(
        units=raw.get("units", "percent"),
        open=float(raw.get("open", 100.0)),
        closed=float(raw.get("closed", 0.0)),
        driver=raw.get("driver"),
    )


def load_hand_descriptor(path) -> HandDescriptor:
    """Parse ``path`` (a ``schema: prehensile.hand/1`` YAML file) into a ``HandDescriptor``.

    Raises ``ValueError`` naming the offending file/channel/section on any
    structural problem: a non-mapping document, an unsupported/missing schema,
    a missing 'name'/'channels', a channel missing 'name'/'role'/'home', or a
    malformed 'grasp'/'output' section. Everything else (unknown role,
    duplicate channel name, a pinch/group role no channel carries) is enforced
    by ``HandDescriptor.__post_init__`` itself, so it fires the same way
    whether the descriptor came from this loader or was constructed directly.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")

    schema = data.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: unsupported/missing schema {schema!r}; expected {SCHEMA!r}")

    name = _require(data, "name", str(path))
    channels = _load_channels(_require(data, "channels", str(path)), path)

    grasp = _require_mapping(data.get("grasp") or {}, f"{path}: grasp")
    pinch = _load_pinch(grasp, path)
    group = _load_group(grasp)

    output = _load_output(data.get("output"), path)

    tuning = _require_mapping(data.get("tuning") or {}, f"{path}: tuning")
    default_tuning = tuning.get("default")

    driver_joints = dict(_require_mapping(data.get("driver_joints") or {}, f"{path}: driver_joints"))

    return HandDescriptor(
        name=name,
        channels=channels,
        pinch=pinch,
        group=group,
        output=output,
        default_tuning=default_tuning,
        driver_joints=driver_joints,
    )
