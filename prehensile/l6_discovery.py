"""Discovery/resolution core for RealHand L6 hands on CAN.

Each L6 hand burns its side into the device: it answers a 1-byte angle-sense
request ONLY at its own arbitration ID -- left=0x28, right=0x27
(realhand/hand/l6/l6.py hardcodes these, no override). Verified on real
hardware 2026-07-31 across three physical configurations: the known-left hand
answered 0x28 in ~1 ms and stayed silent on 0x27 (2 x 100 ms), mirror image
for the right hand, with berr-counters unchanged on both buses before and
after. Since both hand adapters are identical XCAN-USB (0c72:000c) with no
serial, and udev only ever knows the USB port (never which hand is plugged
into it), probing is the only way to tell left from right.

Two consumers of this module:
  - `tools/bring_up_hand.py` -- the CLI that brings a link up, probes it,
    renames it to the canonical hand_l/hand_r, and reports. Run by a human
    after every reboot or adapter replug.
  - `hand_teleop_node` (ROS; a separate task) -- resolves its own CAN
    interfaces at startup by calling `resolve_hands()` directly. No renaming
    involved there, so it needs only the probe, not the rest of the tool.

Enumeration in this module is sysfs-only -- no `ip` subprocess anywhere below
-- because the node's container may not have iproute2 installed, and sysfs
alone is verified sufficient: `/sys/class/net/<if>/type` == 280 (ARPHRD_CAN)
picks out CAN netdevs, `.../flags` bit 0x1 (IFF_UP) gives link state, and
`.../tx_queue_len` gives the queue length. Verified live 2026-07-31: `hand_l`
reads `type=280 flags=0x40081 qlen=1000`. `tools/bring_up_hand.py` still uses
`ip` itself for the things sysfs can't do: link mutation and berr-counters.

`realhand` is imported lazily, inside `probe_side()`, never at module level:
this module must be importable by the node without pulling in the SDK (the
node's own lazy-import discipline, matched here on purpose).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

HAND_VIDPID = "0c72:000c"  # XCAN-USB: both L6 hand adapters. Arms are 1d50:606f.
ARB_ID = {"left": 0x28, "right": 0x27}  # burned into the device; SDK hardcodes them

_ARPHRD_CAN = 280  # linux/if_arp.h
_IFF_UP = 0x1  # linux/if.h
_DEFAULT_NET_ROOT = Path("/sys/class/net")


# --------------------------------------------------------------------------- #
# sysfs readers -- no `ip` subprocess. Each takes `net_root` so tests can
# point these at a tmp_path fake tree instead of the real /sys/class/net.
# --------------------------------------------------------------------------- #
def can_netdevs(net_root: Path = _DEFAULT_NET_ROOT) -> list[str]:
    """Names of CAN netdevs (type file reads 280, ARPHRD_CAN) under net_root."""
    if not net_root.is_dir():
        return []
    names = []
    for entry in sorted(net_root.iterdir(), key=lambda p: p.name):
        try:
            if int((entry / "type").read_text().strip()) == _ARPHRD_CAN:
                names.append(entry.name)
        except (OSError, ValueError):
            continue
    return names


def is_up(ifname: str, net_root: Path = _DEFAULT_NET_ROOT) -> bool:
    """True if IFF_UP (bit 0x1) is set in `ifname`'s flags file."""
    try:
        flags = int((net_root / ifname / "flags").read_text().strip(), 16)
    except (OSError, ValueError):
        return False
    return bool(flags & _IFF_UP)


def qlen(ifname: str, net_root: Path = _DEFAULT_NET_ROOT) -> int:
    """`ifname`'s tx_queue_len (0 if the file is missing/unreadable)."""
    try:
        return int((net_root / ifname / "tx_queue_len").read_text().strip())
    except (OSError, ValueError):
        return 0


def usb_info(ifname: str, net_root: Path = _DEFAULT_NET_ROOT) -> dict[str, str]:
    """Walk sysfs from the netdev to its USB device: vid:pid, port path, product.

    `net_root/ifname/device` symlinks to the USB *interface* node; idVendor /
    idProduct / product sit a few parents up, at the USB *device* node -- so
    walk up looking for them (bounded: real sysfs is never this deep).
    """
    d = (net_root / ifname / "device").resolve()
    for _ in range(6):
        if (d / "idVendor").exists():
            def rd(name: str) -> str:
                p = d / name
                return p.read_text().strip() if p.exists() else "?"

            return {"vidpid": f"{rd('idVendor')}:{rd('idProduct')}",
                    "port": d.name, "product": rd("product")}
        d = d.parent
    return {"vidpid": "?", "port": "?", "product": "?"}


def hand_adapters(net_root: Path = _DEFAULT_NET_ROOT) -> list[str]:
    """CAN netdevs sitting on an XCAN-USB hand adapter (never an arm bus)."""
    return [n for n in can_netdevs(net_root) if usb_info(n, net_root)["vidpid"] == HAND_VIDPID]


# --------------------------------------------------------------------------- #
# The probe itself: the only piece that talks to the hand. Read-only, quiet.
# --------------------------------------------------------------------------- #
def probe_side(
    ifname: str,
    side: str,
    timeout_ms: float = 100.0,  # hardware replies in ~1 ms (measured 2026-07-31); generous
    attempts: int = 2,
) -> tuple[bool, str]:
    """Quietly ask `ifname` whether a `side` hand is present.

    Sends only the 1-byte angle sense request at that side's arbitration ID and
    waits for a reply. Read-only: the 7-byte frame that commands motion is
    never sent. Returns (answered, human-readable detail).
    """
    # Lazy import: callers (in particular hand_teleop_node) must be able to
    # `import prehensile.l6_discovery` without pulling in realhand's SDK.
    # Only an actual probe -- this function -- needs it.
    from realhand import L6

    try:
        l6 = L6(side, ifname)
    except Exception as exc:  # interface down/missing, SocketCAN error, ...
        return False, f"open failed: {type(exc).__name__}: {exc}"
    try:
        l6.stop_polling()  # quiet bus: kill the auto-started poll threads first
        last = "no reply"
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                data = l6.angle.get_blocking(timeout_ms=timeout_ms)
                dt = (time.monotonic() - t0) * 1e3
                angles = [round(a, 1) for a in data.angles.to_list()]
                return True, f"reply in {dt:.0f}ms (attempt {attempt}) angles={angles}"
            except TimeoutError:
                last = f"timeout x{attempt} ({timeout_ms:.0f}ms each)"
            except Exception as exc:
                return False, f"error: {type(exc).__name__}: {exc}"
        return False, last
    finally:
        try:
            l6.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Top-level entry point: resolve a set of requested sides to interfaces.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterInfo:
    """One hand adapter seen during enumeration, whether or not it answered."""

    ifname: str
    usb_port: str
    up: bool


@dataclass(frozen=True)
class ResolvedHand:
    """A side that resolved to exactly one interface."""

    ifname: str
    usb_port: str
    detail: str  # probe_side's human-readable detail line


@dataclass(frozen=True)
class HandResolution:
    """Result of `resolve_hands()`."""

    by_side: dict[str, ResolvedHand | None]  # requested side -> hit or None
    adapters: list[AdapterInfo]  # every hand adapter seen (up or not)
    conflicts: dict[str, list[str]]  # side -> ifnames, when >1 answered


def resolve_hands(
    sides: Iterable[str],
    timeout_ms: float = 100.0,
    attempts: int = 2,
    *,
    net_root: Path = _DEFAULT_NET_ROOT,
    _probe: Callable[[str, str, float, int], tuple[bool, str]] | None = None,
) -> HandResolution:
    """Resolve each of `sides` ("left"/"right") to its CAN interface.

    Enumerates hand adapters (sysfs only), then probes the full matrix of UP
    adapters x requested sides -- bounded in practice (<=3 adapters x 2
    sides), so at most 6 probes, each ~1 ms plus timeout on a miss. A side
    that answers on exactly one adapter resolves into `by_side`; a side that
    answers on >=2 adapters is ambiguous and is recorded in `conflicts`
    instead, with `by_side[side]` left `None`. DOWN adapters are never probed
    (an L6 open would just fail) but are still listed in `adapters`, so
    callers can name them in error/log messages (e.g. "found hand_l but it's
    down"). An adapter that answers no requested side simply resolves
    nothing for anyone; that alone is not an error.

    `_probe` is `probe_side` by default; tests inject a fake here so this can
    be exercised with no realhand and no CAN traffic.
    """
    probe = _probe if _probe is not None else probe_side
    sides = list(sides)

    ifnames = hand_adapters(net_root)
    adapters = [
        AdapterInfo(ifname=n, usb_port=usb_info(n, net_root)["port"], up=is_up(n, net_root))
        for n in ifnames
    ]
    up_adapters = [a.ifname for a in adapters if a.up]
    port_of = {a.ifname: a.usb_port for a in adapters}

    by_side: dict[str, ResolvedHand | None] = {}
    conflicts: dict[str, list[str]] = {}
    for side in sides:
        hits: list[tuple[str, str]] = []
        for ifname in up_adapters:
            ok, detail = probe(ifname, side, timeout_ms=timeout_ms, attempts=attempts)
            if ok:
                hits.append((ifname, detail))
        if len(hits) == 1:
            ifname, detail = hits[0]
            by_side[side] = ResolvedHand(ifname=ifname, usb_port=port_of[ifname], detail=detail)
        else:
            by_side[side] = None
            if len(hits) >= 2:
                conflicts[side] = [ifname for ifname, _ in hits]

    return HandResolution(by_side=by_side, adapters=adapters, conflicts=conflicts)
