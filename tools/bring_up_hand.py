#!/usr/bin/env python3
"""bring_up_hand.py -- find each L6 hand by asking the bus, then name it correctly.

Canonical names: the LEFT hand's interface is `hand_l`, the RIGHT hand's is
`hand_r`, no matter which USB port the adapter is plugged into. Config
(anvil-loader/.env.config) points at these names and never changes again.

Why probing works: the L6 burns its side into the device -- it answers sense
requests ONLY at its own arbitration ID, left=0x28 / right=0x27
(realhand/hand/l6/l6.py hardcodes these with no override). Verified on
hardware 2026-07-31: the known-left hand answered 0x28 in ~1 ms and stayed
silent on 0x27 (2 x 400 ms), mirror image for the right hand, with
berr-counters at tx=0 rx=0 before AND after on both buses.

Why renaming is needed at all: udev names an interface at device-add time,
when only the USB PORT is known -- and the two hand adapters are identical
XCAN-USB (0c72:000c) with no serial, so udev cannot tell left from right.
Identity is only knowable by probing, which needs the link up first. Hence
this tool: bring up -> probe -> rename. udev's port-based name (leader_r,
hand_r, can0, ...) survives only until this tool runs.

The ROS workcell does NOT need this tool -- hand_teleop_node auto-resolves
both interfaces itself at startup. Run it only (a) with --check to diagnose,
or (b) without --check before using the standalone CLI tools, which expect
the canonical hand_l / hand_r names (re-run after every reboot or adapter
replug -- udev re-applies port-based names and the rename does not persist).
Renaming needs CAP_NET_ADMIN: either run inside the dev container
(privileged; just works) or with sudo on the host.

What it does:
  1. enumerate CAN netdevs; keep only XCAN-USB hand adapters -- the arms'
     canable2 (1d50:606f) buses are never probed (see --force)
  2. bring up DOWN adapters (bitrate 1000000, txqueuelen 1000, up)
  3. raise txqueuelen to 1000 where it is still the kernel-default 10 (the
     root cause of the 2026-07-28 ENOBUFS cascade that disabled both arms)
  4. probe each adapter at 0x28/0x27: ONE 1-byte sense request per attempt
     (motion needs a 7-byte set_angles frame -- never sent), with the SDK's
     auto-polling stopped immediately (the other 2026-07-28 lesson)
  5. rename each detected hand's interface to its canonical name (two-phase
     via a temp name if the hands were physically swapped), then re-probe
     the renamed interface to confirm
  6. cross-check anvil-loader/.env.config (should say hand_l / hand_r)

Every mutation is attempted directly and, on EPERM, printed as the exact
sudo command to run instead.

NOTE: the discovery core this file used to define inline -- can_netdevs(),
usb_info(), hand_adapters(), probe_side(), HAND_VIDPID, ARB_ID -- now lives in
prehensile/l6_discovery.py; this file is the CLI wrapper around it (bring-up,
rename, and reporting stay here). hand_teleop_node imports that same module to
resolve its own interfaces at startup, independently of this tool.

Usage:
    tools/bring_up_hand.py            bring up + probe + rename + report
    tools/bring_up_hand.py --check    probe + report only; mutate nothing
    tools/bring_up_hand.py --list     enumerate adapters only, no CAN traffic
    tools/bring_up_hand.py IFACE...   probe exactly these interfaces
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# venv bootstrap: realhand lives in this repo's own .venv, or /prehensile_venv
# (dev container) -- never in system python. Re-exec if needed.
# --------------------------------------------------------------------------- #
def _bootstrap() -> None:
    try:
        import realhand  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".venv" / "bin" / "python",  # this package's own venv (future-proof)
        Path("/prehensile_venv/bin/python"),  # dev container venv
    ]
    for py in candidates:
        if py.exists() and sys.executable != str(py):
            os.execv(str(py), [str(py), str(here), *sys.argv[1:]])
    raise SystemExit(
        "realhand is not importable and no known venv was found; tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


_bootstrap()

# `prehensile` (this tree, editable-installed) is importable on the same venv
# `_bootstrap()` just ensured we are running under.
from prehensile.l6_discovery import (  # noqa: E402  (needs the bootstrap above)
    ARB_ID,
    HAND_VIDPID,
    can_netdevs,
    hand_adapters,
    is_up,
    probe_side,
    qlen,
    usb_info,
)

CANON = {"left": "hand_l", "right": "hand_r"}  # canonical interface names
BITRATE = 1_000_000
TXQUEUELEN = 1000  # kernel default 10 caused the 2026-07-28 ENOBUFS cascade

_REPO_ROOT = Path(__file__).resolve().parents[4]
ENV_CONFIG = _REPO_ROOT / "anvil-loader" / ".env.config"
ENV_KEY = {"left": "LEFT_HAND_CAN_INTERFACE", "right": "RIGHT_HAND_CAN_INTERFACE"}


# --------------------------------------------------------------------------- #
# Interface enumeration. can_netdevs/usb_info/hand_adapters are sysfs-only and
# now live in prehensile.l6_discovery (imported above). all_netdev_names() and
# bus_health() genuinely need iproute2 (full netdev list / berr-counters), so
# they stay here.
# --------------------------------------------------------------------------- #
def all_netdev_names() -> list[str]:
    out = subprocess.run(["ip", "-json", "link", "show"],
                         capture_output=True, text=True, check=True).stdout
    return [e["ifname"] for e in json.loads(out)]


def bus_health(ifname: str) -> str:
    try:
        out = subprocess.run(["ip", "-details", "-s", "link", "show", ifname],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        # `ip` (iproute2) isn't installed in every environment this tool runs in
        # (e.g. the ros2 dev container) -- degrade to a visible skip instead of
        # crashing; CAN mutation/probing below never needs it.
        return "bus health: skipped (`ip` not available -- fine inside the ros2 container)"
    m = re.search(r"can state (\S+).*?berr-counter tx (\d+) rx (\d+)", out, re.S)
    return f"{m.group(1)} tx={m.group(2)} rx={m.group(3)}" if m else "?"


# --------------------------------------------------------------------------- #
# Link mutations (work in the privileged dev container; on an unprivileged
# host each failed command is returned as the exact sudo line to run).
# --------------------------------------------------------------------------- #
def _ip_set(*args: str) -> str | None:
    """Run `ip link set ...`; return the sudo-prefixed command on failure."""
    cmd = ["ip", "link", "set", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return None if r.returncode == 0 else "sudo " + " ".join(cmd)


def ensure_ready(ifname: str, apply: bool) -> tuple[bool, list[str]]:
    """Bring the link up / fix txqueuelen. Returns (probe-able, suggestions)."""
    up, ql = is_up(ifname), qlen(ifname)
    sugg: list[str] = []
    if not up:
        cmds = ((ifname, "type", "can", "bitrate", str(BITRATE)),
                (ifname, "txqueuelen", str(TXQUEUELEN)),
                (ifname, "up"))
        if not apply:
            return False, ["would run: ip link set " + " ".join(c) for c in cmds]
        for args in cmds:
            fail = _ip_set(*args)
            if fail:
                sugg.append(fail)
        up = is_up(ifname)
    elif ql < TXQUEUELEN:
        note = f"   # qlen {ql} -> {TXQUEUELEN} (ENOBUFS fix)"
        if not apply:
            sugg.append(f"would run: ip link set {ifname} txqueuelen {TXQUEUELEN}{note}")
        else:
            fail = _ip_set(ifname, "txqueuelen", str(TXQUEUELEN))
            if fail:
                sugg.append(fail + note)
    return up, sugg


def rename_iface(old: str, new: str) -> list[str]:
    """down -> rename -> up. Empty list on success, else the remaining sudo cmds."""
    seq = [["ip", "link", "set", old, "down"],
           ["ip", "link", "set", old, "name", new],
           ["ip", "link", "set", new, "up"]]
    for i, cmd in enumerate(seq):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return ["sudo " + " ".join(c) for c in seq[i:]]
    return []


def plan_renames(desired: dict[str, str], in_use: set[str]) -> list[tuple[str, str]]:
    """Order renames so no step targets a name that is still taken.

    Handles the physically-swapped-hands case (hand_l <-> hand_r) by routing
    one side through a temporary name.
    """
    steps: list[tuple[str, str]] = []
    pending = dict(desired)
    names = set(in_use)
    while pending:
        progressed = False
        for old, new in list(pending.items()):
            if new == old or new not in names:
                steps.append((old, new))
                names.discard(old)
                names.add(new)
                del pending[old]
                progressed = True
        if not progressed:  # pure swap cycle: break it with a temp name
            old, new = next(iter(pending.items()))
            tmp = f"{new[:8]}_tmp"  # IFNAMSIZ limit is 15 chars
            while tmp in names:
                tmp += "0"
            steps.append((old, tmp))
            names.discard(old)
            names.add(tmp)
            del pending[old]
            pending[tmp] = new
    return steps


# --------------------------------------------------------------------------- #
# Config cross-check.
# --------------------------------------------------------------------------- #
def env_config_values() -> dict[str, str]:
    vals: dict[str, str] = {}
    if ENV_CONFIG.exists():
        for line in ENV_CONFIG.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            vals[k.strip()] = v.split("#", 1)[0].strip()
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ifaces", nargs="*",
                    help="probe exactly these interfaces (default: all hand adapters)")
    ap.add_argument("--list", action="store_true",
                    help="enumerate adapters only; send no CAN traffic")
    ap.add_argument("--check", action="store_true",
                    help="probe and report only; mutate nothing (no bring-up, no rename)")
    ap.add_argument("--force", action="store_true",
                    help="allow probing an interface that is not a hand adapter")
    args = ap.parse_args()

    print("CAN netdevs:")
    infos = {n: usb_info(n) for n in can_netdevs()}
    for name, u in infos.items():
        tag = "HAND adapter" if u["vidpid"] == HAND_VIDPID else "(arm bus -- not probed)"
        print(f"  {name:12} usb={u['port']:8} {u['vidpid']}  {u['product']:16} {tag}")
    print()

    targets = args.ifaces or hand_adapters()
    if args.list:
        print(f"--list: would probe {targets}; no CAN traffic sent.")
        return 0
    if not targets:
        print("No hand adapters found (and no interfaces given).")
        return 2

    found: dict[str, list[str]] = {"left": [], "right": []}
    port_of: dict[str, str] = {}
    problems: list[str] = []
    suggestions: list[str] = []

    for ifname in targets:
        vidpid = infos.get(ifname, usb_info(ifname))["vidpid"]
        if vidpid != HAND_VIDPID and not args.force:
            print(f"== {ifname} ==  SKIPPED: not an XCAN-USB hand adapter "
                  f"({vidpid}); refusing to probe an arm bus (--force to override)")
            print()
            continue

        ready, sugg = ensure_ready(ifname, apply=not args.check)
        print(f"== {ifname} ==  {bus_health(ifname)}")
        for s in sugg:
            print(f"  needs root:  {s}")
        suggestions += [s for s in sugg if not s.startswith("would run")]
        if not ready:
            problems.append(f"{ifname}: link is DOWN (run the commands above, then re-run)")
            print()
            continue

        for side in ("left", "right"):
            ok, detail = probe_side(ifname, side)
            mark = "ANSWERED" if ok else "silent  "
            print(f"  probe side={side:5} (0x{ARB_ID[side]:02X}): {mark}  {detail}")
            if ok:
                found[side].append(ifname)
                port_of[side] = infos.get(ifname, {}).get("port", "?")
        after = bus_health(ifname)
        print(f"                 after:  {after}")
        # "skipped" (no `ip`, e.g. inside the ros2 container) means unknown, not
        # unhealthy -- must not be reported as a bus-error problem.
        if "skipped" not in after and "tx=0 rx=0" not in after:
            problems.append(f"{ifname}: bus errors after probe ({after}) -- "
                            f"hand unpowered or wiring fault?")
        print()

    # --- canonical naming: rename each detected hand to hand_l / hand_r ----- #
    final: dict[str, str] = {s: f[0] for s, f in found.items() if len(f) == 1}
    desired = {cur: CANON[side] for side, cur in final.items() if cur != CANON[side]}

    if desired:
        if args.check:
            for old, new in desired.items():
                print(f"would rename: {old} -> {new}  (skipped, --check)")
            print()
        else:
            steps = plan_renames(desired, set(all_netdev_names()))
            aborted = False
            for idx, (old, new) in enumerate(steps):
                fails = rename_iface(old, new)
                if fails:
                    suggestions += fails
                    for old2, new2 in steps[idx + 1:]:
                        suggestions += [f"sudo ip link set {old2} down",
                                        f"sudo ip link set {old2} name {new2}",
                                        f"sudo ip link set {new2} up"]
                    aborted = True
                    break
                print(f"renamed: {old} -> {new}")
                for side, cur in final.items():
                    if cur == old:
                        final[side] = new
            if aborted:
                problems.append("rename needs root: run the commands below, then re-run "
                                "(or run this tool inside the dev container / with sudo)")
            else:
                # confirm the renamed interfaces still answer
                for side, iface in final.items():
                    if iface == CANON[side] and CANON[side] in desired.values():
                        ok, detail = probe_side(iface, side)
                        mark = "confirmed" if ok else "LOST CONTACT"
                        print(f"re-probe {iface} ({side}): {mark}  {detail}")
                        if not ok:
                            problems.append(f"{iface}: no reply after rename")
            print()

    # --- verdict ------------------------------------------------------------ #
    print("=" * 62)
    env = env_config_values()
    ok = True
    for side in ("left", "right"):
        n = len(found[side])
        if n == 1:
            iface = final[side]
            configured = env.get(ENV_KEY[side])
            bits = [f"usb {port_of.get(side, '?')}"]
            if iface != CANON[side]:
                bits.append(f"NOT canonical (want {CANON[side]})")
                ok = False
            if configured == iface:
                bits.append("matches .env.config")
            elif configured == "auto":
                bits.append(".env.config says auto (node resolves at startup) -- OK")
            else:
                bits.append(f".env.config says {configured!r}")
                ok = False
            print(f"  {side.upper():5} hand -> {iface:12} ({', '.join(bits)})")
        elif n == 0:
            print(f"  {side.upper():5} hand -> NOT FOUND (powered? cable seated?)")
            ok = False
        else:
            print(f"  {side.upper():5} hand -> AMBIGUOUS: answered on {found[side]}")
            ok = False
    for p in problems:
        print(f"  ! {p}")
        ok = False
    if suggestions:
        print("\n  run these, then re-run this tool:")
        for s in dict.fromkeys(suggestions):  # dedupe, keep order
            print(f"    {s}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
