"""Unit tests for prehensile/l6_discovery.py -- the RealHand L6 CAN discovery core.

Fully offline: no CAN hardware, no realhand SDK calls. The sysfs reader tests
build a fake /sys/class/net-shaped tree under tmp_path; the resolve_hands()
tests inject a fake `_probe` callable, so no `L6`/`realhand` import ever
happens in-process. The one exception -- proving the module itself avoids
importing realhand at load time -- runs in a subprocess (last test), since by
the time this file runs, other tests in the session may already have touched
realhand-importing modules and polluted sys.modules for an in-process check.
"""

import subprocess
import sys
from pathlib import Path

from prehensile.l6_discovery import (
    HAND_VIDPID,
    AdapterInfo,
    ResolvedHand,
    can_netdevs,
    hand_adapters,
    is_up,
    qlen,
    resolve_hands,
    usb_info,
)


# --------------------------------------------------------------------------- #
# Fake-sysfs helpers.
# --------------------------------------------------------------------------- #
def _make_netdev(net_root, ifname, *, can=True, up=True, qlen_val=1000,
                  vidpid=None, usb_port="1-1", product="Fake Adapter"):
    """Create one fake sysfs netdev dir under net_root, wired up like the real thing.

    `vidpid` ("vvvv:pppp") also creates a `device` symlink to a fake USB
    *interface* node whose parent is the USB *device* node carrying
    idVendor/idProduct/product -- exactly one level below where usb_info()
    expects them, mirroring real sysfs (.../usb1/1-1/1-1:1.0). Leaving
    `vidpid` unset (a plain CAN netdev with no USB device backing it) is also
    a valid fixture: usb_info() must degrade gracefully, not raise.
    """
    d = net_root / ifname
    d.mkdir(parents=True)
    (d / "type").write_text("280\n" if can else "1\n")  # 280 == ARPHRD_CAN
    (d / "flags").write_text("0x40081\n" if up else "0x40080\n")  # bit 0x1 == IFF_UP
    (d / "tx_queue_len").write_text(f"{qlen_val}\n")

    if vidpid is not None:
        vendor, _, product_id = vidpid.partition(":")
        usb_dev_dir = net_root / "usb" / usb_port
        usb_dev_dir.mkdir(parents=True, exist_ok=True)
        (usb_dev_dir / "idVendor").write_text(vendor + "\n")
        (usb_dev_dir / "idProduct").write_text(product_id + "\n")
        (usb_dev_dir / "product").write_text(product + "\n")
        usb_iface_dir = usb_dev_dir / f"{usb_port}:1.0"
        usb_iface_dir.mkdir(exist_ok=True)
        (d / "device").symlink_to(usb_iface_dir)
    return d


def _fake_probe(answers):
    """Build a `_probe` fake: answers = {ifname: {sides it answers}}."""
    def probe(ifname, side, timeout_ms=100.0, attempts=2):
        if side in answers.get(ifname, ()):
            return True, f"fake reply from {ifname}/{side}"
        return False, "fake: no reply"
    return probe


# --------------------------------------------------------------------------- #
# can_netdevs -- CAN (type 280) vs non-CAN filtering.
# --------------------------------------------------------------------------- #
def test_can_netdevs_filters_by_type(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", can=True, vidpid=HAND_VIDPID)
    _make_netdev(net_root, "eth0", can=False)
    assert can_netdevs(net_root) == ["hand_l"]


def test_can_netdevs_missing_root_returns_empty():
    assert can_netdevs(net_root=Path("/no/such/dir")) == []


# --------------------------------------------------------------------------- #
# is_up -- IFF_UP bit in the flags file.
# --------------------------------------------------------------------------- #
def test_is_up_true_and_false(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "up_if", up=True)
    _make_netdev(net_root, "down_if", up=False)
    assert is_up("up_if", net_root) is True
    assert is_up("down_if", net_root) is False


def test_is_up_missing_iface_is_false(tmp_path):
    net_root = tmp_path / "class_net"
    net_root.mkdir()
    assert is_up("ghost", net_root) is False


# --------------------------------------------------------------------------- #
# qlen -- tx_queue_len file.
# --------------------------------------------------------------------------- #
def test_qlen_reads_value(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", qlen_val=1000)
    assert qlen("hand_l", net_root) == 1000


def test_qlen_missing_iface_is_zero(tmp_path):
    net_root = tmp_path / "class_net"
    net_root.mkdir()
    assert qlen("ghost", net_root) == 0


# --------------------------------------------------------------------------- #
# usb_info -- the sysfs walk from netdev to USB device node.
# --------------------------------------------------------------------------- #
def test_usb_info_walks_up_to_device_node(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", vidpid=HAND_VIDPID, usb_port="1-1", product="XCAN-USB")
    assert usb_info("hand_l", net_root) == {
        "vidpid": HAND_VIDPID, "port": "1-1", "product": "XCAN-USB",
    }


def test_usb_info_missing_device_returns_unknowns(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "no_usb", vidpid=None)  # no `device` symlink at all
    assert usb_info("no_usb", net_root) == {"vidpid": "?", "port": "?", "product": "?"}


# --------------------------------------------------------------------------- #
# hand_adapters -- vid:pid filtering (up/down is resolve_hands' job, not this).
# --------------------------------------------------------------------------- #
def test_hand_adapters_filters_by_vidpid(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", vidpid=HAND_VIDPID, usb_port="1-1")
    _make_netdev(net_root, "arm_l", vidpid="1d50:606f", usb_port="1-5")  # canable2 -- not a hand
    assert hand_adapters(net_root) == ["hand_l"]


def test_hand_adapters_includes_down_ones(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_down", vidpid=HAND_VIDPID, usb_port="1-2", up=False)
    assert hand_adapters(net_root) == ["hand_down"]


# --------------------------------------------------------------------------- #
# resolve_hands -- the top-level entry point, probing via an injected fake.
# --------------------------------------------------------------------------- #
def test_resolve_hands_two_hands_normal(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", vidpid=HAND_VIDPID, usb_port="1-1")
    _make_netdev(net_root, "hand_r", vidpid=HAND_VIDPID, usb_port="1-4")
    probe = _fake_probe({"hand_l": {"left"}, "hand_r": {"right"}})

    res = resolve_hands(["left", "right"], net_root=net_root, _probe=probe)

    assert res.by_side["left"] == ResolvedHand(
        ifname="hand_l", usb_port="1-1", detail="fake reply from hand_l/left")
    assert res.by_side["right"] == ResolvedHand(
        ifname="hand_r", usb_port="1-4", detail="fake reply from hand_r/right")
    assert res.conflicts == {}
    assert set(res.adapters) == {
        AdapterInfo(ifname="hand_l", usb_port="1-1", up=True),
        AdapterInfo(ifname="hand_r", usb_port="1-4", up=True),
    }


def test_resolve_hands_one_hand_only(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", vidpid=HAND_VIDPID, usb_port="1-1")
    probe = _fake_probe({"hand_l": {"left"}})

    res = resolve_hands(["left", "right"], net_root=net_root, _probe=probe)

    assert res.by_side["left"] == ResolvedHand(
        ifname="hand_l", usb_port="1-1", detail="fake reply from hand_l/left")
    assert res.by_side["right"] is None
    assert res.conflicts == {}


def test_resolve_hands_ambiguous_side_is_conflict(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_a", vidpid=HAND_VIDPID, usb_port="1-1")
    _make_netdev(net_root, "hand_b", vidpid=HAND_VIDPID, usb_port="1-9")
    # Both adapters answer "left" -- e.g. two left hands, or a probe/wiring bug.
    probe = _fake_probe({"hand_a": {"left"}, "hand_b": {"left"}})

    res = resolve_hands(["left"], net_root=net_root, _probe=probe)

    assert res.by_side["left"] is None
    assert set(res.conflicts["left"]) == {"hand_a", "hand_b"}


def test_resolve_hands_down_adapter_excluded_but_listed(tmp_path):
    net_root = tmp_path / "class_net"
    _make_netdev(net_root, "hand_l", vidpid=HAND_VIDPID, usb_port="1-1", up=True)
    _make_netdev(net_root, "hand_down", vidpid=HAND_VIDPID, usb_port="1-2", up=False)

    probed_ifnames = set()

    def probe(ifname, side, timeout_ms=100.0, attempts=2):
        probed_ifnames.add(ifname)
        return (ifname == "hand_l" and side == "left"), "fake"

    res = resolve_hands(["left", "right"], net_root=net_root, _probe=probe)

    assert "hand_down" not in probed_ifnames  # DOWN: never probed ...
    assert {a.ifname for a in res.adapters} == {"hand_l", "hand_down"}  # ... but still listed
    down = next(a for a in res.adapters if a.ifname == "hand_down")
    assert down.up is False
    assert res.by_side["left"].ifname == "hand_l"
    assert res.by_side["right"] is None


def test_resolve_hands_no_adapters_at_all(tmp_path):
    net_root = tmp_path / "class_net"
    net_root.mkdir()

    def boom(*_args, **_kwargs):
        raise AssertionError("must not probe when there are no adapters")

    res = resolve_hands(["left", "right"], net_root=net_root, _probe=boom)

    assert res.by_side == {"left": None, "right": None}
    assert res.adapters == []
    assert res.conflicts == {}


# --------------------------------------------------------------------------- #
# Import hygiene: the module must not pull in realhand at import time.
# --------------------------------------------------------------------------- #
def test_importing_module_does_not_import_realhand():
    """Guards the lazy-import rule (`from realhand import L6` inside probe_side()).

    Run in a subprocess: this pytest session may, via other test modules,
    already have imported something that pulls in realhand, which would make
    an in-process `sys.modules` check pass or fail depending on test order
    rather than on this module's own behaviour.
    """
    code = (
        "import sys\n"
        "import prehensile.l6_discovery\n"
        "assert 'realhand' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "OK"
