"""The teleop tests that need a REAL L6Retargeter (the `research` extra).

Split out of tests/test_teleop.py deliberately, and the split is load-bearing
rather than cosmetic: `from prehensile.retarget import L6Retargeter` drags in
dex_retargeting at COLLECTION time, and pytest aborts the whole session on a
collection error rather than failing the one file. With these three tests still
sitting in tests/test_teleop.py, that single import made the entire module
uncollectable on a bare `pip install prehensile`, so CI's lean job (the
pull-request gate) had to --ignore the file and all of its ~20 hardware-free
tests went ungated. They now run leanly; this file is the one CI ignores.

Not solved with `importorskip`/`skipif` on purpose: .github/scripts/
check_pytest_report.py treats ANY skip as a failure, by design, so a
skip-based fix would turn the lean job red instead of green.

The small fakes below are copies of tests/test_teleop.py's (see there for the
fuller commentary on why loop() is driven this way). Copied, not imported:
cross-importing one test module from another is exactly the coupling this split
exists to remove -- each file must collect on its own.
"""

import socket
import time
from pathlib import Path

import pytest
from prehensile import fk
from prehensile.command import qpos_index_map
from prehensile.udcap import UDCapSource
from prehensile._vendor.udex_protobuf import handdriver_teleop_pb2 as pb2  # for building a synthetic datagram
from prehensile import teleop
from prehensile.retarget import L6Retargeter

_REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = _REPO_ROOT / "configs" / "real_hand_left.yml"
URDF_DIR = _REPO_ROOT / "assets" / "realhand_description"


def _make_quat_proto(quats_15x4, side="left") -> bytes:
    """A serialized TeleopDataQuat with the given side's 15 joint quats set."""
    msg = pb2.TeleopDataQuat()
    hand = msg.LeftHand if side == "left" else msg.RightHand
    hand.serialNumber = "UDXST4810L" if side == "left" else "UDXST4810R"
    for q in quats_15x4:
        j = hand.joints.add()
        j.x, j.y, j.z, j.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return msg.SerializeToString()


class _StopScript(Exception):
    """Raised by a fake source once its scripted frames are exhausted, to
    unwind out of loop()'s ``while True`` from inside a test."""


class _ScriptedButtonSource:
    """Fake glove source: replays a scripted ``(kp, bButton)`` sequence, one
    pair per ``poll()`` call, then raises ``_StopScript``."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.bButton = False

    def poll(self):
        if self._i >= len(self._script):
            raise _StopScript
        kp, pressed = self._script[self._i]
        self._i += 1
        self.bButton = pressed
        return kp


class _RecordingSink:
    """``sink`` fake: records every list of angles it is called with."""

    def __init__(self):
        self.calls: list[list[float]] = []

    def __call__(self, angles) -> None:
        self.calls.append(list(angles))


def _identity_kp():
    """The neutral-pose (21,3) keypoint frame the retarget tests use."""
    return fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)


def test_frame_to_angles_on_synthetic_frame():
    """keypoints -> 6 L6 angles in [0,100] (the whole command step, no socket)."""
    retargeter = L6Retargeter(CONFIG, URDF_DIR, side="left")
    index_map = qpos_index_map(retargeter.joint_names)
    kp = fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)
    angles = teleop.frame_to_angles(kp, retargeter, index_map)
    assert angles is not None
    assert len(angles) == 6
    assert all(0.0 <= a <= 100.0 for a in angles)


def test_dry_run_receive_to_angles_over_loopback():
    """The full dry-run path headless: UDCapSource.poll() -> frame_to_angles -> 6 angles."""
    retargeter = L6Retargeter(CONFIG, URDF_DIR, side="left")
    index_map = qpos_index_map(retargeter.joint_names)
    data = _make_quat_proto(fk.identity_quats(fk.FK_MODE), side="left")
    with UDCapSource(port=0, side="left") as src:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(data, ("127.0.0.1", src.port))
        finally:
            sender.close()
        time.sleep(0.05)  # let the kernel enqueue the datagram on loopback
        kp = src.poll()
    assert kp is not None
    angles = teleop.frame_to_angles(kp, retargeter, index_map)
    assert angles is not None
    assert len(angles) == 6
    assert all(0.0 <= a <= 100.0 for a in angles)


def test_loop_retarget_path_ignores_button_and_has_no_marker(capsys):
    """mapper=None (--map retarget): a bButton press must not crash (no
    mapper.locked to touch) and the readout must never carry a marker."""
    retargeter = L6Retargeter(CONFIG, URDF_DIR, side="left")
    index_map = qpos_index_map(retargeter.joint_names)
    source = _ScriptedButtonSource([(_identity_kp(), True)])
    sink = _RecordingSink()
    with pytest.raises(_StopScript):
        teleop.loop(source, retargeter, index_map, sink, fps=1000.0, mapper=None)
    assert len(sink.calls) == 1
    assert "[PARKED" not in capsys.readouterr().out
