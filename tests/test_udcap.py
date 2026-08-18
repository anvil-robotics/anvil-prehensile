"""TDD validators for prehensile/udcap.py (Branch B UDP glove source).

SYNTHETIC-ONLY: no glove hardware (the user is remote). These tests build both
JSON and protobuf-quat datagrams in code and exercise the drain-latest socket
over a loopback port. Real recordings arrive later.

The five behaviours proven here (mirroring the Task-6 spec):
  1. protobuf-quat round-trip: identity quats -> rest pose,
  2. JSON round-trip: identity quats -> rest pose,
  3. right-glove-only datagrams -> None (both serializations),
  4. garbage / foreign datagrams -> None, never raising,
  5. drain-latest over a real loopback socket returns the NEWEST valid frame.

(4b, added for B4: rejected datagrams bump a module-level counter and fire a
one-shot warning -- not part of the original Task-6 spec, just visibility for
what was already tested as a silent None above.)

(The former #6, a synthetic-fixture-to-retarget smoke test, moved to
``tests/test_udcap_retarget_chain.py`` in ``prehensile_v1.1`` when the
pipeline core and the research lab split into separate packages -- it depends
on the research-only retarget module, which no longer lives alongside this
one.)
"""

import json
import math
import socket
import time

import numpy as np

from prehensile import fk, udcap
from prehensile.udcap import UDCapSource, parse_quat_datagram
from prehensile._vendor.udex_protobuf import handdriver_teleop_pb2 as pb2  # for building synthetic datagrams

# Rest-pose keypoints: fk's identity sentinel is the neutral/open-hand input, so a
# datagram carrying it must decode straight back to the rest skeleton.
REST_KP = fk.keypoints_from_quats(fk.identity_quats(fk.FK_MODE), fk.FK_MODE)


# --------------------------------------------------------------------------- #
# Synthetic datagram builders.
# --------------------------------------------------------------------------- #
def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    """XYZW quaternion for a rotation of ``angle_rad`` about ``axis``."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s = math.sin(angle_rad / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle_rad / 2.0)])


def _make_quat_proto(quats_15x4, side="left", frame_index=0) -> bytes:
    """A serialized ``TeleopDataQuat`` with the given side's 15 joint quats set."""
    msg = pb2.TeleopDataQuat()
    msg.FrameIndex = frame_index
    hand = msg.LeftHand if side == "left" else msg.RightHand
    hand.serialNumber = "UDXST4810L" if side == "left" else "UDXST4810R"
    for q in quats_15x4:
        j = hand.joints.add()
        j.x, j.y, j.z, j.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return msg.SerializeToString()


def _make_quat_json(quats_15x4, device="UDXST4810L", frame_index=0) -> bytes:
    """A per-glove JSON datagram: 15 finger quats + 1 IMU quat = 16 Bones."""
    bones = [[float(x) for x in q] for q in quats_15x4]
    bones.append([0.0, 0.0, 0.0, 1.0])  # IMU quat at index 15
    obj = {
        "DeviceName": device,
        "FrameIndex": frame_index,
        "CalibrationStatus": 3,
        "Battery": 88,
        "aButton": False,
        "Bones": bones,
    }
    return json.dumps(obj).encode("utf-8")


# --------------------------------------------------------------------------- #
# 1. protobuf round-trip -> rest pose.
# --------------------------------------------------------------------------- #
def test_protobuf_roundtrip_reproduces_rest_pose():
    data = _make_quat_proto(fk.identity_quats(fk.FK_MODE), side="left")
    kp = parse_quat_datagram(data)
    assert kp is not None
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_array_equal(kp[0], np.zeros(3, dtype=np.float32))  # wrist ~0
    np.testing.assert_allclose(kp, REST_KP, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. JSON round-trip -> rest pose.
# --------------------------------------------------------------------------- #
def test_json_roundtrip_reproduces_rest_pose():
    data = _make_quat_json(fk.identity_quats(fk.FK_MODE), device="UDXST4810L")
    kp = parse_quat_datagram(data)
    assert kp is not None
    assert kp.shape == (21, 3)
    assert kp.dtype == np.float32
    np.testing.assert_array_equal(kp[0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(kp, REST_KP, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. right-glove-only -> None (both serializations).
# --------------------------------------------------------------------------- #
def test_right_glove_only_returns_none():
    ident = fk.identity_quats(fk.FK_MODE)
    # protobuf with only RightHand set (no LeftHand field present).
    assert parse_quat_datagram(_make_quat_proto(ident, side="right")) is None
    # JSON whose DeviceName ends with 'R'.
    assert parse_quat_datagram(_make_quat_json(ident, device="UDXST4810R")) is None


# --------------------------------------------------------------------------- #
# 4. garbage -> None, never raises.
# --------------------------------------------------------------------------- #
def test_garbage_returns_none_never_raises():
    rng = np.random.default_rng(0)
    random_bytes = bytes(int(b) for b in rng.integers(0, 256, size=96, dtype=np.uint8))
    assert parse_quat_datagram(random_bytes) is None
    assert parse_quat_datagram(b"") is None
    # Valid JSON object but missing the Bones list.
    assert parse_quat_datagram(json.dumps({"DeviceName": "UDXST4810L"}).encode()) is None
    # JSON that isn't even an object.
    assert parse_quat_datagram(b"[1, 2, 3]") is None
    # A TeleopDataAngle (23 float joints) fed to the quat parser must NOT
    # mis-decode into a bogus (21,3); proto3 leniency is caught by validation.
    ang = pb2.TeleopDataAngle()
    ang.LeftHand.serialNumber = "UDXST4810L"
    for i in range(23):
        ang.LeftHand.joints.append(0.1 * i)
    assert parse_quat_datagram(ang.SerializeToString()) is None
    # Non-JSON, non-protobuf bytes.
    assert parse_quat_datagram(b"\xff\x00\x01 not a datagram") is None


# --------------------------------------------------------------------------- #
# 4b. rejected datagrams are counted, and warned about exactly once (B4:
# visibility for what was previously a silent None). Both tests read/reset
# module-level state directly (`udcap._reject_count`, `udcap._warned`) since
# that state is process-global and must not leak across tests / depend on
# execution order (e.g. test 4 above already drives some rejects).
# --------------------------------------------------------------------------- #
def test_garbage_increments_reject_counter():
    before = udcap._reject_count
    assert parse_quat_datagram(b"\xff\x00\x01 not a datagram") is None
    assert udcap._reject_count == before + 1
    # Empty data is a separate, earlier guard in _parse_datagram (no datagram
    # at all, not a malformed one) -- must NOT count as a reject.
    before2 = udcap._reject_count
    assert parse_quat_datagram(b"") is None
    assert udcap._reject_count == before2


def test_first_reject_emits_one_shot_warning(capsys):
    udcap._warned.discard("reject")  # force "first time" regardless of test order
    capsys.readouterr()  # drain anything buffered from earlier tests

    assert parse_quat_datagram(b"\xff\x00\x01 not a datagram") is None
    first_err = capsys.readouterr().err
    assert "udcap:" in first_err
    assert "rejected" in first_err

    # One-shot: a second reject must not print again.
    assert parse_quat_datagram(b"\xff\x00\x01 not a datagram") is None
    second_err = capsys.readouterr().err
    assert second_err == ""


# --------------------------------------------------------------------------- #
# 5. drain-latest over a real loopback socket.
# --------------------------------------------------------------------------- #
def test_drain_latest_returns_newest_valid_frame():
    ident = fk.identity_quats(fk.FK_MODE)
    payloads, expected = [], []
    for i in range(5):
        q = ident.copy()
        # Vary the index-proximal joint so each frame's keypoints differ.
        q[0] = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(10.0 * (i + 1)))
        payloads.append(_make_quat_proto(q, side="left", frame_index=i))
        expected.append(fk.keypoints_from_quats(q, fk.FK_MODE))

    with UDCapSource(port=0, side="left") as src:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for p in payloads:
                sender.sendto(p, ("127.0.0.1", src.port))
        finally:
            sender.close()
        time.sleep(0.05)  # let the kernel enqueue all 5 on the loopback path

        kp = src.poll()
        assert kp is not None
        # poll() drains ALL queued and returns the NEWEST valid frame (the 5th).
        np.testing.assert_allclose(kp, expected[4], atol=1e-5)
        assert not np.allclose(kp, expected[0], atol=1e-4)  # genuinely distinct
        # Immediate re-poll with nothing queued -> None.
        assert src.poll() is None
