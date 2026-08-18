"""Smoke test for the full udcap -> fk -> retarget chain: a synthetic
length-prefixed datagram fixture is generated in-code, replayed through
``iter_datagrams``/``parse_quat_datagram``, and the decoded keypoints are fed
into ``L6Retargeter`` to confirm the whole pipeline runs end-to-end and stays
within the URDF's joint limits.
"""

import math
import struct
from pathlib import Path

import numpy as np

from prehensile.udcap import iter_datagrams, parse_quat_datagram
from prehensile import fk
from prehensile._vendor.udex_protobuf import handdriver_teleop_pb2 as pb2  # for building a synthetic datagram
from prehensile.retarget import L6Retargeter

_REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = _REPO_ROOT / "assets" / "realhand_description"
CONFIG = _REPO_ROOT / "configs" / "real_hand_left.yml"

_LEN = struct.Struct("<I")  # uint32 LE frame-length prefix


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


# --------------------------------------------------------------------------- #
# synthetic fixture end-to-end + L6 retarget smoke (full Branch-B chain).
# --------------------------------------------------------------------------- #
def test_synthetic_fixture_end_to_end_and_retarget_smoke():
    # Generate a clearly-synthetic fixture in-code, in the same length-prefixed
    # format iter_datagrams expects.
    fixtures = Path(__file__).parent / "fixtures"
    fixtures.mkdir(exist_ok=True)
    fixture = fixtures / "synth_quat_left.bin"
    ident = fk.identity_quats(fk.FK_MODE)
    with fixture.open("wb") as f:
        for i in range(30):
            q = ident.copy()
            q[0] = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(0.5 * i))
            payload = _make_quat_proto(q, side="left", frame_index=i)
            f.write(_LEN.pack(len(payload)))
            f.write(payload)

    # iter_datagrams -> parse_quat_datagram over every frame.
    kps = [parse_quat_datagram(d) for d in iter_datagrams(fixture)]
    assert len(kps) == 30
    assert all(kp is not None and kp.shape == (21, 3) and kp.dtype == np.float32
               for kp in kps)

    # Retarget smoke: build L6Retargeter exactly as offline_check.py does.
    rt = L6Retargeter(CONFIG, URDF_DIR, side="left")
    qpos = rt.retarget(kps[15])
    assert qpos is not None
    assert np.all(np.isfinite(qpos))

    # qpos is full-DOF and includes the mimic *_dip joints (coupled, not commanded),
    # whose values can exceed their naive per-joint URDF limit by design. The
    # optimizer only bounds the actuated/driver joints (idx_pin2target) -- those
    # (which are exactly what command.py sends) must lie within their URDF limits,
    # within the optimizer's own 1e-3 epsilon.
    opt = rt.retargeting.optimizer
    actuated = np.asarray(qpos)[opt.idx_pin2target]
    lower, upper = rt.retargeting.joint_limits[:, 0], rt.retargeting.joint_limits[:, 1]
    assert actuated.shape == lower.shape == upper.shape
    assert np.all(actuated >= lower - 1e-3)
    assert np.all(actuated <= upper + 1e-3)
