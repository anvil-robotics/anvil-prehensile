#!/usr/bin/env python3
"""M4: MuJoCo viewer -- watch the L6 (and your hand) driven live by the glove, in sim.

Pure simulation: does NOT touch the real L6. Reads the selected glove
(``--glove wuji|udcap``, default ``udcap``) for the selected hand (``--hand
left|right``, default ``left``), retargets each frame, and renders the L6 URDF
tracking your hand. By default it also overlays the human hand keypoints as red
spheres (transformed into the robot wrist frame; the alignment is approximate --
pass --no-keypoints to hide them).

    uv run python -m prehensile.viz                    # sim hand + keypoint overlay (UDCap, left)
    uv run python -m prehensile.viz --glove wuji        # Wuji glove instead
    uv run python -m prehensile.viz --hand right        # right hand/URDF
    uv run python -m prehensile.viz --no-keypoints      # sim hand only
    uv run python -m prehensile.viz --map retarget      # preview the optimizer instead

udcap needs HandDriver streaming the Quater content to 127.0.0.1:<port>. wuji
needs the Wuji glove connected and wuji_sdk installed. Opens a MuJoCo window
(needs a display). Close the window or Ctrl-C to stop.

By default (``--map curl``) the sim previews the direct per-finger curl map
(``prehensile.curlmap``, the same path ``teleop.py`` drives): each L6 angle
(0-100) is mapped back across its driver joint's URDF limit range and the mimic
(*_dip) joints are propagated, so you see exactly what the curl command does to
the hand. Tuning is always read from configs/curl_tuning.yml (see
``prehensile.tuning``); there are no CLI tuning flags.

With ``--map retarget`` the sim is instead driven by the raw retargeted qpos
(radians) -- the most faithful view of what retargeting produces, before the
0-100 quantization the real hand receives.
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dex_retargeting.constants import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT

from prehensile.command import L6_DRIVER_JOINTS, L6_OPEN, L6_SDK_ORDER
from prehensile.curlmap import CurlMapper
from prehensile.profiles import GLOVES, HANDS
from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning
from prehensile.retarget import L6Retargeter, estimate_frame_from_hand_points

ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = ROOT / "assets" / "realhand_description"


def build_qpos_map(model, pin_joint_names):
    """[(mujoco qpos addr, pinocchio qpos index)] for joints present in both models.

    Mapped BY JOINT NAME so it is robust to mujoco-vs-pinocchio ordering. The
    retargeted qpos already includes the mimic (*_dip) joint values, so setting
    every matched joint reproduces the full pose without needing the constraint solver.
    """
    pin_idx = {n: i for i, n in enumerate(pin_joint_names)}
    pairs = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in pin_idx:
            pairs.append((int(model.jnt_qposadr[jid]), pin_idx[name]))
    return pairs


def apply_frame(model, data, retargeter, qmap, kp) -> bool:
    """Retarget one keypoint frame into the sim state, headless-safe.

    Runs ``retargeter.retarget(kp)``; if it yields a pose, writes each mapped
    joint into ``data.qpos`` and calls ``mujoco.mj_forward`` (no viewer needed).
    Returns True if a pose was applied, False if retargeting returned None. This
    is the whole per-frame model update, factored out of the viewer loop so it is
    unit-testable without opening a window.
    """
    qpos = retargeter.retarget(kp)
    if qpos is None:
        return False
    for addr, idx in qmap:
        data.qpos[addr] = qpos[idx]
    mujoco.mj_forward(model, data)
    return True


def draw_keypoints(viewer, kp, operator2mano):
    """Overlay the human keypoints as spheres, in the robot wrist frame (approximate)."""
    kp = np.asarray(kp, dtype=np.float64)
    pts = kp @ estimate_frame_from_hand_points(kp) @ operator2mano
    scn = viewer.user_scn
    scn.ngeom = 0
    size = np.array([0.004, 0.0, 0.0])
    mat = np.eye(3).flatten()
    rgba = np.array([0.9, 0.2, 0.2, 1.0], dtype=np.float32)
    for p in pts:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size, p, mat, rgba
        )
        scn.ngeom += 1


# --------------------------------------------------------------------------- #
# --map curl support: pose the URDF from the 6 L6 angles (0-100) the CurlMapper
# produces, rather than the retargeter's qpos. There is no optimizer qpos on this
# path, so command.py's angle convention is inverted analytically -- each driver
# joint's 0-100 maps linearly across its URDF limit range (100=open=lower,
# 0=closed=upper) -- and the URDF mimic (*_dip) joints are propagated, clamped to
# their own limits (a multiplier can drive a dip past its declared range).
# --------------------------------------------------------------------------- #
def _parse_urdf_joints(urdf_path):
    """{joint_name: (lower, upper, mimic)} from the URDF; ``mimic`` is
    ``(parent_joint, multiplier, offset)`` or ``None``."""
    out = {}
    for j in ET.parse(str(urdf_path)).getroot().findall("joint"):
        lim = j.find("limit")
        lo = float(lim.get("lower", 0.0)) if lim is not None else 0.0
        up = float(lim.get("upper", 0.0)) if lim is not None else 0.0
        m = j.find("mimic")
        mimic = ((m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
                 if m is not None else None)
        out[j.get("name")] = (lo, up, mimic)
    return out


def build_curl_plan(model, urdf_path):
    """Precompute the qpos-write plan for the --map curl path.

    Returns ``(drivers, dips)``:
      ``drivers`` -- per ``L6_SDK_ORDER`` slot: ``(qpos_addr, lower, upper)`` of its
        driver joint.
      ``dips`` -- per mimic joint whose parent is a driver:
        ``(qpos_addr, parent_slot, multiplier, offset, lower, upper)``.
    A ``qpos_addr`` of ``None`` (joint absent from the model) is skipped by
    ``apply_curl_frame``.
    """
    urdf = _parse_urdf_joints(urdf_path)

    def addr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[jid]) if jid >= 0 else None

    drivers, driver_slot = [], {}
    for slot_i, slot in enumerate(L6_SDK_ORDER):
        jname = L6_DRIVER_JOINTS[slot]
        lo, up, _ = urdf.get(jname, (0.0, 0.0, None))
        drivers.append((addr(jname), lo, up))
        driver_slot[jname] = slot_i

    dips = []
    for name, (lo, up, mimic) in urdf.items():
        if mimic is not None and mimic[0] in driver_slot:
            parent, mult, off = mimic
            dips.append((addr(name), driver_slot[parent], mult, off, lo, up))
    return drivers, dips


def apply_curl_frame(model, data, curl_plan, mapper, kp) -> bool:
    """Map one keypoint frame through the CurlMapper into the sim state, headless-safe.

    Mirrors ``apply_frame``'s contract (True if a pose was applied, False if the
    mapper returned None), but poses the URDF from the 6 L6 angles rather than the
    retargeter's qpos.
    """
    angles = mapper(kp)
    if angles is None:
        return False
    drivers, dips = curl_plan
    driver_rad = []
    for (a, lo, up), ang in zip(drivers, angles):
        rad = lo + (1.0 - ang / L6_OPEN) * (up - lo)  # 100=open=lower, 0=closed=upper
        driver_rad.append(rad)
        if a is not None:
            data.qpos[a] = rad
    for a, slot, mult, off, lo, up in dips:
        if a is not None:
            data.qpos[a] = float(np.clip(mult * driver_rad[slot] + off, lo, up))
    mujoco.mj_forward(model, data)
    return True


def main():
    ap = argparse.ArgumentParser(description="MuJoCo viewer for glove->L6 retargeting")
    ap.add_argument("--glove", choices=sorted(GLOVES), default="udcap",
                    help="glove source to read from (default: udcap)")
    ap.add_argument("--hand", choices=sorted(HANDS), default="left",
                    help="which hand/URDF to render (default: left)")
    ap.add_argument("--port", type=int, default=5555, help="UDP port HandDriver streams to (udcap only)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--no-keypoints", action="store_true", help="hide the human keypoint overlay")
    ap.add_argument("--map", choices=["retarget", "curl"], default="curl",
                    help="angle-mapping strategy to preview: the direct per-finger curl map "
                         "(default, see curlmap.py) or the dex_retargeting vector optimizer")
    args = ap.parse_args()

    glove = GLOVES[args.glove]
    hand = HANDS[args.hand]
    URDF = ROOT / "assets" / "realhand_description" / "l6" / hand.side / f"realhand_l6_{hand.side}.urdf"
    CONFIG = ROOT / "configs" / hand.config
    operator2mano = OPERATOR2MANO_LEFT if hand.side == "left" else OPERATOR2MANO_RIGHT

    print("Loading MuJoCo model + retargeter...")
    model = mujoco.MjSpec.from_file(str(URDF)).compile()
    data = mujoco.MjData(model)
    retargeter = L6Retargeter(CONFIG, URDF_DIR, side=hand.side)
    qmap = build_qpos_map(model, retargeter.joint_names)
    print(f"mapped {len(qmap)}/{len(retargeter.joint_names)} joints into the sim")

    mapper = curl_plan = None
    if args.map == "curl":
        # abd_invert comes solely from the glove profile now (see teleop.py /
        # profiles.py); tuning is always configs/curl_tuning.yml, and a
        # missing thumb_flex.couple_low is now a hard error (see resolve_tuning).
        tuning = resolve_tuning(DEFAULT_TUNING_PATH, side=hand.side, require_couple_low=True)
        mapper = CurlMapper(side=hand.side, abd_invert=glove.abd_invert, tuning=tuning)
        curl_plan = build_curl_plan(model, URDF)

    print(f"Reading the {glove.name} glove "
          f"({'UDP :' + str(args.port) if args.glove == 'udcap' else 'SDK'}).")
    budget = 1.0 / args.fps
    last_kp = None
    try:
        src_cm = glove.build_source(hand.side, args.port)
    except OSError as exc:
        print(f"Cannot bind UDP :{args.port}: {exc}")
        return 1
    try:
        with src_cm as src:
            print("Opening viewer (close window or Ctrl-C to stop)...")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.azimuth = 180
                viewer.cam.elevation = -30
                viewer.cam.distance = 0.4
                viewer.cam.lookat[:] = [0.0, 0.0, 0.08]
                while viewer.is_running():
                    t0 = time.monotonic()
                    kp = src.poll()
                    if kp is None:
                        kp = last_kp
                    else:
                        last_kp = kp
                    if kp is not None:
                        if mapper is not None:
                            apply_curl_frame(model, data, curl_plan, mapper, kp)
                        else:
                            apply_frame(model, data, retargeter, qmap, kp)
                        if not args.no_keypoints:
                            draw_keypoints(viewer, kp, operator2mano)
                    viewer.sync()
                    dt = time.monotonic() - t0
                    if dt < budget:
                        time.sleep(budget - dt)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
