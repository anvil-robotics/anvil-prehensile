"""Glove x hand profiles: the small config seam letting teleop/viz pick a glove
source and a hand at runtime.

Axes:
  GLOVE ("wuji"|"udcap") -> which Source impl + per-glove invert_flex (open/close
    sense for the retarget path, measured on real hardware) + per-glove abd_invert
    (thumb_abd sense for the --map curl path; per-glove and hardware-observed --
    UDCap inverts under the palm-plane abduction metric, Wuji does not).
  HAND  ("left"|"right") -> which retarget config (URDF) + L6 side + default CAN iface.

Extension point for future per-finger gains (extend GloveProfile + command.py) and
user-skeleton matching (a pre-seam transform on the (21,3) keypoints)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GloveProfile:
    name: str
    invert_flex: bool
    build_source: Callable[[str, int], object]  # (hand_side, port) -> Source
    # thumb_abd sense for the --map curl path (CurlMapper.abd_invert). Retarget
    # path is unaffected. Per-glove, hardware-observed under the palm-plane
    # abduction metric (curlmap._thumb_abd_angle_deg): UDCap True 2026-07-24,
    # Wuji False 2026-08-10 (checked on the right hand).
    abd_invert: bool = False


@dataclass(frozen=True)
class HandProfile:
    side: str        # "left" | "right"
    # yaml filename under this repo's top-level configs/ (real_hand_left.yml /
    # real_hand_right.yml), resolved by the CLI entry points (teleop.py/
    # viz.py).
    config: str
    interface: str   # default L6 SocketCAN interface


# Constraint: module-level imports here must stay stdlib-light --
# hand_teleop_node (lean /prehensile_venv, no wuji-sdk/torch) imports this
# module for GLOVES, so glove-SDK imports stay function-local in build_source.
def _build_wuji(hand_side: str, port: int):
    # Lazy on purpose: the hand_teleop node path never calls _build_wuji (it
    # only reads GLOVES for udcap), and its lean /prehensile_venv must never
    # be forced to pull in the Wuji glove SDK just because this function
    # exists -- so this import stays lazy/function-local.
    from prehensile.wuji import WujiSource
    return WujiSource(side=hand_side)


def _build_udcap(hand_side: str, port: int):
    from prehensile.udcap import UDCapSource
    return UDCapSource(port=port, side=hand_side)


GLOVES = {
    "wuji":  GloveProfile("wuji",  invert_flex=False, build_source=_build_wuji,  abd_invert=False),
    "udcap": GloveProfile("udcap", invert_flex=True,  build_source=_build_udcap, abd_invert=True),
}

HANDS = {
    # Interface names are the CANONICAL ones applied by tools/bring_up_hand.py
    # (probe the burned-in CAN ID -- left=0x28, right=0x27 -- then rename), so
    # they hold no matter which USB port an adapter is plugged into. After a
    # reboot or adapter replug, run bring_up_hand once to re-apply them.
    "left":  HandProfile(side="left",  config="real_hand_left.yml",  interface="hand_l"),
    "right": HandProfile(side="right", config="real_hand_right.yml", interface="hand_r"),
}
