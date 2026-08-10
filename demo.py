#!/usr/bin/env python3
"""
L6 dexterous hand - minimal open/close demo (RealHand realbot SDK, SocketCAN).

Hardware assumptions:
  * L6 LEFT hand on a PEAK PCAN-USB adapter.
  * The SocketCAN interface is named `hand_l` -- the CANONICAL name applied by
    tools/bring_up_hand.py (probe the burned-in CAN ID, then
    rename). After a reboot or adapter replug, run bring_up_hand once; it
    also brings the link UP at 1 Mbps. Verify any time with:
    ip -details link show hand_l

What it does:
  Sets a gentle motor speed, then cycles OPEN -> CLOSE -> OPEN, printing the
  hand's angle readback after each pose so you can confirm two-way CAN comms.

Run (from the repo root):
    uv run python demo.py                              # defaults: hand_l, left hand
    uv run python demo.py --speed 60                   # faster finger motion
    uv run python demo.py --interface can0 --side right  # a different rig

SAFETY: the hand physically moves. Clear the area around the fingers first.
Ctrl-C aborts; the hand simply holds its last commanded pose.
"""

import argparse
import time

import can
from realhand import L6
from realhand.exceptions import CANError

# Joint order for every 6-value command and readback (see realhand L6Angle):
JOINTS = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]

# Poses, 0-100 per joint (0 = fully closed/flexed, 100 = fully open/extended).
# CLOSE keeps thumb_abd (index 1) at 100 - the thumb cannot retract past its
# mechanical stop, matching RealHand's own DEFAULT_CLOSED_ANGLES.
OPEN_POSE = [100.0, 100.0, 100.0, 50.0, 50.0, 100.0]
CLOSE_POSE = [0.0, 60.0, 0.0, 0.0, 0.0, 0.0]


def read_and_print(hand, label):
    """Print the hand's current angle readback (or a notice if it doesn't reply)."""
    try:
        data = hand.angle.get_blocking(timeout_ms=500)
    except TimeoutError:
        print(f"    [{label}] no readback (hand did not reply within 500 ms)")
        return
    pretty = "  ".join(f"{n}={v:5.1f}" for n, v in zip(JOINTS, data.angles.to_list()))
    print(f"    [{label}] {pretty}")


def move(hand, pose, label, dwell):
    """Command a pose, hold it, then read back the achieved angles."""
    print(f"-> {label}: {pose}")
    hand.angle.set_angles(pose)
    time.sleep(dwell)
    read_and_print(hand, label)


def main():
    parser = argparse.ArgumentParser(description="RealHand L6 open/close demo")
    parser.add_argument("--interface", default="hand_l",
                        help="SocketCAN interface name (default: hand_l)")
    parser.add_argument("--side", default="left", choices=["left", "right"],
                        help="Hand side; selects CAN ID 0x28 (left) / 0x27 (right)")
    parser.add_argument("--speed", type=float, default=40.0,
                        help="Motor speed 0-100 for all joints (default: 40, gentle)")
    parser.add_argument("--dwell", type=float, default=2.0,
                        help="Seconds to hold each pose (default: 2.0)")
    args = parser.parse_args()

    print(f"L6 demo - side={args.side}  interface={args.interface}  "
          f"speed={args.speed}  dwell={args.dwell}s")
    print("SAFETY: the hand will move. Clear the area. Starting in", end=" ", flush=True)
    for n in (3, 2, 1):
        print(n, end=" ", flush=True)
        time.sleep(1.0)
    print("go!\n")

    try:
        # Constructing L6 opens the CAN bus and auto-starts sensor polling.
        # Using it as a context manager guarantees a clean shutdown on exit.
        with L6(side=args.side, interface_name=args.interface) as hand:
            # Motors will not move at speed 0 - set a nonzero speed once up front.
            hand.speed.set_speeds([args.speed] * 6)
            time.sleep(0.2)

            move(hand, OPEN_POSE, "OPEN", args.dwell)
            move(hand, CLOSE_POSE, "CLOSE", args.dwell)
            move(hand, OPEN_POSE, "OPEN (rest)", args.dwell)

            print("\nDone. Hand left in the open pose.")
    except (CANError, can.CanError, OSError) as exc:
        print(f"\nCAN/bus error: {exc}")
        print("Check the hand is powered and the interface is up:")
        print(f"    ip -details link show {args.interface}")
        return 1
    except KeyboardInterrupt:
        print("\nAborted (the hand holds its last commanded pose).")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
