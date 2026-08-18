#!/usr/bin/env python3
"""Ground-truth L6 endpoint + slot-mapping probe. MOVES THE HARDWARE.

Bypasses the whole glove -> FK -> retarget -> command chain and commands the L6
directly, so you can confirm two things on the *real* hand, independent of any
glove:

  1. DIRECTION: does angle 100 = fully OPEN and 0 = fully CLOSED (the convention
     prehensile.command assumes), or is it reversed?
  2. SLOT MAPPING: which physical finger each of the 6 SDK slots
     [thumb_flex, thumb_abd, index, middle, ring, pinky] actually drives.

Every motion is gated on you pressing Enter, so the hand only moves when you are
watching and ready. Ctrl-C aborts; the hand then holds its last commanded pose.

    uv run python tools/l6_endpoints.py                    # left hand, iface hand_l
    uv run python tools/l6_endpoints.py --side left --speed 30
    uv run python tools/l6_endpoints.py --interface can0

SAFETY: the fingers WILL move. Clear the area and keep hands/cables away from the
hand before you start.
"""

import argparse
import contextlib
import time

# SDK slot order (realhand==0.5.3 L6Angle.to_list). thumb_abd is an ABDUCTION
# (spread) axis, not a curl -- called out separately in the prompts below.
SLOTS = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]
OPEN = [100.0] * 6   # the "fully open" command under command.py's convention
CLOSED = [0.0] * 6   # the "fully closed" command under command.py's convention


def _confirm(prompt: str) -> None:
    """Block until the user presses Enter. Ctrl-C/EOF aborts the whole run."""
    try:
        input(prompt)
    except EOFError:
        raise KeyboardInterrupt


def _send(hand, angles, note: str) -> None:
    """Command one pose after an explicit Enter gate, and echo what was sent."""
    _confirm(f"\n[Enter] to command {note}\n   angles = {angles}  (order {SLOTS}) : ")
    hand.angle.set_angles(list(angles))
    print("   sent. observe the hand, then continue.")


def run(hand) -> None:
    # --- Endpoint test: all channels together --------------------------------
    print("\n=== 1/3  ENDPOINTS (all channels together) ===")
    _send(hand, OPEN, "ALL = 100  -> EXPECT fully OPEN (flat hand, thumb spread)")
    _send(hand, CLOSED, "ALL = 0    -> EXPECT fully CLOSED (fist, thumb tucked)")
    _send(hand, OPEN, "ALL = 100  -> back to open")

    # --- Per-slot isolation: baseline open, drop ONE channel to 0 ------------
    print("\n=== 2/3  PER-SLOT (baseline open=100; one channel -> 0) ===")
    print("For each step: note WHICH physical finger moves and WHICH WAY it goes.")
    for i, name in enumerate(SLOTS):
        one = OPEN.copy()
        one[i] = 0.0
        if name == "thumb_abd":
            expect = "thumb should ADDUCT (tuck toward palm) -- a sideways spread, not a curl"
        elif name == "thumb_flex":
            expect = "thumb should FLEX (curl toward palm)"
        else:
            expect = f"the {name} finger should CLOSE (curl); the others stay open"
        _send(hand, one, f"{name} -> 0, rest 100  -> EXPECT: {expect}")
        _send(hand, OPEN, "ALL = 100  -> reset to open")

    # --- Leave the hand open --------------------------------------------------
    print("\n=== 3/3  DONE -- leaving the hand OPEN ===")
    _send(hand, OPEN, "ALL = 100  -> final open pose")


def main() -> int:
    ap = argparse.ArgumentParser(description="Directly command the L6 to probe endpoints + slot mapping (MOVES HARDWARE)")
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--interface", default="hand_l",
                    help="L6 SocketCAN interface (default: hand_l, the left hand's "
                         "canonical iface -- applied by tools/bring_up_hand.py)")
    ap.add_argument("--speed", type=float, default=30.0,
                    help="motor speed 0-100 (default 30; motors won't move at 0)")
    args = ap.parse_args()

    # realhand/CAN imported here so --help works without the SDK/bus present.
    import can
    from realhand import L6
    from realhand.exceptions import CANError

    print(__doc__.split("\n\n")[0])
    print(f"\nside={args.side}  interface={args.interface}  speed={args.speed}")
    print("\n*** SAFETY: the L6 fingers WILL move. Clear the area now. ***")
    try:
        _confirm("Type nothing -- just press [Enter] to open the hand and begin (Ctrl-C aborts): ")
        print("Starting in", end=" ", flush=True)
        for n in (3, 2, 1):
            print(n, end=" ", flush=True)
            time.sleep(1.0)
        print("go!")
    except KeyboardInterrupt:
        print("\naborted before any motion.")
        return 0

    try:
        with L6(side=args.side, interface_name=args.interface) as hand:
            hand.speed.set_speeds([args.speed] * 6)  # motors won't move at speed 0
            time.sleep(0.2)
            with contextlib.suppress(KeyboardInterrupt):
                run(hand)
                print("\nfinished cleanly.")
                return 0
            print("\naborted -- hand holds its last commanded pose.")
    except (CANError, can.CanError, OSError) as exc:
        print(f"\nCAN/L6 error: {exc}")
        print(f"Check the L6 is powered and {args.interface} is up: "
              f"ip -details link show {args.interface}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
