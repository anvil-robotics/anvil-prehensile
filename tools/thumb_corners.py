#!/usr/bin/env python3
"""Drive the L6 THUMB to the four corners of its (flex, abd) space. MOVES HARDWARE.

Commands the L6 directly (no glove / FK / curl-map involved), holding the four
fingers OPEN and stepping the thumb through every combination of:

    thumb_flex : 100 = EXTENDED (straight)   0 = FLEXED (curled to palm)
    thumb_abd  : 100 = SPREAD (away from palm) 0 = TUCKED (adducted to index)

so you can see each extreme and confirm the thumb's two axes + their directions
on the real hand. Because it sends RAW L6 values, thumb_abd here is the
HARDWARE-NATIVE sense (100 = spread), independent of any glove/abd_invert -- handy
for sanity-checking the abduction direction.

The four corners (flex, abd):
    [100,100] extended+spread   [100,0] extended+tucked
    [  0,  0] flexed+tucked      [  0,100] flexed+spread

By default every move is gated on Enter (moves only when you're watching);
--auto cycles through them with a fixed dwell instead. Ctrl-C aborts; the hand
then holds its last commanded pose.

    uv run python tools/thumb_corners.py                     # right hand, iface hand_r, Enter-gated
    uv run python tools/thumb_corners.py --auto --dwell 2    # auto-cycle, 2s per corner
    uv run python tools/thumb_corners.py --side left --interface hand_l
    uv run python tools/thumb_corners.py --speed 20

SAFETY: the thumb WILL move (fingers stay open). Clear the area and keep
hands/cables away before you start.
"""

import argparse
import contextlib
import time

# SDK slot order (realhand==0.5.3 L6Angle.to_list); 100 = open, 0 = closed.
SLOTS = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]
FINGERS_OPEN = 100.0  # hold the four fingers open, out of the thumb's way

# (thumb_flex, thumb_abd, human-readable expectation). Walked in a loop around
# the quadrants, then returned to extended+spread as a rest pose.
CORNERS = [
    (100.0, 100.0, "EXTENDED + SPREAD  (flex=100, abd=100)  -> thumb straight, swung away from the palm"),
    (100.0,   0.0, "EXTENDED + TUCKED  (flex=100, abd=0)    -> thumb straight, pulled in beside the index"),
    (0.0,     0.0, "FLEXED + TUCKED    (flex=0,   abd=0)    -> thumb curled to the palm, pulled in"),
    (0.0,   100.0, "FLEXED + SPREAD    (flex=0,   abd=100)  -> thumb curled, but swung away from the palm"),
]
REST = (100.0, 100.0, "EXTENDED + SPREAD  (rest pose)")


def _pose(flex: float, abd: float) -> list[float]:
    """The 6-value L6 command for a thumb corner, fingers held open."""
    return [flex, abd, FINGERS_OPEN, FINGERS_OPEN, FINGERS_OPEN, FINGERS_OPEN]


def _confirm(prompt: str) -> None:
    """Block until the user presses Enter. Ctrl-C/EOF aborts the whole run."""
    try:
        input(prompt)
    except EOFError:
        raise KeyboardInterrupt


def run(hand, auto: bool, dwell: float) -> None:
    steps = CORNERS + [REST]
    for i, (flex, abd, desc) in enumerate(steps, 1):
        angles = _pose(flex, abd)
        label = f"{i}/{len(steps)}  {desc}"
        if auto:
            print(f"\n-> {label}\n   angles = {angles}  (order {SLOTS})")
            hand.angle.set_angles(angles)
            time.sleep(dwell)
        else:
            _confirm(f"\n[Enter] to command {label}\n   angles = {angles}  (order {SLOTS}) : ")
            hand.angle.set_angles(angles)
            print("   sent. observe the thumb, then continue.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drive the L6 thumb to the 4 (flex, abd) corners (MOVES HARDWARE)")
    ap.add_argument("--side", choices=("left", "right"), default="right",
                    help="L6 side (default: right; selects CAN ID 0x27)")
    ap.add_argument("--interface", default="hand_r",
                    help="L6 SocketCAN interface (default: hand_r, the right hand's iface)")
    ap.add_argument("--speed", type=float, default=25.0,
                    help="motor speed 0-100 (default 25; motors won't move at 0)")
    ap.add_argument("--auto", action="store_true",
                    help="cycle through the corners automatically instead of Enter-gating each")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="seconds to hold each corner in --auto mode (default 2.0)")
    args = ap.parse_args()

    # realhand/CAN imported here so --help works without the SDK/bus present.
    import can
    from realhand import L6
    from realhand.exceptions import CANError

    print(__doc__.split("\n\n")[0])
    print(f"\nside={args.side}  interface={args.interface}  speed={args.speed}  "
          f"{'auto, dwell=' + str(args.dwell) + 's' if args.auto else 'Enter-gated'}")
    print("\n*** SAFETY: the L6 thumb WILL move. Clear the area now. ***")
    try:
        _confirm("Press [Enter] to open the hand and begin (Ctrl-C aborts): ")
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
                run(hand, args.auto, args.dwell)
                print("\nfinished cleanly (hand left at extended+spread).")
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
