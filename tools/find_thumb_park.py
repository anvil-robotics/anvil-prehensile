#!/usr/bin/env python3
"""Live LEFT-hand tool to find a thumb "park" value for the grasp park-lock.

Parks one channel (default ``thumb_abd`` -- the thumb opposition/spread axis;
override with ``--channel``) at a fixed value while every other channel keeps
tracking the glove normally, so you can grasp a real object and tune the parked
value by hand until it looks/feels right. That value is what Stage 2 writes into
``configs/curl_tuning.yml`` as the ``park`` key on that channel (see
``prehensile.curlmap.CurlMapper.locked``/``set_park``).

At the console, type at any time (Enter to submit):
    <number>   set the parked channel to that value directly (0-100)
    +          nudge the park up by --step
    -          nudge the park down by --step
    q / quit   stop and print the chosen value

Requires: HandDriver streaming Content=Quater to 127.0.0.1:<port> (default
5555), and the LEFT L6 hand up on the given SocketCAN interface (default
hand_l, the canonical name applied by tools/bring_up_hand.py; verify with
``ip -details link show hand_l``).

    uv run python tools/find_thumb_park.py                       # park thumb_abd (default)
    uv run python tools/find_thumb_park.py --channel thumb_flex  # park a different channel
    uv run python tools/find_thumb_park.py --start 20 --step 2.5

SAFETY: the L6 will move (the parked channel snaps immediately; the rest track
the glove). Clear the area before you start.
"""

import argparse
import threading
import time

from prehensile.command import L6_SDK_ORDER
from prehensile.curlmap import CurlMapper
from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning
from prehensile.udcap import UDCapSource


def _listener(mapper: CurlMapper, channel: str, state: dict, step: float,
              stop: threading.Event) -> None:
    """Daemon thread: read stdin lines and update the live park for ``channel``.

    ``state["cur"]`` is the shared current value (clamped to [0,100]); a bare
    number sets it absolutely, "+"/"-" nudge it by ``step``, "q"/"quit"/EOF/
    Ctrl-C stop the run. Every change is pushed onto ``mapper`` immediately
    via ``set_park``.
    """
    while True:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            stop.set()
            return
        if not line:
            continue
        if line in ("q", "quit"):
            stop.set()
            return
        if line == "+":
            cur = min(100.0, state["cur"] + step)
        elif line == "-":
            cur = max(0.0, state["cur"] - step)
        else:
            try:
                cur = min(100.0, max(0.0, float(line)))
            except ValueError:
                print(f"   ? unrecognized input {line!r} -- type a number, '+', '-', or 'q'")
                continue
        state["cur"] = cur
        mapper.set_park(channel, cur)
        print(f"   {channel} park = {cur:g}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find the LEFT L6 thumb park value by grasping a real object (MOVES HARDWARE)")
    ap.add_argument("--speed", type=float, default=25.0,
                    help="motor speed 0-100 (default 25; motors won't move at 0)")
    ap.add_argument("--channel", default="thumb_abd", choices=L6_SDK_ORDER,
                    help="which L6 channel to park (default: thumb_abd, the opposition/spread axis)")
    ap.add_argument("--interface", default="hand_l",
                    help="L6 SocketCAN interface for the left hand (default: hand_l)")
    ap.add_argument("--port", type=int, default=5555, help="UDP port HandDriver streams to")
    ap.add_argument("--start", type=float, default=15.0, help="initial park value for --channel")
    ap.add_argument("--step", type=float, default=5.0, help="+/- nudge size")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    # realhand/CAN imported here (lazily, like teleop.py) so --help works
    # without the SDK/bus present.
    import can
    from realhand import L6
    from realhand.exceptions import CANError

    print(__doc__.split("\n\n")[0])
    print(f"\nside=left  channel={args.channel}  interface={args.interface}  port={args.port}  "
          f"speed={args.speed}  start={args.start:g}  step={args.step:g}")

    # require_couple_low=True: the same safety gate the --map curl CLI paths use
    # (prehensile/teleop.py, prehensile/viz.py). This tool runs the curl map with
    # locked=True while you put your own hand around a real object, so a silently
    # defaulted thumb_flex.couple_low -- which closes the thumb's full travel into
    # a parked, tucked thumb_abd -- would close it straight into your fingers.
    # Fail loudly on a missing/unreadable tuning file rather than falling back.
    mapper = CurlMapper(
        side="left",
        tuning=resolve_tuning(DEFAULT_TUNING_PATH, side="left", require_couple_low=True),
    )
    mapper.locked = True
    mapper.set_park(args.channel, args.start)

    state = {"cur": args.start}
    stop = threading.Event()
    threading.Thread(target=_listener, args=(mapper, args.channel, state, args.step, stop),
                     daemon=True).start()

    print(f"\nType a number to set the {args.channel} park directly, '+'/'-' to nudge by "
          f"{args.step:g}, or 'q' to stop. Grasp a real object and tune until it looks right.")
    print("(the commanded L6 angles print a few times a second; '*' marks the parked channel)\n")

    try:
        with UDCapSource(port=args.port, side="left") as src:
            try:
                with L6(side="left", interface_name=args.interface) as hand:
                    hand.speed.set_speeds([args.speed] * 6)  # motors won't move at speed 0
                    time.sleep(0.2)

                    budget = 1.0 / args.fps
                    last_kp = None
                    last_status = 0.0  # throttle the angle readout to ~3 Hz
                    while not stop.is_set():
                        t0 = time.monotonic()
                        kp = src.poll()
                        if kp is None:
                            kp = last_kp  # no fresh frame: hold last pose
                        else:
                            last_kp = kp
                        if kp is not None:
                            angles = mapper(kp)
                            if angles is not None:
                                hand.angle.set_angles(angles)
                                if t0 - last_status >= 0.3:
                                    last_status = t0
                                    cells = [
                                        f"{name}={val:5.1f}{'*' if name == args.channel else ' '}"
                                        for name, val in zip(L6_SDK_ORDER, angles)
                                    ]
                                    print("  ".join(cells))
                        dt = time.monotonic() - t0
                        if dt < budget:
                            time.sleep(budget - dt)
            except (CANError, can.CanError, OSError) as exc:
                print(f"\ncould not open L6 on '{args.interface}' -- check "
                      f"'ip -details link show {args.interface}' and that the hand is powered "
                      f"({exc})")
                return 1
    except (OSError, RuntimeError) as exc:
        print(f"\nCannot open UDCap glove source: {exc}")
        return 1

    cur = state["cur"]
    print(f"\n>> chosen {args.channel} park = {cur:g} -- add it to the {args.channel} line in "
          f"configs/curl_tuning.yml, e.g.  {args.channel}: {{..., park: {cur:g}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
