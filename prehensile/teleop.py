#!/usr/bin/env python3
"""Live glove -> REALHAND L6 teleop.

DRY-RUN BY DEFAULT: reads the selected glove (``--glove wuji|udcap``, default
``udcap``) for the selected hand (``--hand left|right``, default ``left``) and
prints the 6 retargeted angles; NO hardware motion. Pass --live to also open
the L6 and send commands (fingers move). Retargeting is wrist-relative, so it
is immune to the glove's IMU yaw drift.

udcap needs HandDriver streaming the Quater content to 127.0.0.1:<port> (see
README.md). wuji needs the Wuji glove connected and wuji_sdk installed.

    uv run python -m prehensile.teleop                     # dry-run, UDCap glove, left hand
    uv run python -m prehensile.teleop --glove wuji         # dry-run, Wuji glove
    uv run python -m prehensile.teleop --hand right --live  # drive the L6 (clear the area first!)
    uv run python -m prehensile.teleop --map retarget       # the optimizer instead (see retarget.py)

Dry-run needs only the glove source. --live also needs the L6 powered on the
hand's CAN interface. Ctrl-C stops; in --live the hand holds its last
commanded pose.
"""

import argparse
import contextlib
import time
from pathlib import Path

from prehensile.command import qpos_index_map, qpos_to_l6_angles
from prehensile.curlmap import CurlMapper
from prehensile.profiles import GLOVES, HANDS
from prehensile.tuning import DEFAULT_TUNING_PATH, resolve_tuning

# prehensile.retarget (dex_retargeting) is imported lazily, function-locally in
# main()'s --map retarget branch below -- a bare `pip install prehensile` (no
# extras) doesn't have dex_retargeting, and the default --map curl path never
# touches a retargeter (see loop()'s short-circuit at the `mapper is not None`
# check), so importing it at module scope would break `import prehensile.teleop`
# for the common case.

ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = ROOT / "assets" / "realhand_description"
JOINTS = ["thumb_flex", "thumb_abd", "index", "middle", "ring", "pinky"]
# (driven, source) channels of the thumb<-index coupling, for the readout marker.
_COUPLED_PAIR = ("thumb_flex", "index")


def _fmt(angles, parked=(), coupled=None, floored=None):
    """Format the 6-channel angle readout, e.g. ``"thumb_flex= 82.3  ..."``.

    The optional markers call out channels whose displayed number is the live
    tracked value rather than what is actually being commanded. They share one
    trailing bracket, space-separated, and are omitted entirely when all are
    empty -- which reproduces the old bare per-channel line byte-for-byte.

    ``parked`` is an iterable of channel names (e.g. ``mapper.parked_channels``)
    forced to a fixed value by the park-lock -> ``"PARKED <comma-joined names>"``.
    ``coupled`` is an optional ``(driven, source)`` channel-name pair for the
    thumb<-index coupling -> ``"COUPLED <driven><-<source>"``.
    ``floored`` is an optional ``(channel, value)`` pair for the coupling's floor
    on the driving finger's own command -> ``"FLOOR <channel>=<value>"``.
    """
    line = "  ".join(f"{n}={a:5.1f}" for n, a in zip(JOINTS, angles))
    marks = []
    if parked:
        marks.append(f"PARKED {', '.join(parked)}")
    if coupled:
        marks.append(f"COUPLED {coupled[0]}<-{coupled[1]}")
    if floored:
        marks.append(f"FLOOR {floored[0]}={floored[1]:g}")
    if marks:
        line += "  [" + "  ".join(marks) + "]"
    return line


def frame_to_angles(kp, retargeter, index_map, invert_flex: bool = True):
    """One keypoint frame -> the 6 L6 angles (0-100), or None if retargeting fails.

    This is the whole keypoints->command step; loop() calls it per frame and it is
    the unit under test for the dry-run path (no socket, no hardware needed).
    ``invert_flex`` selects the glove's open/close convention (see
    ``prehensile.command.qpos_to_l6_angles``); default True matches UDCap.
    """
    qpos = retargeter.retarget(kp)
    if qpos is None:
        return None
    return qpos_to_l6_angles(qpos, index_map, invert_flex=invert_flex)


def loop(source, retargeter, index_map, sink, fps, invert_flex=True, mapper=None):
    """Read source -> retarget -> command -> sink, at ~fps. Prints angles (throttled).

    When ``mapper`` is given, each frame also edge-detects the glove's
    bButton (``getattr(source, "bButton", False)`` -- sources without one,
    e.g. ``WujiSource``, read as never-pressed, a silent no-op) and flips
    ``mapper.locked`` on a rising edge only, so holding the button down
    toggles once, not every frame; see the park-lock in ``CurlMapper``'s
    docstring. This happens before ``mapper(kp)`` is called, so the toggle
    takes effect the same frame it is pressed, and it runs regardless of
    ``--live`` so an operator can rehearse it in dry-run.

    The sink always receives the real ``mapper(kp)`` output (parked channels
    included -- this is what actually drives the hardware). Only the console
    readout differs while locked: it shows ``mapper.last_unparked`` (the live
    tracked values underneath the lock) plus a ``[PARKED ...]`` marker naming
    the parked channels, so the operator can see what they are actually
    holding steady versus what is being sent. When the mapper also has
    ``couple_thumb_index`` on, a ``COUPLED thumb_flex<-index`` segment joins the
    same bracket -- plus ``FLOOR index=<v>`` when the coupling also clamps the
    index -- and every channel the coupling writes is dropped from the ``PARKED``
    list (the coupling is applied after the park, so it owns those). The
    ``--map retarget`` path (``mapper is None``) is unaffected either way: plain
    ``angles``, no marker.
    """
    budget = 1.0 / fps
    last_kp = None
    last_print = 0.0
    prev_button = False  # bButton state as of the previous frame, for edge detection
    prev_len = 0  # width of the last redraw's raw content, so a shrinking line still erases it
    while True:
        t0 = time.monotonic()
        kp = source.poll()
        if kp is None:
            kp = last_kp  # no fresh frame: hold last pose
            if kp is None:
                time.sleep(budget)
                continue
        else:
            last_kp = kp

        if mapper is not None:
            # Sources without a bButton (e.g. WujiSource) read as never-pressed
            # via the getattr default, making this a silent no-op for them.
            pressed = getattr(source, "bButton", False)
            if pressed and not prev_button:  # rising edge only; a held press toggles once
                mapper.locked = not mapper.locked
            prev_button = pressed

        angles = mapper(kp) if mapper is not None else frame_to_angles(kp, retargeter, index_map, invert_flex=invert_flex)
        if angles is None:
            continue
        sink(angles)  # the real (possibly parked) command always drives the sink

        if mapper is not None and mapper.locked:
            # Show the live tracked values underneath the lock rather than the
            # frozen ones actually sent. last_unparked can only be None before
            # the mapper's first valid frame, which can't be the case here
            # since angles (from that same call) already came back non-None.
            display = mapper.last_unparked if mapper.last_unparked is not None else angles
            coupled = _COUPLED_PAIR if mapper.couple_thumb_index else None
            floor = mapper.couple_index_floor if coupled else None
            floored = (_COUPLED_PAIR[1], floor) if floor is not None else None
            parked = mapper.parked_channels
            if coupled:
                # Coupling is applied after the park override, so it owns the
                # channels it writes -- don't report those as PARKED as well (a
                # park on either channel is legitimate; see configs/curl_tuning.yml).
                owned = {coupled[0]} | ({coupled[1]} if floored else set())
                parked = tuple(ch for ch in parked if ch not in owned)
        else:
            display, parked, coupled, floored = angles, (), None, None

        if t0 - last_print > 0.08:  # throttle console to ~12 Hz
            # ljust to the PREVIOUS redraw's raw width (not a fixed suffix): if
            # the line just shrank (e.g. unlocking drops "  [PARKED ...]"), this
            # pads it out far enough to fully overwrite the old one in place --
            # a fixed few-space suffix isn't enough once the marker is involved.
            content = _fmt(display, parked, coupled, floored)
            print("\r" + content.ljust(prev_len), end="", flush=True)
            prev_len = len(content)
            last_print = t0
        dt = time.monotonic() - t0
        if dt < budget:
            time.sleep(budget - dt)


def _build_parser() -> argparse.ArgumentParser:
    """The CLI parser, split out of ``main()`` so the tests can assert flag defaults
    without building a retargeter or touching hardware."""
    ap = argparse.ArgumentParser(description="glove -> REALHAND L6 teleop")
    ap.add_argument("--live", action="store_true",
                    help="open the L6 and send commands (fingers move). Default: dry-run print only.")
    ap.add_argument("--glove", choices=sorted(GLOVES), default="udcap",
                    help="glove source to read from (default: udcap)")
    ap.add_argument("--hand", choices=sorted(HANDS), default="left",
                    help="which hand to retarget/drive (default: left)")
    ap.add_argument("--interface", default=None,
                    help="L6 SocketCAN interface (live only); default is the hand's own iface")
    ap.add_argument("--speed", type=float, default=40.0, help="L6 motor speed 0-100 (live only)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--port", type=int, default=5555, help="UDP port HandDriver streams to (udcap only)")
    ap.add_argument("--map", choices=["retarget", "curl"], default="curl",
                    help="angle-mapping strategy: the direct per-finger curl map "
                         "(default, see curlmap.py) or the dex_retargeting vector optimizer")
    ap.add_argument("--tune", type=Path, default=DEFAULT_TUNING_PATH,
                    help="curl tuning YAML (--map curl only); default is the shipped "
                         f"{DEFAULT_TUNING_PATH.name} (see prehensile/tuning.py)")
    return ap


def main():
    args = _build_parser().parse_args()

    glove = GLOVES[args.glove]
    hand = HANDS[args.hand]
    CONFIG = ROOT / "configs" / hand.config
    interface = args.interface or hand.interface

    # abd_invert now comes solely from the glove profile -- there is no CLI
    # override anymore.
    abd_invert = glove.abd_invert
    tuning = None
    mapper = None
    retargeter = None
    index_map = None
    if args.map == "curl":
        # args.tune defaults to DEFAULT_TUNING_PATH but --tune can point at a
        # different file; a missing file or a missing thumb_flex.couple_low
        # both raise regardless (see resolve_tuning) -- curl tuning is not
        # something the hand can silently run without, on the shipped file or
        # any other.
        tuning = resolve_tuning(args.tune, side=hand.side, require_couple_low=True)
        print(f"curl tuning: {args.tune}")
        # Print the RESOLVED bounds: couple_low can come from a per-side
        # section, so which value applies depends on --hand. (A side-only
        # couple_low no longer risks the other hand running unfloored --
        # resolve_tuning raises for that side instead -- but the operator
        # still wants to see which numbers this run actually landed on.)
        ch = tuning or {}
        low = float(ch.get("thumb_flex", {}).get("couple_low", 0.0))
        idx_low = float(ch.get("index", {}).get("couple_low", 0.0))
        idx_note = f"index floor={idx_low:g}" if idx_low > 0.0 else "no index floor"
        print(f"thumb-index couple: ON (thumb_flex couple_low={low:g}, "
              f"{idx_note}, hand={hand.side})")
        mapper = CurlMapper(side=hand.side, abd_invert=abd_invert, tuning=tuning)
    elif args.map == "retarget":
        # dex_retargeting imported only here so the default curl path (and a
        # bare `pip install prehensile`) never needs it.
        from prehensile.retarget import L6Retargeter

        print(f"Building retargeter (glove={glove.name}, hand={hand.side})...")
        retargeter = L6Retargeter(CONFIG, URDF_DIR, side=hand.side)
        index_map = qpos_index_map(retargeter.joint_names)
        print(f"qpos index map {JOINTS} -> {index_map}")

    try:
        source_cm = glove.build_source(hand.side, args.port)
    except OSError as exc:
        print(f"Cannot bind UDP :{args.port}: {exc}")
        return 1

    try:
        with source_cm as src:
            if not args.live:
                print(f"DRY-RUN (no hardware motion). Reading {glove.name} glove "
                      f"({'UDP :' + str(args.port) if args.glove == 'udcap' else 'SDK'}). "
                      "Move your hand; Ctrl-C to stop.\n")
                with contextlib.suppress(KeyboardInterrupt):
                    loop(src, retargeter, index_map, sink=lambda a: None, fps=args.fps,
                         invert_flex=glove.invert_flex, mapper=mapper)
                print("\nstopped.")
                return 0

            # realhand/CAN imported only here so dry-run (and the tests) never need them.
            import can
            from realhand import L6
            from realhand.exceptions import CANError

            print("SAFETY: the L6 will move. Clear the area. Starting in", end=" ", flush=True)
            for n in (3, 2, 1):
                print(n, end=" ", flush=True)
                time.sleep(1.0)
            print("go!\n")
            try:
                with L6(side=hand.side, interface_name=interface) as l6hand:
                    l6hand.speed.set_speeds([args.speed] * 6)  # motors won't move at speed 0
                    time.sleep(0.2)
                    with contextlib.suppress(KeyboardInterrupt):
                        loop(src, retargeter, index_map,
                             sink=lambda a: l6hand.angle.set_angles(a), fps=args.fps,
                             invert_flex=glove.invert_flex, mapper=mapper)
                    print("\nstopped (hand holds its last commanded pose).")
            except (CANError, can.CanError, OSError) as exc:
                print(f"\nCAN/L6 error: {exc}")
                print(f"Check the L6 is powered and {interface} is up: "
                      f"ip -details link show {interface}")
                return 1
    except (OSError, RuntimeError) as exc:
        print(f"Cannot open {glove.name} glove: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
