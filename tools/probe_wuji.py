#!/usr/bin/env python3
"""Probe the Wuji glove and report whether it is ready for prehensile teleop.

The Wuji counterpart to ``tools/probe_udp.py``, but NOT the same kind of tool.
``probe_udp`` is a passive wire sniffer: HandDriver unicasts datagrams at a
port, so that tool binds a socket and never touches the glove. Wuji has no
readable wire -- it is a Zenoh/UDP session whose framing only ``wuji_sdk``
decodes -- so everything here goes through the SDK instead.

There is no single SDK call for "is this glove ready?"; the SDK answers
"connected" (``scan`` / ``is_connected``) but has no notion of what prehensile
needs. This composes the primitives into one verdict:

  * a glove is discoverable at all, and on which transport/address
  * which hand it is. The glove knows (``hand_side()``), so this is DETECTED,
    never passed in, and reported as the ``--hand`` value to run teleop with.
    That matters because ``WujiSource`` ignores its own ``side`` argument (see
    its docstring), so nothing downstream can catch a wrong ``--hand``: a left
    glove driving ``--hand right`` fails silently but plausibly.
  * whether a calibrated hand URDF is loaded, or the built-in default
  * ``hand_skeleton`` actually delivers sane 21-landmark frames, at what rate.
    ``read_latest_keypoints`` returns None both for "no new frame yet" and for
    "the stream is dead", so only counting across a window separates them

Run (from the repo root):
    uv run python tools/probe_wuji.py                  # scan, connect, 5 s of frames
    uv run python tools/probe_wuji.py --seconds 2
    uv run python tools/probe_wuji.py --scan-only      # discovery only, see below

``--scan-only`` needs no session, so it is the one mode safe to run while
teleop holds the glove. Every other mode CONNECTS, which takes the glove's
exclusive session -- close Wuji Studio and any teleop/viz first, exactly as for
teleop itself.

Exit code 1 (with guidance) on a hard fault: nothing discovered, connect
failed, no frames, malformed keypoints, or a handedness the stack cannot run
(anything but left/right -- ``prehensile.profiles.HANDS``). A built-in
default URDF and a frozen-looking stream are reported as warnings and do NOT
fail the run -- the shipped setup runs on the built-in URDF today.

LAZY IMPORT: ``wuji_sdk`` is imported inside ``main()``, matching
``prehensile.wuji``'s policy, so this module imports (and its verdict helpers
run) on a machine with no glove SDK at all.
"""

import argparse
import contextlib
import time

import numpy as np

from prehensile.wuji import connect_glove, read_latest_keypoints

# MediaPipe landmark chains, wrist-rooted. Order matches prehensile's (21,3)
# seam; index 0 is the wrist and 4/8/12/16/20 are the fingertips.
CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
SPAN_LANDMARK = 12  # middle fingertip; |kp[12] - kp[0]| is the hand's reach

# Thresholds are MEASURED, not guessed -- captured off WG1KA03260524030
# (firmware 0.11.4) on 2026-08-27: span 0.1838 m, bones 0.0185-0.0869 m, and a
# max per-joint sd of 3.6e-4 m with the glove sitting still on the bench. The
# ranges below are those numbers widened to cover a fist (short span) and a
# larger hand, so a real frame can never sit near an edge.
SPAN_RANGE_M = (0.05, 0.30)
BONE_RANGE_M = (0.005, 0.15)
# A live-but-motionless glove still jitters ~3.6e-4 m; a stream repeating one
# frame is exactly 0. Two orders of magnitude below the measured floor.
STILL_SD_M = 1e-5


def _build_parser():
    p = argparse.ArgumentParser(
        description="Probe the Wuji glove: discovery, handedness, hand model, "
                    "and hand_skeleton rate/sanity."
    )
    p.add_argument("--seconds", type=float, default=5.0,
                   help="how long to sample hand_skeleton (default: 5.0)")
    p.add_argument("--scan-only", action="store_true",
                   help="discovery only -- takes no session, so it is safe to "
                        "run while teleop holds the glove")
    p.add_argument("--verbose", action="store_true",
                   help="restore the SDK's own INFO logging (~200 lines while "
                        "connecting); off by default so the report stays legible")
    # Deliberately NO --hand: the glove reports its own side, so asking the
    # operator would only create a second source of truth to disagree with.
    return p


def _enum_name(value):
    """Bare name of an SDK enum member.

    The bindings are inconsistent: ``TransportType`` exposes ``.name`` ('UDP')
    but ``DeviceType`` does not, and its str() is the qualified
    'DeviceType.WujiGlove'. Take the last dotted segment either way.
    """
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def describe_device(d):
    """One line for a ``DiscoveredDevice`` (fields verified against the SDK stub)."""
    return (f"sn={d.sn} type={_enum_name(d.device_type)} "
            f"transport={_enum_name(d.transport_type)} addr={d.address}")


def keypoint_verdict(kp):
    """(ok, detail) for one (21,3) frame: shape, finiteness, and real scale.

    Scale matters because a decode can succeed and still be useless: an FK that
    produced nothing yields an all-zero frame, and a units change yields a hand
    the size of a room. Both parse perfectly.
    """
    kp = np.asarray(kp)
    if kp.shape != (21, 3):
        return False, f"shape {tuple(kp.shape)}, expected (21, 3)"
    if not np.isfinite(kp).all():
        bad = int((~np.isfinite(kp)).any(axis=1).sum())
        return False, f"{bad} landmark(s) hold non-finite values (NaN/inf)"

    span = float(np.linalg.norm(kp[SPAN_LANDMARK] - kp[0]))
    lo, hi = SPAN_RANGE_M
    if not lo <= span <= hi:
        return False, (f"wrist->middle-fingertip span {span:.4g} m is outside "
                       f"{lo}-{hi} m -- collapsed FK, or not metres?")

    bones = [float(np.linalg.norm(kp[c[i + 1]] - kp[c[i]]))
             for c in CHAINS.values() for i in range(len(c) - 1)]
    blo, bhi = BONE_RANGE_M
    worst = min(bones) if min(bones) < blo else max(bones)
    if not blo <= worst <= bhi:
        return False, (f"implausible bone length {worst:.4g} m (expected "
                       f"{blo}-{bhi} m)")
    return True, (f"21x3 finite, span {span:.3f} m, "
                  f"bones {min(bones):.3f}-{max(bones):.3f} m")


def motion_verdict(frames):
    """(ok, detail): does the stream actually change across frames?"""
    if len(frames) < 2:
        return True, f"only {len(frames)} frame(s) -- not enough to judge"
    sd = float(np.linalg.norm(np.stack(frames).std(axis=0), axis=-1).max())
    if sd < STILL_SD_M:
        return False, (f"FROZEN: every frame identical (max per-joint sd "
                       f"{sd:.2g} m) -- a live glove jitters even at rest")
    return True, f"moving (max per-joint sd {sd:.2g} m)"


def side_verdict(reported):
    """(ok, detail) for the handedness the glove reports about itself.

    Detected, not supplied. The only way this fails is a side prehensile cannot
    run: ``profiles.HANDS`` knows left and right and nothing else.
    """
    side = str(reported).strip().lower()
    if side not in ("left", "right"):
        return False, (f"glove reports an unrecognised hand side {reported!r} "
                       f"-- prehensile can only run 'left' or 'right'")
    return True, f"{side} detected -- run teleop with --hand {side}"


def read_model_path(glove):
    """The custom hand URDF path, or None when the SDK has none set.

    The SDK RAISES ``WujiException: Path not found: calibration.hand_model_path``
    rather than returning "" when unset (measured 2026-08-27), so "unset" has to
    be read out of the exception. Anything else is a real fault and propagates.
    """
    try:
        return glove.hand_model_path().get()
    except Exception as exc:
        if "not found" in str(exc).lower():
            return None
        raise


def model_verdict(path):
    """(ok, detail) for the loaded hand model."""
    if not path:
        return False, ("built-in default URDF -- calibration.hand_model_path is "
                       "unset, so online IK is NOT using a calibrated hand")
    return True, f"custom URDF: {path}"


def collect(read, seconds, monotonic=time.monotonic, sleep=time.sleep,
            interval=0.002):
    """Call ``read()`` for ``seconds``; return (frames, elapsed).

    ``read`` is the drain-latest reader (``read_latest_keypoints`` bound to a
    subscription); None means nothing new was queued this tick. The clock and
    sleep are injectable so the window is testable without real time.
    """
    frames = []
    start = monotonic()
    deadline = start + seconds
    try:
        while monotonic() < deadline:
            kp = read()
            if kp is not None:
                frames.append(kp)
            sleep(interval)
    except KeyboardInterrupt:
        print("\n(interrupted - reporting what was captured so far)")
    return frames, monotonic() - start


def format_row(status, label, detail):
    """One report row. The status field is fixed-width so that a FAIL row does
    not shift the label column relative to the ok rows around it."""
    return f"  [{status:4s}] {label:11s} {detail}"


def render_report(sections):
    """``[(title, [(status, label, detail), ...]), ...]`` -> the whole block."""
    out = []
    for title, rows in sections:
        out.append(f"{title}:")
        out.extend(format_row(*row) for row in rows)
        out.append("")
    return "\n".join(out).rstrip()


def verdict(failures, warnings):
    """The single bottom line. Warnings never make a run NOT READY."""
    if failures:
        return f"NOT READY - {', '.join(failures)}"
    if warnings:
        return f"READY (with warnings: {', '.join(warnings)})"
    return "READY"


def main():
    args = _build_parser().parse_args()
    import wuji_sdk  # lazy: see the module docstring
    from wuji_sdk import SdkManager

    # The SDK logs ~200 INFO lines while connecting and subscribing (every URDF
    # link and joint, twice). Printing rows as they were computed put them in
    # the middle of that, so rows are collected and rendered once at the end AND
    # the SDK is quieted. Its own warnings are suppressed too: everything worth
    # acting on is re-reported below from the API, not scraped from the log.
    with contextlib.suppress(Exception):
        wuji_sdk.set_log_level("info" if args.verbose else "error")

    manager = SdkManager.instance()
    sections, failures, warnings = [], [], []

    print("probe_wuji: scanning (USB + UDP) ...")
    devices = manager.scan()
    if not devices:
        print("NO DEVICES - the SDK discovered no Wuji glove.\n"
              "Is it powered and on the same network/USB bus? A glove reachable "
              "over UDP shows up as an ip:port address.")
        return 1
    sections.append((f"discovery ({len(devices)} device(s))",
                     [("ok", "glove", describe_device(d)) for d in devices]))

    if args.scan_only:
        print()
        print(render_report(sections))
        return 0

    previous_user = manager.current_user()
    # Match teleop exactly: WujiSource.__enter__ switches to the default user
    # for its built-in URDF FK, so the probe must too or it would report a
    # hand model teleop will not actually use.
    manager.switch_to_default_user()
    print(f"connecting (takes the glove's exclusive session), "
          f"then sampling {args.seconds:g}s ...")
    try:
        glove = connect_glove(manager)
        if glove is None:
            print("CONNECT FAILED - no glove among the discovered devices, or a "
                  "session never cleared. Close Wuji Studio and any teleop/viz.")
            return 1

        sections.append(("identity", [
            ("ok", "serial", glove.sn().get()),
            ("ok", "firmware", glove.version().get()),
            ("ok", "address", f"{glove.ip().get()}:{glove.port().get()}"),
            ("ok", "user", previous_user.get("display_name")),
        ]))

        rows = []
        ok, detail = side_verdict(glove.hand_side().get())
        rows.append(("ok" if ok else "FAIL", "hand side", detail))
        if not ok:
            failures.append("unrecognised hand side")

        ok, detail = model_verdict(read_model_path(glove))
        rows.append(("ok" if ok else "warn", "hand model", detail))
        if not ok:
            warnings.append("built-in default URDF")

        sub = glove.hand_skeleton().subscribe()
        frames, elapsed = collect(lambda: read_latest_keypoints(sub), args.seconds)
        if not frames:
            rows.append(("FAIL", "stream", f"no frames in {elapsed:.2f}s"))
            failures.append("no keypoint frames")
        else:
            rate = len(frames) / elapsed if elapsed > 0 else float("nan")
            rows.append(("ok", "stream", f"{len(frames)} frames in "
                                         f"{elapsed:.2f}s -> {rate:.1f} Hz"))
            ok, detail = keypoint_verdict(frames[-1])
            rows.append(("ok" if ok else "FAIL", "keypoints", detail))
            if not ok:
                failures.append("malformed keypoints")
            ok, detail = motion_verdict(frames)
            rows.append(("ok" if ok else "warn", "motion", detail))
            if not ok:
                warnings.append("frozen stream")
        sections.append(("readiness", rows))
    finally:
        with contextlib.suppress(Exception):
            manager.switch_user(previous_user["user_id"])
        with contextlib.suppress(Exception):
            manager.disconnect_all()

    # Everything below prints only after teardown, so no SDK output can land
    # inside the block.
    print()
    print(render_report(sections))
    print()
    print(verdict(failures, warnings))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
