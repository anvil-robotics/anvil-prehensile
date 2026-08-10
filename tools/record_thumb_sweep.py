#!/usr/bin/env python3
"""Record the GLOVE's thumb output across a full (flex, abd) sweep. NO hardware motion.

Glove-only diagnostic: reads the selected glove, and for every frame logs the
curl-map's thumb outputs AND the raw underlying metrics, continuously -- so you
capture not just the four corners but the whole path BETWEEN them. That's what
lets you confirm the glove output is monotonic, uses its range, and keeps the
two thumb axes decoupled (moving flex shouldn't move abd, and vice versa).

Per frame it records:
    thumb_flex, thumb_abd, index, middle, ring, pinky   (curl-map %, 100=open)
    thumb_bend_deg   = raw MCP+IP bend  (curlmap._thumb_flex_bend_deg; ~0 straight)
    thumb_abd_deg    = raw palm-plane spread (curlmap._thumb_abd_angle_deg)
    marker           = how many times you've tapped [Enter] (label the corners)
The %s come from an UNSMOOTHED mapper (alpha=1.0) so nothing is masked; the raw
*_deg columns are calibration-independent ground truth.

Suggested sweep once recording starts (go SLOWLY):
    1) EXTENDED+SPREAD -> 2) EXTENDED+TUCKED -> 3) FLEXED+TUCKED -> 4) FLEXED+SPREAD -> back to 1
    then a pure-FLEX sweep (extend<->flex, keep abduction fixed)
    then a pure-ABD sweep  (spread<->tuck,  keep flexion fixed)
Tap [Enter] each time you reach a labeled point to drop a marker. Ctrl-C stops
and saves.

    uv run python tools/record_thumb_sweep.py                       # udcap, right
    uv run python tools/record_thumb_sweep.py --glove wuji --side left
    uv run python tools/record_thumb_sweep.py --seconds 40          # auto-stop after 40s

Output: recordings/thumb_sweep_<glove>_<side>_<HHMMSS>.{csv,npz} under the project root.
"""

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np

from prehensile.curlmap import CurlMapper, _thumb_flex_bend_deg, _thumb_abd_angle_deg
from prehensile.profiles import GLOVES
from prehensile.command import L6_SDK_ORDER

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "recordings"


def _summarize(name: str, arr: np.ndarray, unit: str = "") -> str:
    if arr.size == 0:
        return f"  {name:16s} (no data)"
    return (f"  {name:16s} min={arr.min():6.1f}{unit}  max={arr.max():6.1f}{unit}  "
            f"range={arr.max() - arr.min():6.1f}{unit}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record the glove's thumb output across a full (flex, abd) sweep (NO hardware)")
    ap.add_argument("--glove", choices=sorted(GLOVES), default="udcap")
    ap.add_argument("--side", choices=("left", "right"), default="right",
                    help="which glove hand to read (default: right)")
    ap.add_argument("--port", type=int, default=5555, help="UDP port HandDriver streams to (udcap only)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=None,
                    help="auto-stop after this many seconds (default: until Ctrl-C)")
    ap.add_argument("--out", default=None,
                    help="output path prefix (default: recordings/thumb_sweep_<glove>_<side>_<HHMMSS>)")
    args = ap.parse_args()

    glove = GLOVES[args.glove]
    # UNSMOOTHED (alpha=1.0) so per-frame values aren't lagged; abd sense follows
    # the glove profile so the %s match what teleop would emit.
    mapper = CurlMapper(side=args.side, alpha=1.0, abd_invert=glove.abd_invert)

    print(__doc__.split("\n\n")[0])
    print(f"\nglove={glove.name}  side={args.side}  "
          f"{'UDP :' + str(args.port) if args.glove == 'udcap' else 'SDK'}")

    try:
        source_cm = glove.build_source(args.side, args.port)
    except OSError as exc:
        print(f"Cannot bind UDP :{args.port}: {exc}")
        return 1

    rows = []  # (t, marker, a0..a5, bend, abd_deg)
    kps = []
    marker = {"n": 0}

    def _marker_listener():
        while True:
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return
            marker["n"] += 1
            print(f"   >> marker {marker['n']} dropped")

    try:
        with source_cm as src:
            print("\nMove your thumb through the corners + pure-axis sweeps (see the header).")
            print("Tap [Enter] at each labeled point to drop a marker. Ctrl-C to stop & save.")
            try:
                input("Press [Enter] to START recording... ")
            except (EOFError, KeyboardInterrupt):
                print("\naborted before recording.")
                return 0

            threading.Thread(target=_marker_listener, daemon=True).start()

            budget = 1.0 / args.fps
            t0 = time.monotonic()
            last_print = 0.0
            print()
            try:
                while True:
                    loop_t = time.monotonic()
                    t = loop_t - t0
                    if args.seconds is not None and t >= args.seconds:
                        break
                    kp = src.poll()
                    if kp is not None:
                        angles = mapper(kp)
                        if angles is not None:
                            kp = np.asarray(kp, dtype=np.float64)
                            bend = _thumb_flex_bend_deg(kp)
                            abd_deg = _thumb_abd_angle_deg(kp)
                            rows.append([t, marker["n"], *angles, bend, abd_deg])
                            kps.append(kp)
                            if loop_t - last_print > 0.08:  # ~12 Hz console
                                by = dict(zip(L6_SDK_ORDER, angles))
                                print(f"\rt={t:6.1f}s  flex={by['thumb_flex']:5.1f}  "
                                      f"abd={by['thumb_abd']:5.1f}  |  bend={bend:5.1f}deg  "
                                      f"abd={abd_deg:5.1f}deg  [marker {marker['n']}]   ",
                                      end="", flush=True)
                                last_print = loop_t
                    dt = time.monotonic() - loop_t
                    if dt < budget:
                        time.sleep(budget - dt)
            except KeyboardInterrupt:
                pass
            print("\nstopped.")
    except (OSError, RuntimeError) as exc:
        print(f"\nCannot open {glove.name} glove: {exc}")
        return 1

    if not rows:
        print("No frames recorded -- is the glove streaming? Nothing saved.")
        return 1

    # ---- save -------------------------------------------------------------- #
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    prefix = Path(args.out) if args.out else OUT_DIR / f"thumb_sweep_{glove.name}_{args.side}_{stamp}"
    csv_path = prefix.with_suffix(".csv")
    npz_path = prefix.with_suffix(".npz")

    header = ["t_s", "marker", *L6_SDK_ORDER, "thumb_bend_deg", "thumb_abd_deg"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{r[0]:.4f}", int(r[1])] + [f"{v:.3f}" for v in r[2:]])

    data = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(
        npz_path,
        t=data[:, 0], marker=data[:, 1].astype(int),
        angles=data[:, 2:8], thumb_bend_deg=data[:, 8], thumb_abd_deg=data[:, 9],
        kp=np.asarray(kps, dtype=np.float32),
        slots=np.array(L6_SDK_ORDER), glove=glove.name, side=args.side,
    )

    # ---- summary ----------------------------------------------------------- #
    by_col = {name: data[:, 2 + i] for i, name in enumerate(L6_SDK_ORDER)}
    print(f"\nrecorded {len(rows)} frames over {data[-1, 0]:.1f}s, {marker['n']} markers.")
    print(_summarize("thumb_flex %", by_col["thumb_flex"]))
    print(_summarize("thumb_abd %", by_col["thumb_abd"]))
    print(_summarize("thumb_bend deg", data[:, 8], "deg"))
    print(_summarize("thumb_abd deg", data[:, 9], "deg"))
    print(f"\nsaved:\n  {csv_path}\n  {npz_path}")
    print("Tip: plot thumb_flex & thumb_abd vs t_s; during a pure-flex sweep abd should stay flat "
          "(and vice versa) if the axes are decoupled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
