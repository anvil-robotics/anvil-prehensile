# Anvil-Prehensile

An end-to-end teleoperation stack for dexterous manipulation, bridging the **UDCap**/**Wuji** glove and the **REALHAND L6** hand. Both sides are normalized to the same `(21, 3)` MediaPipe-order keypoint frame; everything downstream of that seam shares one code path.

This file covers getting running and how the pipeline works. The full flag
reference, bring-up detail and troubleshooting are in [`RUNBOOK.md`](RUNBOOK.md).

## Quick start

Defaults are `--glove udcap --hand left --map curl`. For the right hand, add `--hand right`
to every command below.

```bash
uv sync
uv run python -m prehensile.offline_check   # glove-free environment check
```

**UDCap** — in **HandDriver**
([download](https://drive.google.com/drive/folders/1g4cA-hjEnTmQNcY6zyK04Bo9R_qfPcOC),
[guide](https://udexreal.gitbook.io/udexreal-docs/robotics/usage-for-udexreal-robotics-products#id-3.-udexreal-robotics-teleoperation-system)) set Data Transmission **ON**, Format **Quater**,
target `127.0.0.1:5555`, FPS 120. For UDCap, always pass `--map curl` here: the `retarget` is degenerate for UDCap ([why](#angle-mapping)).

```bash
uv run python tools/probe_udp.py --seconds 2   # expect ~120 pkt/s, then stop it (it holds :5555)
uv run python -m prehensile.teleop --map curl  # dry run: prints angles, opens no CAN
```

**Wuji** — close Wuji Studio first: only one session may talk to the glove.
No `--map` needed; the default `curl` is what's validated here.

```bash
uv run python -m prehensile.teleop --glove wuji  # dry run: prints angles, opens no CAN
```

```bash
sudo python3 tools/bring_up_hand.py         # before any --live run; re-run after a reboot or replug
uv run python -m prehensile.viz --map curl  # sim preview: MuJoCo, never touches the hand
```

A good dry run reads near 100 on every channel for a flat open hand, near 0 for a
fist, and moves each finger in its own slot. Once it does:

```bash
uv run python -m prehensile.teleop --map curl --live --speed 25   # ⚠ MOVES THE HAND
```

Clear the area first. There is a 3-2-1 countdown, and **Ctrl-C leaves the hand
holding its last commanded pose** — no watchdog opens it. Full safety notes:
RUNBOOK.md's [Safety](RUNBOOK.md#safety).

For the full per-glove comparison (link, `--map`, hardware status, and more),
see RUNBOOK.md's [Choose your glove](RUNBOOK.md#choose-your-glove) table. Full
flag reference, bring-up detail and troubleshooting are also there.

## Technical

### Pipeline

```
  Wuji glove (USB)                      UDCap gloves (dongle + HandDriver)
  prehensile/wuji.py                    prehensile/udcap.py
  wuji_sdk gives 21 kp directly         UDP 127.0.0.1:5555, Quater, ~120 Hz
       |                                -> 15 quats -> fk.py + rest_pose.py -> 21 kp
       +------------------+-------------------+
                          v
      shared (21,3) MediaPipe keypoint seam (wrist-local, meters)
       +------------------+-------------------+
       v                                      v
  --map retarget              --map curl (default)
  prehensile/retarget.py                prehensile/curlmap.py
  dex_retargeting optimizer -> qpos     geometric per-finger curl, EMA-smoothed
  prehensile/command.py                       |
  qpos -> 6x [0,100] by joint name            |
       +------------------+-------------------+
                          v
  prehensile/teleop.py --live -> realhand SDK -> SocketCAN
```

`prehensile.viz` renders the same mapping in MuJoCo and never touches the
hand. The curl + UDCap path also runs in the workcell as the ROS
`hand_teleop_node`, which imports `prehensile.*` only.

### Angle mapping

- **`--map curl`** (default) — bypasses the optimizer: the four fingers use the
  scale-invariant chord ratio `|tip − wrist| / Σ(bone lengths)`; thumb
  flexion is the MCP+IP bend angle; thumb abduction is the metacarpal's
  elevation above the palm plane.
- **`--map retarget`**  — the `dex_retargeting` vector optimizer:
  keypoints → qpos → 6 × `[0,100]`.

Why curl exists: the optimizer is **degenerate for UDCap** — index and
middle pinned open, ring and thumb unstable — because UDCap *reconstructs*
its keypoints from a bespoke FK in a frame the optimizer was never tuned
for. Wuji is fine because its SDK emits correctly-framed keypoints directly.
Curl is purely geometric on wrist-local distances, so it works for either
glove.

**Thumb←index coupling** (only effect on UDCap glove) — enabled unconditionally, but it
only takes effect *while the glove's `bButton` grasp lock is engaged*. Under the
lock, `thumb_flex` stops using its own metric and follows the index, rescaled
onto `[couple_low, 100]` — a proportional pinch. Unlocked, both channels track
independently as usual.

Tuning: [`configs/curl_tuning.yml`](configs/curl_tuning.yml), self-documenting.

### Per-glove open/close sense

Two hardware-confirmed constants in `prehensile/profiles.py`; callers never
set them by hand.

| constant | glove | value | why |
| -------- | ----- | ----- | --- |
| `invert_flex` (retarget only) | Wuji | `False` | matches the L6's hand-native convention |
| `invert_flex` (retarget only) | UDCap | `True` | opposite sense — uninverted, an open hand *closed* the robot |
| `abd_invert` (curl only) | Wuji | `False` | thumb-abduction sense, not open/close |
| `abd_invert` (curl only) | UDCap | `True` | same axis, opposite sense on this glove |

`thumb_abd` is the one exception to `invert_flex`: an abduction axis, so it's
always `(1 − normalized) * 100` for both gloves.

## Contributing

Anvil-Prehensile is Apache-2.0 licensed. Contributions and pull requests are
welcome, and so are issues if you hit a bug or a rough edge. Before opening a
PR, please make sure `uv run python -m prehensile.offline_check` passes.
