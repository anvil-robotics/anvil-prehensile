# Runbook — the glove→L6 CLI

Flag reference for the standalone glove→L6 pipeline. Theory:
[`README.md`](README.md).

Everything runs from this directory with `uv run python -m …`. Use that form, not
`uv run <script>`: a console script obeys its own shebang, which points at
whatever path the venv was built in.

## Choose your glove

Both gloves feed the same `(21,3)` keypoint seam, so the rest of this runbook —
bring-up of the hand, flags, console output, safety, tuning — is glove-agnostic.
They diverge only at the front of the pipeline: how you connect, and which
`--map` to use.

| | UDCap | Wuji |
| --- | --- | --- |
| link | HandDriver → UDP `127.0.0.1:5555` | USB + `wuji_sdk` |
| before you start | Data Transmission ON, Format **Quater**, FPS 120 | calibrate in Wuji Studio, then close it (only one session may talk to the glove) |
| verify the wire | `uv run python tools/probe_udp.py --seconds 2` | not applicable — the SDK scan in `--glove wuji` is the check |
| `--map` | **`curl`** — `retarget` is degenerate for UDCap | **the default `retarget`** — validated for Wuji |
| `--port` | applies | ignored |
| `bButton` → park lock + thumb←index pinch | yes | **no** — `WujiSource` exposes no `bButton`, so neither feature ever engages |
| `configs/curl_tuning.yml` | required | not read on the default `retarget` path; still required if you pass `--map curl` |
| hardware status | left and right both validated | left validated; right is code-level only, pending a live confirm |
| launcher | `scripts/teleop.sh` | same script |

On Wuji, `--map curl` still works — it's a degraded mode: no park lock, no
pinch, and `couple_low` is validated (the check is glove-independent) but has
no effect, since there's no `bButton` to engage it.

## Bring-up

**UDCap** — Download **HandDriver**
([download](https://drive.google.com/drive/folders/1g4cA-hjEnTmQNcY6zyK04Bo9R_qfPcOC),
[guide](https://udexreal.gitbook.io/udexreal-docs/robotics/usage-for-udexreal-robotics-products#id-3.-udexreal-robotics-teleoperation-system)), Data Transmission ON, Format **Quater**, `127.0.0.1:5555`, FPS 120.
Check the wire with `uv run python tools/probe_udp.py --seconds 2` — expect ~120
pkts/s and a `verdict:` naming `TeleopDataQuat` or `JSON`; either is fine. It
holds port 5555, so stop it before teleop. Use `--map curl`: `--map retarget` is
degenerate for this glove.

**Wuji** — calibrate the glove first in **Wuji Studio**
([calibration guide](https://docs.wuji.tech/docs/en/wuji-studio/latest/calibration/),
[install](https://docs.wuji.tech/docs/en/wuji-studio/latest/installation/)).
Create a **named profile** before you calibrate — the built-in `Default` profile
does not persist results, so calibrating under it silently changes nothing and
the SDK keeps falling back to a generic hand URDF. Studio then walks you through
six guided poses; results land in `~/.wuji/sdk/users/<profile-id>/models/` and
the SDK reads the same directory. After all above, close Wuji Studio since only one session may talk to the glove.

**The hand**, before any `--live` run:

```bash
sudo python3 tools/bring_up_hand.py
```

Every tool here defaults to a canonical name (`hand_l`/`hand_r`) rather than
probing, and udev gives hand adapters no name — so those names exist only after
this runs, and not across a reboot or replug. The ROS workcell does not need it.

## Entry points

| command | purpose | moves the hand |
| ------- | ------- | :-: |
| `uv run python -m prehensile.teleop` | glove → angles, printed. No CAN is opened; `realhand` is never imported. | no |
| `uv run python -m prehensile.teleop --live` | same, but drives the L6 | **yes** |
| `uv run python -m prehensile.viz` | MuJoCo preview with keypoints overlaid. Needs a display. | no |
| `uv run python -m prehensile.offline_check` | synthetic smoke test. No flags, left hand hardcoded. | no |
| `uv run python demo.py` | glove-free L6 self-test: OPEN → CLOSE → OPEN | **yes** |
| `./scripts/teleop.sh [args…]` | UDCap: probes the stream for 2 s first. Wuji: skips the probe (no UDP stream to check). Either way, execs `teleop` with your args. | passthrough |

A dry run is good when a flat open hand reads near 100 on every channel, a fist
near 0, and each finger moves its own slot. In `viz`, check finger identity, curl
direction, abduction, and that the model is not mirrored.

## Flags

`T` = `teleop`, `V` = `viz`.

Shared by both:

| flag | T | V | default | what it does | moves hw |
| ---- | :-: | :-: | ------- | ------------ | :-: |
| `--glove {udcap,wuji}` | ● | ● | `udcap` | glove source | no |
| `--hand {left,right}` | ● | ● | `left` | URDF, retarget config, L6 side, default interface | no |
| `--map {retarget,curl}` | ● | ● | `retarget` | optimizer, or the direct curl map | no |
| `--fps N` | ● | ● | `60.0` | loop rate (console readout is throttled to ~12 Hz separately) | no |
| `--port N` | ● | ● | `5555` | UDP port; ignored for Wuji | no |

Specific to one command:

| flag | T | V | default | what it does | moves hw |
| ---- | :-: | :-: | ------- | ------------ | :-: |
| `--live` | ● | | off | open the L6 and send the angles | **yes** |
| `--interface IFACE` | ● | | from `--hand` (`hand_l`/`hand_r`) | which physical hand receives commands | **yes** |
| `--speed N` | ● | | `40.0` | motor speed 0–100, set once at startup. Motors do not move at 0. | **yes** |
| `--no-keypoints` | | ● | off | hide the keypoint overlay | no |

That's all of them: 8 for `teleop`, 6 for `viz`. All tuning now lives in
[`configs/curl_tuning.yml`](configs/curl_tuning.yml), not on the command line —
there is no flag to point at a different file, or to bypass it.

> ⚠ **`thumb_flex: couple_low` is required.** A missing floor would mean the
> thumb closes its full travel into a `thumb_abd` parked across the palm — into
> your closing fingers — so `--map curl` now fails loudly at startup instead: a
> missing `configs/curl_tuning.yml`, or one missing that key, raises a
> `ValueError` naming the file and the key. The shipped file sets
> `thumb_flex.couple_low: 30` and `index.couple_low: 50`, identical for both
> hands. Still confirm the resolved values in a dry run before `--live`.

Thumb←index coupling under the `bButton` lock is always on, in both `teleop` and
`viz` — there is no flag to disable it. `viz` isn't a full stand-in for `teleop`
even so: it has no `--live`, `--speed`, or `--interface`.

## Reading the console

`bButton` (UDCap only) toggles the grasp lock, in dry-run too. Locked channels
keep displaying the *tracked* value, not the frozen one being sent, and are
named in the trailing bracket:

```
thumb_flex=100.0  thumb_abd= 47.1  index= 12.0  middle= 88.4 ...  [PARKED thumb_abd  COUPLED thumb_flex<-index  FLOOR index=20]
```

Every `--map curl` run also prints what it resolved, e.g. `thumb-index couple: ON
(thumb_flex couple_low=30, index floor=50, hand=left)`. Read that, not the file:
`couple_low` can come from a per-side section, so which value applies depends on
`--hand`. A side-only `couple_low` can no longer leave the other hand unfloored —
that side raises at startup instead — but the printed line is still the only place
you see the numbers this run actually landed on.

## Safety

`--live` and `demo.py` are the only things in this repo that move the hand.
`tools/bring_up_hand.py` only changes network interfaces — it brings the CAN
link up, it does not command the L6. `tools/probe_udp.py` never opens the L6
either. Clear the area, start at a low `--speed`. There is a 3-2-1 countdown,
and **Ctrl-C leaves the hand holding its last commanded pose** — no watchdog
opens it. Dry runs, `viz` and `offline_check` never open the L6.

## Cheat sheet

These are the UDCap forms (`--map curl`). For Wuji, drop `--map curl` and add
`--glove wuji` instead — the default `retarget` is what's validated there (see
[Choose your glove](#choose-your-glove)).

| goal | command |
| ---- | ------- |
| plumbing | `uv run python -m prehensile.offline_check` |
| check the glove stream (UDCap) | `uv run python tools/probe_udp.py --seconds 2` |
| dry-run, curl map | `uv run python -m prehensile.teleop --map curl` |
| sim preview | `uv run python -m prehensile.viz --map curl` |
| self-test ⚠ | `uv run python demo.py --speed 30` |
| **live teleop** ⚠ | `uv run python -m prehensile.teleop --map curl --live --speed 25` |
| live, right hand ⚠ | `… --map curl --hand right --live --speed 25` |

## Troubleshooting

| symptom | cause | fix |
| ------- | ----- | --- |
| `probe_udp` says `NO DATA`. | HandDriver transmission off, or aimed at the wrong IP/port. | Redo the settings above. `ss -ulpn 'sport = :5555'` shows if something else took the port. |
| `CAN/L6 error:` on a `--live` run. | The interface is missing or down — usually `bring_up_hand.py` has not run since the last reboot or replug. | `ip -details link show hand_l` (must be UP at 1 Mbps), then re-run the bring-up. |
| Index/middle fingers stay open under `--map retarget`. | Known: the optimizer is degenerate for UDCap ([README](README.md#angle-mapping)). | Use `--map curl`. |

## Tuning

Two files. `configs/real_hand_{left,right}.yml` here affects **`--map retarget`
only**: `scaling_factor` (1.15) if a fist does not reach ~0, `low_pass_alpha`
(0.2) up if motion lags, down if it jitters. Everything for `--map curl` — which
is also what the ROS node uses — is in
[`configs/curl_tuning.yml`](configs/curl_tuning.yml), self-documented in its
header comments.

## Checks

```bash
uv run python -m prehensile.offline_check        # synthetic kp -> 6 angles
```

If `offline_check` fails, the environment is broken — stop there, it is not the
glove.
