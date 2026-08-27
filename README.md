# Anvil-Prehensile

An end-to-end teleoperation stack for dexterous manipulation, bridging the
**UDCap**/**Wuji** glove and the **REALHAND L6** hand. Both sides are normalized
to the same `(21, 3)` MediaPipe-order keypoint frame; everything downstream of
that seam shares one code path.

Below are some demos of Anvil-Prehensile.

Driving the REALHAND L6 from each supported glove:

<table>
  <tr>
    <td width="50%"><img src="docs/img/udcap_glove_demo.gif" width="100%" alt=""></td>
    <td width="50%"><img src="docs/img/wuji_glove_demo.gif" width="100%" alt=""></td>
  </tr>
  <tr align="center">
    <td><b>UDCap glove demo</b></td>
    <td><b>Wuji glove demo</b></td>
  </tr>
</table>

Anvil-Prehensile integrated with our OpenArm system:

<table>
  <tr>
    <td width="33%"><img src="docs/img/IMG_1197.gif" width="100%" alt=""></td>
    <td width="33%"><img src="docs/img/IMG_1199.gif" width="100%" alt=""></td>
    <td width="33%"><img src="docs/img/IMG_1205.gif" width="100%" alt=""></td>
  </tr>
  <tr align="center">
    <td><b>Logistics packing</b></td>
    <td><b>Cube inserting</b></td>
    <td><b>Lego stacking</b></td>
  </tr>
  <tr align="center">
    <td>Open the delivery box, put items inside, then close it.</td>
    <td>Insert one cube into the other.</td>
    <td>Stack Lego bricks on top of each other.</td>
  </tr>
</table>

## Install packages & offline check

```bash
uv sync
uv run python -m prehensile.offline_check   # glove-free environment check
```

If `offline_check` fails, the environment is broken — stop there, it is not the
glove.

## Bring-up
### 1 · Glove — pick one

#### **UDCap Data Glove**

Download **HandDriver**
([download](https://drive.google.com/drive/folders/1g4cA-hjEnTmQNcY6zyK04Bo9R_qfPcOC),
[guide](https://udexreal.gitbook.io/udexreal-docs/robotics/usage-for-udexreal-robotics-products#id-3.-udexreal-robotics-teleoperation-system)).
Under **Config → Data Trans**, set Data Transmission **ON**, Format **Quater**,
FPS **120**, Target `127.0.0.1:5555`:

![HandDriver data transmission settings](docs/img/udcap-data-transmission.png)

After following the UDCap guide and **seeing the hands move in HandDriver**,
run the command below to check that the glove is connected to the computer.
```bash
uv run python tools/probe_udp.py
```

#### **Wuji Glove**

Install **Wuji Studio** and calibrate the glove
([install](https://docs.wuji.tech/docs/en/wuji-studio/latest/installation/), [calibration guide](https://docs.wuji.tech/docs/en/wuji-studio/latest/calibration/)).
After calibrating, **make sure the hands move on the Wuji Studio visualization
page**, then run the command below to check that the glove is connected to the
computer.
```bash
uv run python tools/probe_wuji.py
```
Note: **Close Wuji Studio before running prehensile teleop** — only one session may talk to the glove at a time.

### 2 · Hand (REALHAND L6)

Plug the USB adapter into the computer, then run the script below to set up the udev rule:

```bash
sudo python3 tools/bring_up_hand.py   # Re-run this after every replug.
```

Every tool here defaults to a canonical interface name (`hand_l`/`hand_r`)
rather than probing, and udev gives hand adapters no name — so those names exist
only after this runs, and not across a reboot or replug.

### 3 · Anvil-Prehensile commands

After choosing the glove and hand and making sure they are connected to the
computer, use the basic Anvil-Prehensile commands below. They fall into two
groups, viz (MuJoCo preview) and teleop:

```bash
# viz
uv run python -m prehensile.viz --hand right --glove wuji --map curl # Open a MuJoCo preview showing the 21 keypoints and the L6 URDF
```
The command above uses the right hand, the Wuji glove, and curl mapping. Use
`-h` to see more flags.
```bash
# teleop
uv run python -m prehensile.teleop --hand left --glove udcap --map retarget --speed 25 # Only prints the joint angles in the terminal; the hand does NOT move.
uv run python -m prehensile.teleop --hand left --glove udcap --map retarget --speed 25 --live # Adding --live MOVES THE HAND!
```
The commands above use the left hand, the UDCap glove, a speed of 25, and
retarget mapping. Use `-h` to see more flags.

### Special setting for UDCap

`bButton` (UDCap only) toggles the grasp lock, in dry-run too. Locked channels
keep displaying the *tracked* value, not the frozen one being sent, and are
named in the trailing bracket:

```
thumb_flex=100.0  thumb_abd= 47.1  index= 12.0  middle= 88.4 ...  [PARKED thumb_abd  COUPLED thumb_flex<-index  FLOOR index=20]
```

## Tuning

Two files. `configs/real_hand_{left,right}.yml` affects **`--map retarget`
only**: raise `scaling_factor` (1.15) if a fist does not reach ~0, `low_pass_alpha`
(0.2) up if motion lags, down if it jitters. Everything for `--map curl` is in
[`prehensile/configs/curl_tuning.yml`](prehensile/configs/curl_tuning.yml), self-documented in its
header comments.

## Technical

### Pipeline

```mermaid
flowchart LR
    W(["Wuji glove<br/>wuji.py"])
    U(["UDCap glove<br/>udcap.py"])
    KP["shared seam<br/>(21,3) MediaPipe<br/>keypoints<br/>wrist-local, meters"]
    C["--map curl (default)<br/>curlmap.py<br/>geometric"]
    R["--map retarget<br/>retarget.py<br/>optimizer → qpos<br/>→ command.py"]
    A["6 angles<br/>0–100 each"]
    T(["teleop --live<br/>realhand → CAN"])
    V(["viz<br/>MuJoCo, no hardware"])

    W --> KP
    U --> KP
    KP --> C
    KP --> R
    C --> A
    R --> A
    A --> T
    A --> V

    classDef seam stroke-width:3px
    class KP seam
```

The curl + UDCap path also runs in the workcell as the ROS `hand_teleop_node`,
which imports `prehensile.*` only.

### Angle mapping

- **`--map curl`** (default) — bypasses the optimizer: the four fingers use the
  scale-invariant chord ratio `|tip − wrist| / Σ(bone lengths)`; thumb flexion is
  the MCP+IP bend angle; thumb abduction is the metacarpal's elevation above the
  palm plane.
- **`--map retarget`** — the `dex_retargeting` vector optimizer:
  keypoints → qpos → 6 × `[0,100]`.

Why curl is the default: the optimizer is **degenerate for UDCap** — index and
middle pinned open, ring and thumb unstable — because UDCap *reconstructs* its
keypoints from a bespoke FK in a frame the optimizer was never tuned for. Wuji is
fine under either, because its SDK emits correctly-framed keypoints directly.
Curl is purely geometric on wrist-local distances, so it works for both gloves.

**Thumb←index coupling** (UDCap only in practice) — enabled unconditionally, but
it only takes effect *while the glove's `bButton` grasp lock is engaged*. Under
the lock, `thumb_flex` stops using its own metric and follows the index, rescaled
onto `[couple_low, 100]` — a proportional pinch. Unlocked, both channels track
independently. Wuji exposes no `bButton`, so it never engages there.

### Per-glove open/close sense

Hardware-confirmed constants in `prehensile/profiles.py`; callers never set them
by hand.

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
