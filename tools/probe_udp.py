#!/usr/bin/env python3
"""Sniff the HandDriver UDP stream and report what is on the wire.

The vendor app (UdexReal HandDriver) unicasts glove data to 127.0.0.1:5555 at
up to 120 Hz, but the serialization is a runtime choice: it may be per-glove
JSON (top-level keys like ``DeviceName`` / ``Bones``) or binary protobuf
(``TeleopDataPose`` / ``TeleopDataQuat`` / ``TeleopDataAngle`` from
``vendor/udex_protobuf/handdriver_teleop.proto``). This tool binds the port,
listens for a few seconds, then prints rate/size stats and a per-sample verdict
so the operator can decide which capability branch to take.

Because proto3 parses foreign bytes leniently (a wrong type usually will NOT
throw), a protobuf type is only accepted here if its *content* checks pass:
  * pose  -> a present hand has 21 JointPose entries incl. a "...Wrist" name
  * quat  -> a present hand has 15 quaternion joints
  * angle -> a present hand has 23 float joints

Run (from the repo root):
    uv run python tools/probe_udp.py                 # port 5555, 5 s
    uv run python tools/probe_udp.py --port 5556 --seconds 3

Exit code 1 (with guidance) if no datagrams arrive.
"""

import argparse
import json
import socket
import sys
import time
from collections import Counter

from prehensile import udexio as _udex


def _sample_indices(n_unique, want):
    """Evenly-spread indices into a list of ``n_unique`` items (up to ``want``)."""
    if n_unique <= want:
        return list(range(n_unique))
    step = (n_unique - 1) / (want - 1)
    return sorted({int(round(i * step)) for i in range(want)})


def _distinct_payloads(packets):
    """Payloads deduped by exact bytes, in first-seen order."""
    seen = set()
    out = []
    for data in packets:
        if data not in seen:
            seen.add(data)
            out.append(data)
    return out


def _analyze_json(data):
    """Verdict string for a datagram whose first byte is ``{``."""
    try:
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        return f"looks like JSON but failed to parse ({exc}); first 64 bytes hex: {data[:64].hex()}"
    if not isinstance(obj, dict):
        return f"JSON but top level is {type(obj).__name__}, not an object"
    keys = list(obj.keys())
    device = obj.get("DeviceName")
    dev = f", DeviceName={device!r}" if device is not None else ""
    return f"JSON (top-level keys: {keys}{dev})"


def _score_pose(pb2, data):
    msg = pb2.TeleopDataPose()
    try:
        msg.ParseFromString(data)
    except Exception:
        return None
    hands = []
    for side, field in (("L", "LeftHand"), ("R", "RightHand")):
        if not msg.HasField(field):
            continue
        hand = getattr(msg, field)
        if len(hand.pose) == 21 and any("Wrist" in jp.name for jp in hand.pose):
            hands.append(side)
    if not hands:
        return None
    return ("TeleopDataPose", "21 named joint poses", hands, msg.RoleName)


def _score_quat(pb2, data):
    msg = pb2.TeleopDataQuat()
    try:
        msg.ParseFromString(data)
    except Exception:
        return None
    hands = []
    for side, field in (("L", "LeftHand"), ("R", "RightHand")):
        if msg.HasField(field) and len(getattr(msg, field).joints) == 15:
            hands.append(side)
    if not hands:
        return None
    return ("TeleopDataQuat", "15 joint quats", hands, msg.RoleName)


def _score_angle(pb2, data):
    msg = pb2.TeleopDataAngle()
    try:
        msg.ParseFromString(data)
    except Exception:
        return None
    hands = []
    for side, field in (("L", "LeftHand"), ("R", "RightHand")):
        if msg.HasField(field) and len(getattr(msg, field).joints) == 23:
            hands.append(side)
    if not hands:
        return None
    return ("TeleopDataAngle", "23 joint angle floats", hands, msg.RoleName)


def _analyze_binary(data, pb2):
    """Verdict string for a non-JSON datagram."""
    hexhead = data[:64].hex()
    if pb2 is None:
        return f"binary, but protobuf is unavailable - cannot classify; first 64 bytes hex: {hexhead}"
    # Fixed preference order (pose, quat, angle) as tie-break; the content
    # checks use distinct joint counts so real collisions are near-impossible.
    candidates = [f(pb2, data) for f in (_score_pose, _score_quat, _score_angle)]
    hits = [c for c in candidates if c is not None]
    if not hits:
        return f"UNRECOGNIZED binary (first 64 bytes hex: {hexhead})"
    # Prefer the verdict validating the most hands.
    name, detail, hands, role = max(hits, key=lambda c: len(c[2]))
    return f"protobuf {name} ({detail}, RoleName={role!r}, hands: {'+'.join(hands)})"


def _analyze(data, pb2):
    if data[:1] == b"{":
        return _analyze_json(data)
    return _analyze_binary(data, pb2)


def collect(sock, seconds):
    """Receive datagrams for ``seconds``; return (payloads, elapsed_seconds)."""
    packets = []
    start = time.monotonic()
    deadline = start + seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, _addr = sock.recvfrom(_udex.RECV_BUFSIZE)
            except socket.timeout:
                break
            packets.append(data)
    except KeyboardInterrupt:
        print("\n(interrupted - reporting what was captured so far)")
    return packets, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(
        description="Probe the HandDriver UDP stream: rate, size, JSON-vs-protobuf verdict."
    )
    parser.add_argument("--port", type=int, default=5555,
                        help="UDP port to listen on (default: 5555)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="How long to listen (default: 5.0)")
    args = parser.parse_args()

    # protobuf is optional: JSON analysis needs none. Degrade with a warning.
    try:
        pb2 = _udex.load_pb2()
    except Exception as exc:  # ImportError or protobuf runtime errors
        pb2 = None
        print(f"warning: protobuf types unavailable ({exc}); "
              f"binary payloads will not be classified. "
              f"Regenerate the pb2 files if you need this.", file=sys.stderr)

    sock = _udex.bind_udp(args.port)
    print(f"probe_udp: listening on 0.0.0.0:{args.port} for {args.seconds:g}s ...")
    try:
        packets, elapsed = collect(sock, args.seconds)
    finally:
        sock.close()

    if not packets:
        print(f"NO DATA - no datagrams on port {args.port} in {args.seconds:g}s.\n"
              f"Is HandDriver Data Transmission enabled and targeting this host:{args.port}?")
        return 1

    sizes = [len(p) for p in packets]
    rate = len(packets) / elapsed if elapsed > 0 else float("nan")
    print(f"packets: {len(packets)}")
    print(f"rate: {rate:.1f} Hz (over {elapsed:.2f}s)")
    print(f"size bytes: min={min(sizes)} avg={sum(sizes) / len(sizes):.1f} max={max(sizes)}")

    unique = _distinct_payloads(packets)
    idxs = _sample_indices(len(unique), 3)
    print(f"payload analysis ({len(idxs)} distinct sample(s) of {len(unique)} unique / "
          f"{len(packets)} total):")
    verdicts = []
    for n, i in enumerate(idxs, 1):
        sample = unique[i]
        verdict = _analyze(sample, pb2)
        verdicts.append(verdict)
        print(f"  sample {n} (len {len(sample)}): verdict: {verdict}")

    # A one-line consensus to make the GATE decision easy to eyeball.
    common = Counter(verdicts).most_common(1)[0][0]
    print(f"summary: {common}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
