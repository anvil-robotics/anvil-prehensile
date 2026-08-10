#!/usr/bin/env bash
# Convenience launcher: on UDCap, sanity-check the HandDriver UDP stream first;
# on Wuji there's no UDP stream to check, so that probe is skipped. Either way,
# then start teleop.
#
# Usage:
#   scripts/teleop.sh                              # dry-run, UDCap (default glove)
#   scripts/teleop.sh --live --speed 25             # drive the L6 (clear the area!)
#   scripts/teleop.sh --glove wuji                  # dry-run, Wuji (close Wuji Studio first)
#
# Any extra args are passed straight through to `prehensile.teleop`.
set -euo pipefail
cd "$(dirname "$0")/.."

glove="udcap"
expect_glove=0
for arg in "$@"; do
    if [ "$expect_glove" -eq 1 ]; then
        glove="$arg"
        expect_glove=0
        continue
    fi
    case "$arg" in
        --glove) expect_glove=1 ;;
        --glove=*) glove="${arg#--glove=}" ;;
    esac
done

if [ "$glove" = "udcap" ]; then
    echo "== Probing HandDriver UDP stream (2s) =="
    if ! uv run python tools/probe_udp.py --seconds 2; then
        echo >&2
        echo "No usable stream on :5555. In HandDriver enable Data Transmission" >&2
        echo "(Format=Quater, target 127.0.0.1:5555, FPS 120) and retry." >&2
        exit 1
    fi
else
    echo "== Skipping UDP probe: $glove does not stream over UDP =="
    if [ "$glove" = "wuji" ]; then
        echo "Make sure Wuji Studio is closed first -- only one session may talk to the glove."
    fi
fi

echo
echo "== Launching teleop =="
exec uv run python -m prehensile.teleop "$@"
