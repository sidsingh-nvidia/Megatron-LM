#!/bin/bash
# Agent-count ramp against one mcore server: for each level in LEVELS, run the OpenSandbox
# SWE-bench harness with that many agents for DURATION while step_mix_watch.py records the
# server's step mix once a minute. Levels run back to back in one allocation; the harness stops
# its agents and kills their sandboxes at the end of each level.
#
# Per level, in $OUT/w<N>/:
#   harness.log      the harness output (per-minute TTFT / TPOT / ITL buckets, final summary)
#   step_mix.jsonl   one line per minute (see step_mix_watch.py), also printed in step_mix.log
#
# Required env: SERVER (host:port), OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY.
# Optional: LEVELS (default "25 50 100 150 200"), DURATION (default 10m), HARNESS (the
# live_client directory), PYTHON, OUT.
set -uo pipefail

SERVER="${SERVER:?pass SERVER=<ip>:<port>}"
LEVELS="${LEVELS:-25 50 100 150 200}"
DURATION="${DURATION:-10m}"
HARNESS="${HARNESS:-/home/sidsingh/opensandbox/inference-bench/agentic/live_client}"
PYTHON="${PYTHON:-/home/sidsingh/opensandbox/.venv/bin/python}"
WATCH="$(dirname "$(realpath "$0")")/step_mix_watch.py"
OUT="${OUT:-$HARNESS/runs/ramp_${SLURM_JOB_ID:-local}_${SERVER%%:*}}"
mkdir -p "$OUT"
echo "ramp: server=$SERVER levels=[$LEVELS] duration=$DURATION out=$OUT node=$(hostname)"

for N in $LEVELS; do
    dir="$OUT/w$N"
    mkdir -p "$dir"
    echo "$(date +%T) level w$N: start"
    "$PYTHON" -u "$WATCH" --server "http://$SERVER" --interval 60 \
        --out "$dir/step_mix.jsonl" > "$dir/step_mix.log" 2>&1 &
    watch_pid=$!
    WORKERS=$N PROCS=$(( (N + 6) / 7 )) DURATION=$DURATION RUNS="$dir" \
        bash "$HARNESS/bench_live_opensandbox.sbatch" > "$dir/harness.log" 2>&1
    echo "$(date +%T) level w$N: harness exited $?"
    kill "$watch_pid" 2>/dev/null
    wait "$watch_pid" 2>/dev/null
done
echo "$(date +%T) ramp done"
