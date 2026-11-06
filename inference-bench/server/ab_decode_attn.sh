#!/bin/bash
# A/B of the decode attention backend on one node, run as one task per GPU inside the serving
# container (e.g. a container job with 4 tasks per node). For each backend in DECODE_ATTN_LIST,
# every task runs one server rank of launch_minf.sh (its inner mode); rank 0 waits for the server,
# runs sanity.py (greedy checkable answers) and a short steady_client level, then touches
# <exp dir>/done, which shuts every rank down before the next backend starts.
#
# Knobs (environment): MODEL (superv3), TP (4), EP (4), BENCH concurrency (128), STEADY_DATASET
# (wildchat), STEADY_WINDOWS (2), STEADY_WINDOW (300 s), DECODE_ATTN_LIST ("fa4 flashinfer"),
# MCORE_FA4_MIN_NUM_SPLITS (8, applies to the fa4 leg). Output: inference-bench/experiments/
# <job id>_<backend>_ab/ with server_rank*.log, sanity.log, steady.jsonl.

set -u
cd /home/sidsingh/megatron-lm
source inference-bench/cluster.env
export MODEL=${MODEL:-superv3} TP=${TP:-4} EP=${EP:-4} ETP=1 BUFFER_GB=${BUFFER_GB:-50}
export MAMBA_RATIO=${MAMBA_RATIO:-0.4} PREFIX_CACHING=0 COMBINE_PRECISION=bf16 LOG_INTERVAL=19
export MCORE_FA4_MIN_NUM_SPLITS=${MCORE_FA4_MIN_NUM_SPLITS:-8}
export INFERENCE_BENCH_INNER=1

leg=0
for backend in ${DECODE_ATTN_LIST:-fa4 flashinfer}; do
    export DECODE_ATTN=${backend} EXP_NAME=${backend}_ab MASTER_PORT=$((29500 + 10 * leg))
    export EXP_DIR="inference-bench/experiments/${SLURM_JOB_ID}_${backend}_ab"
    mkdir -p "${EXP_DIR}"
    # launch_minf.sh's rank 0 runs client.py itself when BENCH is set; the clients here run below.
    BENCH= bash inference-bench/server/launch_minf.sh &
    server=$!
    if [ "${SLURM_PROCID}" == "0" ]; then
        until [ -f "${EXP_DIR}/host" ] || [ -f "${EXP_DIR}/done" ]; do sleep 5; done
        if [ -f "${EXP_DIR}/host" ]; then
            python3 -u inference-bench/server/sanity.py --server http://localhost:5000 --batch 16 \
                > "${EXP_DIR}/sanity.log" 2>&1
            python3 -u inference-bench/server/steady_client.py --server http://localhost:5000 \
                --concurrency "${BENCH:-128}" --dataset "${STEADY_DATASET:-wildchat}" \
                --windows "${STEADY_WINDOWS:-2}" --window "${STEADY_WINDOW:-300}" --warmup 60 \
                --out "${EXP_DIR}/steady.jsonl" > "${EXP_DIR}/bench.log" 2>&1
        fi
        touch "${EXP_DIR}/done"
    fi
    wait "${server}"
    leg=$((leg + 1))
done
