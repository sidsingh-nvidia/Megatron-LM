#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
# GPU partitions reject GPU-less jobs; taking all 4 also keeps the node (and its CPUs) to ourselves.
#SBATCH --gres=gpu:4
#
# Run a client from its own node against a server started by launch.sh (without BENCH), so the
# client does not share CPUs with rank 0's engine, frontend and coordinator. CLIENT=steady (the
# default) runs steady_client.py: BENCH concurrency levels of closed-loop math-blend load,
# throughput from the server's /metrics and interactivity per finished request. CLIENT=smoke runs
# the client.py GSM8K sweep (fixed BENCH_OSL with ignore_eos).
# Waits for the server, writes bench.log / bench.json into the server's experiment directory,
# then touches <exp dir>/done, which shuts the server job down.
#
# Run from the Megatron-LM repo root:
#   source inference-bench/cluster.env
#   SERVER_JOB=$(sbatch --parsable -q short -t 2:00:00 -N 16 --export=ALL,EXP_NAME=decode,MODEL=ultrav3,... inference-bench/server/launch.sh)
#   sbatch --dependency=after:${SERVER_JOB} --export=ALL,SERVER_JOB=${SERVER_JOB},BENCH="1 2 4 8" inference-bench/server/client.sh
#
# Knobs: SERVER_JOB (required: the launch.sh job id), CLIENT (steady | smoke; default steady),
# STEADY_WARMUP / STEADY_WINDOW (seconds per level; default 60 / 300), MAX_GEN_TOKENS (steady
# output cap; default 0 = none), KEEP_SERVER (1 = leave the server running
# after the sweep; default 0), PROFILE (1 = bracket the first batch size's timed iterations with
# /start_profile + /stop_profile; the server must also run with PROFILE=1) and the BENCH_* knobs of
# launch.sh (BENCH, BENCH_DATASET,
# BENCH_MIN_ISL, BENCH_ISL, BENCH_EXACT_ISLS, BENCH_OSL, BENCH_ITERS, BENCH_WARMUP).

SCRIPT=inference-bench/server/client.sh

if [ -z "${INFERENCE_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    : "${SERVER_JOB:?set SERVER_JOB to the launch.sh job id}" "${BENCH:?set BENCH, e.g. \"1 16 64\"}"
    export INFERENCE_BENCH_INNER=1
    # Private copy, as in launch.sh.
    COPY="inference-bench/experiments/.client_${SLURM_JOB_ID:-$$}.sh"
    mkdir -p inference-bench/experiments && cp "${SCRIPT}" "${COPY}"
    srun -N 1 -n 1 \
        --container-image "${INFERENCE_BENCH_SERVE_IMAGE:?set in cluster.env}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${COPY}"
    status=$?
    rm -f "${COPY}"
    exit ${status}
fi

# Find the server's experiment directory (rank 0 writes config.env at startup).
EXP_DIR=
until [ -n "${EXP_DIR}" ]; do
    EXP_DIR=$(dirname "$(grep -lx "SLURM_JOB_ID=${SERVER_JOB}" inference-bench/experiments/*/config.env 2>/dev/null | head -1)" 2>/dev/null)
    [ "${EXP_DIR}" == "." ] && EXP_DIR=
    [ -z "${EXP_DIR}" ] && sleep 10
done
echo "Server experiment directory: ${EXP_DIR}"

# launch.sh writes <exp dir>/host once the HTTP server is up; done means it already exited.
until [ -f "${EXP_DIR}/host" ]; do
    if [ -f "${EXP_DIR}/done" ]; then
        echo "Server exited before becoming ready; see ${EXP_DIR}/server_rank0.log"
        exit 1
    fi
    sleep 10
done
SERVER="http://$(cat "${EXP_DIR}/host"):5000"
TOKENIZER=$(sed -n 's/^TOKENIZER=//p' "${EXP_DIR}/config.env")
echo "Client on $(hostname), server ${SERVER}"
cat >> "${EXP_DIR}/config.env" <<EOF
CLIENT_JOB=${SLURM_JOB_ID} CLIENT_HOST=$(hostname)
BENCH=${BENCH} BENCH_DATASET=${BENCH_DATASET:-gsm8k} BENCH_MIN_ISL=${BENCH_MIN_ISL:-0} BENCH_ISL=${BENCH_ISL:-0} BENCH_EXACT_ISLS=${BENCH_EXACT_ISLS:-}
BENCH_OSL=${BENCH_OSL:-1024} BENCH_ITERS=${BENCH_ITERS:-3} BENCH_WARMUP=${BENCH_WARMUP:-1}
EOF

# One thread per in-flight request.
ulimit -u unlimited 2>/dev/null
if [ "${CLIENT:-steady}" == "steady" ]; then
    echo "CLIENT=steady STEADY_WARMUP=${STEADY_WARMUP:-60} STEADY_WINDOW=${STEADY_WINDOW:-300} MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-0}" >> "${EXP_DIR}/config.env"
    python3 -u inference-bench/server/steady_client.py \
        --server "${SERVER}" \
        --concurrency ${BENCH} \
        --warmup "${STEADY_WARMUP:-60}" \
        --window "${STEADY_WINDOW:-300}" \
        --max-tokens "${MAX_GEN_TOKENS:-0}" \
        --out "${EXP_DIR}/steady.jsonl" \
        2>&1 | tee "${EXP_DIR}/bench.log"
    status=${PIPESTATUS[0]}
else
    python3 -u inference-bench/server/client.py \
        --server "${SERVER}" \
        --batch-sizes ${BENCH} \
        --dataset "${BENCH_DATASET:-gsm8k}" \
        --min-isl "${BENCH_MIN_ISL:-0}" \
        --isl "${BENCH_ISL:-0}" \
        --exact-isls ${BENCH_EXACT_ISLS:-} --tokenizer "${TOKENIZER}" \
        --osl "${BENCH_OSL:-1024}" \
        --iters "${BENCH_ITERS:-3}" \
        --warmup "${BENCH_WARMUP:-1}" \
        --out "${EXP_DIR}/bench.json" \
        --pause "${BENCH_PAUSE:-0}" \
        $([ "${PROFILE:-0}" == "1" ] && { [ "${PROFILE_ALL:-0}" == "1" ] && echo --profile-all || echo --profile; }) \
        2>&1 | tee "${EXP_DIR}/bench.log"
    status=${PIPESTATUS[0]}
fi
[ "${KEEP_SERVER:-0}" == "1" ] || touch "${EXP_DIR}/done"
exit ${status}
