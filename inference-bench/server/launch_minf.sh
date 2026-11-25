#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#
# Serve Nemotron nano-3.5, Super v3 or Ultra v3 with the mcore dynamic inference server
# (4 GPUs per node). nano-3.5 and Super v3 fit on one node; Ultra v3 needs `sbatch -N 2`.
# Multi-node jobs must sit in one NVLink (NVL72) domain for the NVLS / symmetric-memory dispatch:
# pass `--segment <N>` with `-N <N>` (otherwise Slurm may split the nodes across racks and the
# symmetric-memory init fails with "CUDA driver error: invalid resource handle").
#
# Cluster-specific settings (slurm account/partition/QoS, container image, checkpoint and HF
# cache paths) come from inference-bench/cluster.env; copy cluster.env.example and fill it in.
#
# Run from the Megatron-LM repo root:
#   source inference-bench/cluster.env
#   sbatch inference-bench/server/launch_minf.sh
#   sbatch --export=ALL,EXP_NAME=nccl,DISPATCHER=nccl inference-bench/server/launch_minf.sh
#
# The server is examples/inference/launch_inference_server.py (the native OpenAI server), for
# aggregated and disaggregated (PD=1) runs alike.
#
# Knobs (environment variables):
#   EXP_NAME           run label                                    (default: run)
#   MODEL              nanov3p5 | superv3 | ultrav3                 (default: nanov3p5)
#   TP                 tensor parallel size; data parallel = world / TP   (default: 4; ultrav3: 8)
#   ETP                expert tensor parallel size                      (default: 1)
#   EP                 expert parallel size; experts are replicated over world / (EP x ETP)
#                      expert-data-parallel copies, so EP < world / ETP keeps a copy per group of
#                      EP x ETP GPUs (narrow EP)                         (default: world / ETP)
#   BUFFER_GB          dynamic-batching KV/state buffer per GPU   (default: 50; ultrav3: 25)
#   PREFILL_BUFFER_GB  with PD=1: the prefill pool's buffer (BUFFER_GB is then the decode pool's)
#   MAMBA_RATIO        fraction of BUFFER_GB for live Mamba states; with ~100 MB of state per
#                      request on Ultra, this sets the concurrent-request cap   (default: 0.1)
#   MAMBA_PREFIX_GB    Mamba prefix-cache buffer per GPU, with PREFIX_CACHING=1
#                      (default: 50; ultrav3: 25)
#   DISPATCHER         MoE EP dispatcher: nccl | nvls | nvl_a2a | deepep   (default: nvls)
#   COMBINE_PRECISION  MoE combine precision: fp32 | bf16 (bf16 for nvl_a2a / deepep)   (default: fp32)
#   GEMM_BACKEND       routed-expert grouped GEMM: vllm | flashinfer | torch | trtllm_bf16_routed
#                      (default: vllm)
#   ASYNC              1 = async scheduling                         (default: 1)
#   PREFIX_CACHING     1 = prefix caching, LRU eviction             (default: 1)
#   SAMPLING           sampling backend: flashinfer | torch         (default: flashinfer)
#   MAX_REQUESTS       --inference-dynamic-batching-max-requests    (default: unset)
#   MAX_TOKENS         max tokens per engine step (chunked-prefill budget)  (default: 8192)
#   CUDA_GRAPH_MAX_TOKENS  largest token count captured in CUDA graphs, so prefill / mixed steps up
#                      to it are graphed too (default: 2048; the server's own default is 512)
#   LOG_INTERVAL       engine step-log interval; 1 = log every step  (default: 1000)
#   FP8_RECIPE         mxfp8 = MXFP8 weights, quantized from the BF16 checkpoint at load
#                      (--fp8-format e4m3 --fp8-recipe mxfp8 --fp8-param-gather, which
#                      inference_optimized requires); unset = BF16          (default: unset)
#   SHARED_EXPERT_OVERLAP  1 = run the shared experts on a side stream, overlapped with the
#                      routed experts; 0 = in order                         (default: 1)
#   EVAL_MODE          1 = --eval-mode: chat responses skip echoing prompt token IDs, which
#                      speeds up evals with long contexts             (default: 1)
#   PROMPT_LOGPROBS    1 = materialize logits for every prompt token so /v1/completions can
#                      return prompt log-probs (echo + logprobs); costs memory (default: 0)
#   BENCH              batch sizes for the in-job client, e.g. "1 16 64". When set, the job
#                      runs the sweep and exits; when unset, it serves until the time limit or
#                      until <exp dir>/done appears. The in-job client shares rank 0's CPUs with
#                      the server; for clean timings run the sweep from server/client.sh instead.
#   BENCH_DATASET      gsm8k (~100-token questions) | sharegpt (repo-root
#                      sharegpt_filtered_benchmark.json, <= 1024 tokens)   (default: gsm8k)
#   BENCH_MIN_ISL      sharegpt only: keep prompts with >= this many tokens (default: 0)
#   BENCH_ISL          join consecutive prompts to ~this many tokens (default: 0 = off)
#   BENCH_EXACT_ISLS   prompt lengths in exact tokens, e.g. "1024 4096 8000" (default: unset).
#                      With BENCH=1 each request is a single prefill step (keep <= 8192).
#   BENCH_OSL / BENCH_ITERS / BENCH_WARMUP   output tokens / timed / warmup iterations
#                                             (default: 1024 / 3 / 1)
#   PD                 1 = prefill/decode disaggregation (main's native PD): one torch.distributed
#                      world over the whole job, split by --inference-shards into a prefill
#                      engine on the first PREFILL_NODES nodes and the decode engines on the rest;
#                      one coordinator routes each request prefill -> decode, and global rank 0
#                      (on the prefill node) hosts the HTTP frontend. KV and Mamba state move over
#                      NIXL. Prefix caching is forced on (the hand-off needs it) and the Mamba
#                      prefix cache off (decode must not have one). NODES / TP / EP / ETP size the
#                      decode engines; PREFILL_TP / PREFILL_EP / PREFILL_ETP the prefill engine
#                      (default: TP / all prefill GPUs / ETP). Other knobs apply to both pools.
#                                                                 (default: 0 = aggregated)
#   PREFILL_NODES      with PD=1: prefill-pool nodes                         (default: 1)
#   DONE_FILE          file that stops the server when it appears; one watcher per job checks it
#                      every 10 s (default: <exp dir>/done)
#   EXP_DIR            experiment directory (default: a new timestamped one)
#   PROFILE            1 = run every rank under nsys (capture range = cudaProfilerApi) and have the
#                      client wrap the timed iterations of the first BENCH batch size in
#                      /start_profile + /stop_profile; the server exits when profiling stops.
#                      Traces: <exp dir>/trace_rank<r>.nsys-rep            (default: 0)
#   PROFILE_ALL        with PROFILE=1: warm every BENCH batch size first, then profile the timed
#                      iterations of ALL batch sizes in one capture              (default: 0)
#   BENCH_PAUSE        seconds idle between batch sizes, to tell them apart in a trace (default: 0)
#   Prefill-heavy runs: BENCH_OSL=1 with BENCH_DATASET=sharegpt BENCH_MIN_ISL=512, or BENCH_ISL=2048.
#
# Output: inference-bench/experiments/<timestamp>_<EXP_NAME>/ with config.env, server_rank*.log and,
# with BENCH, bench.log + bench.json. The server listens on port 5000 (OpenAI API at /v1).

SCRIPT=inference-bench/server/launch_minf.sh

# ── Outer: inside sbatch, launch one task per GPU in the container ─────────────
if [ -z "${INFERENCE_BENCH_INNER:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
    # Exported settings reach the ranks through srun's environment.
    source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
    export EXP_DIR="${EXP_DIR:-inference-bench/experiments/$(date +%Y%m%d_%H%M%S)_${EXP_NAME:-run}}"
    export DONE_FILE="${DONE_FILE:-${EXP_DIR}/done}"
    mkdir -p "${EXP_DIR}"
    echo "Experiment directory: ${EXP_DIR}"
    # Ranks run a private copy: bash reads scripts incrementally, so editing the original
    # while a job is running would corrupt that job's ranks.
    cp "${SCRIPT}" "${EXP_DIR}/launch_minf.sh"
    export INFERENCE_BENCH_INNER=1
    NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
    if [ "${MODEL:-nanov3p5}" == "ultrav3" ] && [ "${NUM_NODES}" -lt 2 ]; then
        echo "MODEL=ultrav3 needs at least 2 nodes (sbatch -N 2)"
        exit 1
    fi
    # COLOCATE_CLIENT=1 (set by run.sh): run client.sh in this job, as an overlapping step on its
    # last node (away from rank 0's frontend and coordinator), instead of a separate client job.
    # The client gets the sweep as CLIENT_BENCH, since BENCH would make rank 0 run client.py.
    CLIENT_PID=
    if [ "${COLOCATE_CLIENT:-0}" == "1" ]; then
        CLIENT_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tail -1)
        cp inference-bench/server/client.sh "${EXP_DIR}/client.sh"
        BENCH="${CLIENT_BENCH:?COLOCATE_CLIENT=1 needs CLIENT_BENCH}" SERVER_JOB="${SLURM_JOB_ID}" \
            CLIENT_EXP_DIR="${EXP_DIR}" \
            srun --overlap -N 1 -n 1 -w "${CLIENT_NODE}" \
            --container-image "${INFERENCE_BENCH_SERVE_IMAGE}" \
            --container-mounts "${INFERENCE_BENCH_MOUNTS}" \
            --container-workdir="${PWD}" bash "${EXP_DIR}/client.sh" > "${EXP_DIR}/client.out" 2>&1 &
        CLIENT_PID=$!
        echo "Client colocated on ${CLIENT_NODE}; output in ${EXP_DIR}/client.out"
    fi
    # One task per GPU on every node; global rank 0 hosts the HTTP server.
    srun -l -N "${NUM_NODES}" -n $((4 * NUM_NODES)) --ntasks-per-node 4 --cpus-per-task 36 \
        --container-image "${INFERENCE_BENCH_SERVE_IMAGE:?set in cluster.env}" \
        --container-name "inference_bench_${SLURM_JOB_ID}" \
        --container-mounts "${INFERENCE_BENCH_MOUNTS:?set in cluster.env}" \
        --container-workdir="${PWD}" bash "${EXP_DIR}/launch_minf.sh" &
    SRUN_PID=$!
    # One watcher per job, not per rank: when DONE_FILE appears (the in-job sweep or a colocated
    # client finished), SIGTERM every rank with scancel --signal (which signals the tasks without
    # cancelling the job; signalling srun itself would SIGKILL them before their EXIT traps copy
    # the logs). A separate client job does the same. A rank whose server dies exits non-zero, and
    # KillOnBadExit ends the step. Under PROFILE the servers exit on their own once profiling
    # stops; a SIGTERM could cut off nsys while it writes the reports.
    while kill -0 "${SRUN_PID}" 2>/dev/null; do
        if [ "${PROFILE:-0}" != "1" ] && [ -f "${DONE_FILE}" ]; then
            scancel --signal=TERM "${SLURM_JOB_ID}"
            break
        fi
        sleep 10
    done
    wait "${SRUN_PID}"
    status=$?
    # Stopped on DONE_FILE by this watcher or a client: the normal end, unless a server failed.
    [ -f "${DONE_FILE}" ] && [ ! -f "${EXP_DIR}/server_failed" ] && status=0
    # If the server stopped first (e.g. a startup failure), touch DONE_FILE so a client still
    # waiting exits too.
    if [ -n "${CLIENT_PID}" ]; then
        touch "${DONE_FILE}"
        wait "${CLIENT_PID}"
    fi
    exit ${status}
fi

# ── Inner: one rank ────────────────────────────────────────────────────────────
SERVER=examples/inference/launch_inference_server.py
export RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} WORLD_SIZE=${SLURM_NTASKS}
# Multi-node runs use the rendezvous address the cluster exports; fall back to localhost.
export MASTER_ADDR=${MASTER_ADDR:-localhost} MASTER_PORT=${MASTER_PORT:-29500}
DONE_FILE=${DONE_FILE:-${EXP_DIR}/done}
JOB_NODES=${SLURM_JOB_NUM_NODES:-1}
# Decode (or aggregated) nodes; with PD=1 the first PREFILL_NODES nodes are the prefill pool.
POOL_NODES=${JOB_NODES}
[ "${PD:-0}" == "1" ] && POOL_NODES=$(( JOB_NODES - ${PREFILL_NODES:-1} ))
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Per-job node-local scratch (cleared after the job) for compile caches, temp files and the live
# log; nothing there is needed once the job ends.
# (In the container $USER is root; SLURM_JOB_USER is the submitting user.)
SCRATCH=${INFERENCE_BENCH_SCRATCH:-/raid/scratch}/${SLURM_JOB_USER:-${USER}}/${SLURM_JOB_ID:-local}
export TMPDIR=${SCRATCH}/tmp_rank${RANK}
mkdir -p "${TMPDIR}"
export TRITON_CACHE_DIR=${SCRATCH}/triton_cache_rank${RANK}
export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/inductor_cache_rank${RANK}
# Unbuffered, so a rank stopped by SIGTERM loses none of its log.
export PYTHONUNBUFFERED=1

if [ "${PD:-0}" == "1" ]; then
    # The KV hand-off needs prefix caching; math prompts share no prefixes, so the Mamba prefix
    # cache would only take memory (and the decode pool must not have one).
    PREFIX_CACHING=1
    MAMBA_PREFIX_GB=0
    # The cluster sets UCX_TLS=tcp, which keeps NIXL off the GPU transports (cuda_ipc, cuda_copy,
    # RDMA); let UCX choose.
    unset UCX_TLS
    # NIXL from a pip --target directory when the image has none.
    [ -n "${NIXL_PYTHONPATH:-}" ] && export PYTHONPATH="${NIXL_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
    # `import torch` loads HPC-X's UCX (through UCC); NIXL bundles its own, and with two copies in
    # one process GPU-memory registration fails. Preload NIXL's libucs so a single copy serves both.
    NIXL_UCS=$(python -c 'import glob, importlib.util, os
spec = importlib.util.find_spec("nixl_cu13")
root = os.path.dirname(os.path.dirname(spec.origin)) if spec else ""
print(next(iter(sorted(glob.glob(os.path.join(root, "nixl_cu13.libs", "libucs-*.so*")))), ""))')
    if [ -z "${NIXL_UCS}" ]; then
        echo "PD=1 needs NIXL (nixl + nixl-cu13); use the +nixl image or set NIXL_PYTHONPATH"
        touch "${EXP_DIR}/server_failed" "${DONE_FILE}"
        exit 1
    fi
    export LD_PRELOAD="${NIXL_UCS}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# Ultra v3 runs TP8 x EP8 on 2 nodes; the others default to TP4.
if [ "${MODEL:-nanov3p5}" == "ultrav3" ]; then TP=${TP:-8}; else TP=${TP:-4}; fi
ETP=${ETP:-1}
POOL_GPUS=$((4 * POOL_NODES))
EP=${EP:-$((POOL_GPUS / ETP))}
if (( POOL_GPUS % (EP * ETP) || POOL_GPUS % TP )); then
    echo "EP x ETP = $((EP * ETP)) and TP = ${TP} must divide the ${POOL_GPUS} (decode) GPUs"
    exit 1
fi
SHARD_ARGS=(--tensor-model-parallel-size "${TP}" --expert-model-parallel-size "${EP}"
            --expert-tensor-parallel-size "${ETP}")
if [ "${PD:-0}" == "1" ]; then
    PREFILL_GPUS=$((4 * ${PREFILL_NODES:-1}))
    PREFILL_TP=${PREFILL_TP:-${TP}}
    PREFILL_ETP=${PREFILL_ETP:-${ETP}}
    PREFILL_EP=${PREFILL_EP:-$((PREFILL_GPUS / PREFILL_ETP))}
    if (( PREFILL_GPUS % PREFILL_TP || PREFILL_GPUS % (PREFILL_EP * PREFILL_ETP) )); then
        echo "PREFILL_TP and PREFILL_EP x PREFILL_ETP must divide the ${PREFILL_GPUS} prefill GPUs"
        exit 1
    fi
    # Each shard carries its own parallelism; the prefill shard takes the first global ranks.
    SHARDS="tp=${PREFILL_TP},dp=$((PREFILL_GPUS / PREFILL_TP)),ep=${PREFILL_EP},expt_tp=${PREFILL_ETP},role=prefill"
    SHARDS+="+tp=${TP},dp=$((POOL_GPUS / TP)),ep=${EP},expt_tp=${ETP},role=decode"
    # The global parallel sizes only build process groups over the whole world (each model uses
    # its shard's groups), so they just have to tile it; the prefill shard's sizes are used.
    if (( WORLD_SIZE % PREFILL_TP || WORLD_SIZE % (PREFILL_EP * PREFILL_ETP) )); then
        echo "PREFILL_TP and PREFILL_EP x PREFILL_ETP must divide the world size ${WORLD_SIZE}"
        exit 1
    fi
    SHARD_ARGS=(--tensor-model-parallel-size "${PREFILL_TP}" --expert-model-parallel-size "${PREFILL_EP}"
                --expert-tensor-parallel-size "${PREFILL_ETP}"
                --inference-shards "${SHARDS}" --disagg-kv-transport-backend nixl)
fi

ARGS=(
    --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker
    --inference-dynamic-batching
    --inference-dynamic-batching-unified-memory-level 0
    --inference-dynamic-batching-max-tokens "${MAX_TOKENS:-8192}"
    --inference-dynamic-batching-mamba-memory-ratio "${MAMBA_RATIO:-0.1}"
    --inference-dynamic-batching-num-cuda-graphs -1
    --inference-dynamic-batching-sampling-backend "${SAMPLING:-flashinfer}"
    --inference-logging-step-interval "${LOG_INTERVAL:-1000}"
    --enable-chunked-prefill --use-flashinfer-fused-rope --sequence-parallel
    --cuda-graph-impl local --cuda-graph-scope full_iteration_inference
    --transformer-impl inference_optimized
    --inference-max-seq-length 262144 --seq-length 262144 --max-position-embeddings 262144
    # Parallelism (SHARD_ARGS) and MoE dispatch.
    "${SHARD_ARGS[@]}"
    --inference-moe-token-dispatcher-type "${DISPATCHER:-nvls}"
    --inference-moe-combine-precision "${COMBINE_PRECISION:-fp32}"
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --inference-grouped-gemm-backend "${GEMM_BACKEND:-vllm}"
    --inference-use-synchronous-zmq-collectives
    --model-provider mamba
    --dist-ckpt-strictness log_unexpected
    --tokenizer-type HuggingFaceTokenizer --no-use-tokenizer-model-from-checkpoint-args
    --parsers nemotron-v3-reasoning qwen3-coder-tool
    --mamba-inference-ssm-states-dtype fp32 --moe-router-dtype fp32 --mtp-use-repeated-layer
)

case "${MODEL:-nanov3p5}" in
nanov3p5)
    # Nemotron 3.5 Lightning 30B-A3B, the official HF release, converted with
    # convert/convert_lightning_tp4ep4.sh (the same weights launch_vllm.sh serves).
    TOKENIZER=${NANOV3P5_TOKENIZER:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
    ARGS+=(
    --load "${NANOV3P5_CHECKPOINT:-${LIGHTNING_MCORE_OUT:?set NANOV3P5_CHECKPOINT in cluster.env}}"
    --tokenizer-model "${TOKENIZER}"
    --moe-permute-fusion
    --hybrid-layer-pattern 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E'
    --vocab-size 131072 --make-vocab-size-divisible-by 128
    --hidden-size 2688 --ffn-hidden-size 1856
    --num-attention-heads 32 --group-query-attention --num-query-groups 2 --kv-channels 128
    --normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --squared-relu
    --position-embedding-type none --untie-embeddings-and-output-weights
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 64
    --num-experts 128 --moe-ffn-hidden-size 1856 --moe-router-topk 6
    --moe-shared-expert-intermediate-size 3712
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 2.5
    )
    ;;
ultrav3)
    # Nemotron 3 Ultra 550B-A55B (latent MoE), converted with convert/convert_ultra_tp8ep8.sh.
    # ~1.1 TB of bf16 weights: at TP8 x EP8 that is ~140 GB per GPU, hence the smaller buffers.
    TOKENIZER=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    DEFAULT_BUFFER_GB=25
    DEFAULT_MAMBA_PREFIX_GB=25
    ARGS+=(
    --load "${ULTRAV3_CHECKPOINT:-${ULTRA_MCORE_OUT:?set ULTRAV3_CHECKPOINT in cluster.env}}"
    --tokenizer-model "${TOKENIZER}"
    --hybrid-layer-pattern 'MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E'
    --vocab-size 131072 --make-vocab-size-divisible-by 128
    --hidden-size 8192 --ffn-hidden-size 5120
    --num-attention-heads 64 --group-query-attention --num-query-groups 2 --kv-channels 128
    --normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --squared-relu
    --position-embedding-type none --untie-embeddings-and-output-weights
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 256
    --num-experts 512 --moe-ffn-hidden-size 5120 --moe-router-topk 22 --moe-latent-size 2048
    --moe-shared-expert-intermediate-size 10240
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 5.0
    )
    ;;
superv3)
    # Nemotron 3 Super 120B-A12B (latent MoE), the official HF release converted with
    # convert/convert_superv3_tp4ep4.sh.
    TOKENIZER=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    ARGS+=(
    --load "${SUPERV3_CHECKPOINT:-${SUPERV3_MCORE_OUT:?set SUPERV3_CHECKPOINT in cluster.env}}"
    --tokenizer-model "${TOKENIZER}"
    --hybrid-layer-pattern 'MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E'
    --vocab-size 131072 --make-vocab-size-divisible-by 128
    --hidden-size 4096 --ffn-hidden-size 2688
    --num-attention-heads 32 --group-query-attention --num-query-groups 2 --kv-channels 128
    --normalization RMSNorm --norm-epsilon 1e-5 --disable-bias-linear --squared-relu
    --position-embedding-type none --untie-embeddings-and-output-weights
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 128
    --num-experts 512 --moe-ffn-hidden-size 2688 --moe-router-topk 22 --moe-latent-size 1024
    --moe-shared-expert-intermediate-size 5376
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias
    --moe-router-topk-scaling-factor 5.0
    )
    ;;
*)
    echo "Unknown MODEL=${MODEL}; expected nanov3p5, superv3 or ultrav3"
    exit 1
    ;;
esac
ARGS+=(--inference-dynamic-batching-buffer-size-gb "${BUFFER_GB:-${DEFAULT_BUFFER_GB:-50}}")
# PD: PREFILL_BUFFER_GB sizes the prefill pool's buffer; BUFFER_GB then applies to the decode pool.
[ "${PD:-0}" == "1" ] && [ -n "${PREFILL_BUFFER_GB:-}" ] && \
    ARGS+=(--inference-dynamic-batching-prefill-buffer-size-gb "${PREFILL_BUFFER_GB}")
[ "${ASYNC:-1}" == "1" ] && ARGS+=(--inference-dynamic-batching-async-sched-mode async)
[ "${PREFIX_CACHING:-1}" == "1" ] && ARGS+=(
    --inference-dynamic-batching-prefix-caching
    --inference-dynamic-batching-prefix-caching-eviction-policy lru
)
[ "${PREFIX_CACHING:-1}" == "1" ] && [ "${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}}" != "0" ] && ARGS+=(
    --inference-dynamic-batching-prefix-caching-mamba-gb "${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}}"
)
[ -n "${MAX_REQUESTS:-}" ] && ARGS+=(--inference-dynamic-batching-max-requests "${MAX_REQUESTS}")
[ "${EVAL_MODE:-1}" == "1" ] && ARGS+=(--eval-mode)
# The server returns log-probs by default, which materializes logits for every prompt token
# (gathered across TP); skip prompt log-probs so only each request's last-token logits are kept.
[ "${PROMPT_LOGPROBS:-0}" == "1" ] || ARGS+=(--skip-prompt-log-probs)
ARGS+=(--inference-cuda-graph-max-tokens "${CUDA_GRAPH_MAX_TOKENS:-2048}")
# FP8_RECIPE=mxfp8: MXFP8 expert weights (quantized from the BF16 checkpoint at load).
# --stream-ckpt-dequant: load straight into the MXFP8 params; without it the load first dequantizes
# every param to BF16 (both copies at once: OOM for Ultra at EP8).
[ "${FP8_RECIPE:-}" == "mxfp8" ] && ARGS+=(--fp8-format e4m3 --fp8-recipe mxfp8 --fp8-param-gather --stream-ckpt-dequant)
[ "${SHARED_EXPERT_OVERLAP:-1}" == "1" ] && ARGS+=(--moe-shared-expert-overlap)

# Log to node-local scratch; frequent writes to shared storage stall the rank (and, through EP,
# every rank). When the rank exits -- normally, on failure, or on SIGTERM from the job's DONE_FILE
# watcher or a client's scancel -- the EXIT trap stops the server and copies the log to EXP_DIR as
# server_rank<N>.log. To follow it live:
#   srun --jobid <id> --overlap tail -f /raid/scratch/<user>/<id>/server_rank0.log
LOG="${SCRATCH}/server_rank${RANK}.log"
# SIGTERM is the normal stop: exit 0 (after the EXIT trap below), since a non-zero exit would make
# KillOnBadExit SIGKILL the other ranks while they copy their logs.
trap 'exit 0' TERM
trap '[ -n "${SERVER_PID:-}" ] && kill -TERM -- "-${SERVER_PID}" 2>/dev/null
      cp "${LOG}" "${EXP_DIR}/server_rank${RANK}.log" 2>/dev/null' EXIT
if [ "${RANK}" == "0" ]; then
    cat > "${EXP_DIR}/config.env" <<EOF
EXP_NAME=${EXP_NAME:-run}
SLURM_JOB_ID=${SLURM_JOB_ID}
IMAGE=${INFERENCE_BENCH_SERVE_IMAGE:-}
TOKENIZER=${TOKENIZER}
MODEL=${MODEL:-nanov3p5} NODES=${POOL_NODES} TP=${TP} EP=${EP} ETP=${ETP} BUFFER_GB=${BUFFER_GB:-${DEFAULT_BUFFER_GB:-50}} MAMBA_PREFIX_GB=${MAMBA_PREFIX_GB:-${DEFAULT_MAMBA_PREFIX_GB:-50}} MAMBA_RATIO=${MAMBA_RATIO:-0.1}
DISPATCHER=${DISPATCHER:-nvls} COMBINE_PRECISION=${COMBINE_PRECISION:-fp32} GEMM_BACKEND=${GEMM_BACKEND:-vllm}
ASYNC=${ASYNC:-1} PREFIX_CACHING=${PREFIX_CACHING:-1} SAMPLING=${SAMPLING:-flashinfer} EVAL_MODE=${EVAL_MODE:-1} PROMPT_LOGPROBS=${PROMPT_LOGPROBS:-0} PROFILE=${PROFILE:-0} LOG_INTERVAL=${LOG_INTERVAL:-1000} MAX_TOKENS=${MAX_TOKENS:-8192} CUDA_GRAPH_MAX_TOKENS=${CUDA_GRAPH_MAX_TOKENS:-2048} FP8_RECIPE=${FP8_RECIPE:-} SHARED_EXPERT_OVERLAP=${SHARED_EXPERT_OVERLAP:-1}
PD=${PD:-0} PREFILL_NODES=${PREFILL_NODES:-} PREFILL_GPUS=${PREFILL_GPUS:-0} PREFILL_BUFFER_GB=${PREFILL_BUFFER_GB:-} SHARDS=${SHARDS:-} GPUS=${POOL_GPUS}
BENCH=${BENCH:-} BENCH_DATASET=${BENCH_DATASET:-gsm8k} BENCH_MIN_ISL=${BENCH_MIN_ISL:-0} BENCH_ISL=${BENCH_ISL:-0} BENCH_EXACT_ISLS=${BENCH_EXACT_ISLS:-}
BENCH_OSL=${BENCH_OSL:-1024} BENCH_ITERS=${BENCH_ITERS:-3} BENCH_WARMUP=${BENCH_WARMUP:-1}
CMD=python ${SERVER} ${ARGS[*]}
EOF
fi

PRE=()
if [ "${PROFILE:-0}" == "1" ]; then
    PRE=(nsys profile -t cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
         --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown
         --force-overwrite true -o "${EXP_DIR}/trace_rank${RANK}")
fi
# Own process group: the server spawns its HTTP frontends as separate processes, which a signal
# to the server alone would leave running.
setsid "${PRE[@]}" python "${SERVER}" "${ARGS[@]}" > "${LOG}" 2>&1 &
SERVER_PID=$!

# Serve until the server exits. Ranks do not poll for the end: the deployment is stopped by
# SIGTERM (the job's DONE_FILE watcher, or scancel from a client job), on which the EXIT trap
# stops the server. Under PROFILE the server shuts itself down when profiling stops.
serve() {
    wait "${SERVER_PID}"
    local status=$?
    [ "${PROFILE:-0}" == "1" ] && exit 0
    # The server exited on its own: a failure. Tell a waiting client, and exit non-zero so Slurm
    # stops every rank (KillOnBadExit), in PD both pools.
    touch "${EXP_DIR}/server_failed" "${DONE_FILE}"
    exit $(( status == 0 ? 1 : status ))
}

[ "${RANK}" != "0" ] && serve

# Rank 0 hosts the HTTP frontends (all replicas are listening once the server logs that it
# started them; /health then answers). Wait for it, then either run the client sweep or serve.
server_ready() {
    python3 -c 'import sys, urllib.request
try:
    sys.exit(0 if urllib.request.urlopen("http://localhost:5000/health", timeout=5).status == 200 else 1)
except Exception:
    sys.exit(1)'
}
until server_ready; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited during startup; see ${EXP_DIR}/server_rank${RANK}.log"
        touch "${EXP_DIR}/server_failed" "${DONE_FILE}"
        exit 1
    fi
    sleep 5
done
echo "Server ready."
echo "Server reachable at: http://$(hostname):5000/v1"
hostname > "${EXP_DIR}/host.tmp" && mv "${EXP_DIR}/host.tmp" "${EXP_DIR}/host"  # atomic: client.sh reads it

if [ -n "${BENCH:-}" ]; then
    python3 -u inference-bench/server/client.py \
        --server http://localhost:5000 \
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
    # The job's watcher sees this and stops every rank.
    touch "${DONE_FILE}"
fi
serve
