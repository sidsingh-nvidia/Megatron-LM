# inference-bench

Tooling for benchmarking and profiling the mcore dynamic inference server on
Nemotron MoE models, built to compare the MoE expert-parallel dispatchers (`nvls`, `nvl_a2a`).

- `server/launch.sh`: one sbatch script that serves nano-3.5 or Super v3 on one 4-GPU node (Ultra v3
  on two) and, optionally, runs a throughput sweep or an nsys profile inside the same job.
- `server/client.py`: the in-job sweep client (stdlib HTTP plus HF `datasets` / `transformers`).
- `convert/convert_ultra_tp8ep8.sh`: Megatron Bridge HF -> mcore conversion of Nemotron 3 Ultra
  (2 nodes, TP8 x EP8).

Run everything from the Megatron-LM repo root. Every knob and its default is documented at the
top of `server/launch.sh`.

## Setup

Cluster-specific settings (slurm account, partition and QoS; container images and mounts; the
HF cache; checkpoint paths) are not in the scripts. Copy `cluster.env.example` to
`inference-bench/cluster.env` (git-ignored), fill it in, and source it in the shell you submit from:

```bash
cp inference-bench/cluster.env.example inference-bench/cluster.env   # then edit
source inference-bench/cluster.env
```

sbatch reads the `SBATCH_*` variables from the environment; the scripts also source the file
themselves (or `$INFERENCE_BENCH_CONFIG`) for the paths. The examples below assume it is sourced.

## Run configs

`configs/` holds the settings of the paper runs, one `KEY=value` file each; `run.sh` submits the
server and its client sweep from one (the client waits for the server, sweeps, then shuts it down):

```bash
inference-bench/run.sh inference-bench/configs/ultrav3_wide_ep.env
inference-bench/run.sh inference-bench/configs/superv3_narrow_ep.env GEMM_BACKEND=vllm   # override a knob
```

| Config | Nodes | Layout | Experts / GPU | Concurrency per GPU |
|---|---|---|---|---|
| `superv3_narrow_ep` | 1 | TP4 x EP4 | 128 | 1-32 |
| `superv3_wide_ep` | 16 | TP4 x EP64 | 8 | 1-32 |
| `ultrav3_narrow_ep` | 2 | TP8 x EP8 | 64 | 1-16 |
| `ultrav3_wide_ep` | 16 | TP8 x EP64 | 8 | 1-16 |

All four use `nvls` with bf16 combine, the TRT-LLM BF16 routed MoE, `MAMBA_RATIO=0.4`, GSM8K
prompts and 8192 output tokens (one timed batch after a warmup). Besides the knobs below, a config
sets `NODES` (multi-node servers also get `--segment NODES`), and optionally `QOS` and `TIME`.

## Serving

The server listens on port 5000 (OpenAI API at `/v1`) until the 4 h time limit. The host is
printed in the slurm log and written to `<exp dir>/host`.

```bash
sbatch --export=ALL,EXP_NAME=a2a,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16 inference-bench/server/launch.sh
# Super v3, for evals: prefix caching, and chat responses without prompt token ids
sbatch --export=ALL,EXP_NAME=eval,MODEL=superv3,PREFIX_CACHING=1,EVAL_MODE=1 inference-bench/server/launch.sh
# Ultra v3 (~1.1 TB of weights) on 2 nodes, TP8 x EP8; --segment keeps both nodes in one NVLink
# domain, which the cross-node TP8 multicast collectives need
sbatch -N 2 --segment=2 --export=ALL,EXP_NAME=ultra,MODEL=ultrav3 inference-bench/server/launch.sh
```

Multi-node runs launch one task per GPU on every node (`sbatch -N <n>`); global rank 0 hosts the
HTTP server, and ranks rendezvous through the `MASTER_ADDR` / `MASTER_PORT` the cluster exports.

## Knobs

All knobs are environment variables passed with `sbatch --export=ALL,KNOB=value,...`.

| Knob | Values / meaning | Default |
|---|---|---|
| `EXP_NAME` | run label (experiment directory suffix) | `run` |
| `MODEL` | `nanov3p5` \| `superv3` \| `ultrav3` | `nanov3p5` |
| `TP` | tensor parallel size; data parallel = world / TP | 4 (`ultrav3`: 8) |
| `ETP` | expert tensor parallel size; EP = world / ETP | 1 |
| `DISPATCHER` | MoE EP dispatcher: `nccl` \| `nvls` \| `nvl_a2a` | `nvls` |
| `COMBINE_PRECISION` | MoE combine buffer: `fp32` \| `bf16` (`nvl_a2a` needs `bf16`) | `fp32` |
| `ASYNC` | 1 = async scheduling | 1 |
| `PREFIX_CACHING` | 1 = prefix caching with LRU eviction | 0 |
| `SAMPLING` | sampling backend: `flashinfer` \| `torch` | `flashinfer` |
| `MAX_TOKENS` | max tokens per engine step (chunked-prefill budget) | 8192 |
| `MAX_REQUESTS` | max concurrent requests; unset = derived from the buffer | unset |
| `CUDA_GRAPH_MAX_TOKENS` | largest token count captured in CUDA graphs; set to `MAX_TOKENS` to graph prefill steps | server default (512) |
| `BUFFER_GB` | dynamic-batching KV/state buffer per GPU | 50 (`ultrav3`: 25) |
| `MAMBA_PREFIX_GB` | Mamba prefix-cache buffer per GPU (with `PREFIX_CACHING=1`) | 50 (`ultrav3`: 25) |
| `EVAL_MODE` | 1 = chat responses do not echo prompt token ids | 1 |
| `PROMPT_LOGPROBS` | 1 = materialize logits for every prompt token, for prompt log-probs (`echo` + `logprobs`); costs memory. With 0, requests asking for prompt log-probs fail | 0 |
| `LOG_INTERVAL` | engine step-log interval; 1 = log every step | 1000 |
| `BENCH` | batch sizes for the in-job client; unset = serve until the time limit | unset |
| `BENCH_DATASET` | `gsm8k` \| `sharegpt` | `gsm8k` |
| `BENCH_MIN_ISL` | ShareGPT only: minimum prompt tokens | 0 |
| `BENCH_ISL` | join consecutive prompts to ~this many tokens | 0 (off) |
| `BENCH_EXACT_ISLS` | exact prompt lengths in tokens, e.g. `"1024 2048"` | unset |
| `BENCH_OSL` / `BENCH_ITERS` / `BENCH_WARMUP` | output tokens / timed / warmup iterations | 1024 / 3 / 1 |
| `PROFILE` | 1 = nsys capture around the first batch size's timed iterations | 0 |
| `INFERENCE_BENCH_CONFIG` | path to the cluster config | `inference-bench/cluster.env` |

## Ultra v3 recipes

Ultra v3 (~1.1 TB of bf16 weights) needs 2 nodes. At TP8 x EP8 the weights take ~140 GB of each
GPU's ~184 GiB, so memory is tight:

- With `PREFIX_CACHING=1`, use `BUFFER_GB=20,MAMBA_PREFIX_GB=20` (25/25 runs out of memory
  allocating the Mamba prefix cache), and keep `MAX_TOKENS=CUDA_GRAPH_MAX_TOKENS=2048`.
- `PROMPT_LOGPROBS=0` (the default) is required: materializing every prompt token's logits
  runs the TP8 logits all-gather out of memory.
- At `BUFFER_GB=25` the engine admits at most 48 concurrent requests; larger batches queue and
  run in waves, so decode sweeps above batch 48 do not measure a single batch.

```bash
# serve (evals): prefix caching on
sbatch -N 2 --segment=2 --export=ALL,EXP_NAME=ultra_eval,MODEL=ultrav3,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16,MAX_TOKENS=2048,CUDA_GRAPH_MAX_TOKENS=2048,PREFIX_CACHING=1,BUFFER_GB=20,MAMBA_PREFIX_GB=20 inference-bench/server/launch.sh

# prefill sweep: one request per step, exact prompt lengths up to the 2048-token step budget
sbatch -N 2 --segment=2 --export=ALL,EXP_NAME=ultra_prefill,MODEL=ultrav3,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16,MAX_TOKENS=2048,CUDA_GRAPH_MAX_TOKENS=2048,LOG_INTERVAL=1,BENCH=1,BENCH_DATASET=sharegpt,BENCH_EXACT_ISLS="512 1024 2048",BENCH_OSL=1,BENCH_WARMUP=2,BENCH_ITERS=5 inference-bench/server/launch.sh

# decode sweep: GSM8K prompts, 1024 output tokens
sbatch -N 2 --segment=2 --export=ALL,EXP_NAME=ultra_decode,MODEL=ultrav3,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16,MAX_TOKENS=2048,CUDA_GRAPH_MAX_TOKENS=2048,BENCH="1 2 4 8 16 32",BENCH_OSL=1024 inference-bench/server/launch.sh
```

## Ultra v3 trace log

nsys traces of Ultra v3 decode, each adding one change. All: 16 GB200 nodes, EP64, `nvls` dispatch
unless noted, bf16 combine, `BUFFER_GB=80`, `MAMBA_RATIO=0.4`, `LOG_INTERVAL=1000`, eval mode,
client on its own node (`server/client.sh`), 128 GSM8K requests x 128 output tokens (1 warmup +
1 profiled batch). Step period = median over the 128 decode steps of rank 4 (rank 44 agrees to
0.01 ms), stalls excluded, from `analysis/decode_steps.py`; wall = the profiled batch's client wall
time, which also includes prefill and stalls. Run directories are under `experiments/`; details in
`REPORT.md`.

| Run directory | Code | TP | Change | Step period | vs previous | Stalls | Wall |
|---|---|---|---|---|---|---|---|
| `20261014_153243_trace_ultrav3_ep64_nvl_a2a_bs128` | `065e74b66` | 4 | baseline, `nvl_a2a` dispatcher | 14.41 ms | - | 0 | 3.44 s |
| `20261014_153245_trace_ultrav3_ep64_nvls_bs128` | `065e74b66` | 4 | baseline, `nvls` dispatcher | **14.07 ms** | - | 0 | 3.29 s |
| `20261015_134511_trace_ultrav3_tp8_ep64_nvls_bs128_rmsnorm` | `59c9a7583` | 8 | TP8 + FlashInfer RMSNorm | **12.52 ms** | -1.55 ms (-11%) | 8 (487 ms) | 3.81 s |
| `20261015_143831_trace_ultrav3_tp8_ep64_nvls_bs128_pr8114` | `ca3b432f1` | 8 | + #8114 router and alignment kernels (reverted) | 14.15 ms | +1.63 ms (regression) | 4 (222 ms) | 3.88 s |
| `20261015_152634_trace_ultrav3_tp8_ep64_nvls_bs128_tltopk` | `d2ec3f55e` | 8 | + fused `tl.topk` router (instead of #8114) | **12.00 ms** | -0.52 ms (-4.2%) vs `134511` | 5 (327 ms) | 3.62 s |
| `20261016_101531_trace_ultrav3_tp8_ep64_nvls_bs128_trtllm` | `0345cd18e` (pre-dispatch pack not yet) | 8 | + TRT-LLM BF16 routed MoE (`GEMM_BACKEND=trtllm_bf16_routed`, EP token-sum consensus; torch routing pack) | **10.65 ms** | -1.35 ms (-11.2%) | - | 2.67 s |
| `20261016_115006_trace_ultrav3_tp8_ep64_nvls_bs128_trtllm_packed` | `0345cd18e` | 8 | + routing packed before the dispatch (one Triton kernel, also masks padding) + 2-tensor AGV | **10.06 ms** | -0.59 ms (-5.5%) | - | 3.63 s |

Current best: **10.06 ms** per decode step at concurrency 128 (-28.5% vs the TP4 `nvls` baseline).

To analyze a new trace: export two ranks to SQLite (`nsys stats --force-export=true --report
cuda_gpu_kern_sum --format csv -o <run>/rank<r> <run>/trace_rank<r>.nsys-rep`), then run
`analysis/decode_steps.py <label>=<run> <label>=<previous run>` (per-step breakdown and deltas) and
`analysis/gemm_bandwidth.py <run>` (achieved bandwidth per GEMM).

## Throughput sweeps

When `BENCH` is set, rank 0 runs `server/client.py` once the server is ready and the job exits
when the sweep finishes. Results go to `<exp dir>/bench.log` and `bench.json`.

```bash
# decode-heavy: short GSM8K prompts, 1024 output tokens, batch sizes 1-128
sbatch --export=ALL,EXP_NAME=decode,BENCH="1 2 4 8 16 32 64 128" inference-bench/server/launch.sh

# prefill-heavy, batched: 512-1024-token ShareGPT prompts, 1 output token
sbatch --export=ALL,EXP_NAME=prefill,BENCH="1 8 32 128",BENCH_DATASET=sharegpt,BENCH_MIN_ISL=512,BENCH_OSL=1 inference-bench/server/launch.sh

# prefill, one request per step: exact prompt lengths, CUDA graphs up to 8k tokens
sbatch --export=ALL,EXP_NAME=prefill1,MODEL=superv3,CUDA_GRAPH_MAX_TOKENS=8192,LOG_INTERVAL=1,BENCH=1,BENCH_DATASET=sharegpt,BENCH_EXACT_ISLS="1024 2048 4096 8000",BENCH_OSL=1,BENCH_WARMUP=2,BENCH_ITERS=5 inference-bench/server/launch.sh
```

Notes:

- Concurrent requests can be split across engine steps when they arrive a few milliseconds
  apart, which confounds per-step comparisons. For clean prefill numbers use `BENCH=1` with
  `BENCH_EXACT_ISLS` (prompts are sent as exact token-id lists, so each request is one step as
  long as it fits in the 8192-token step budget), and read step times from the engine log with
  `LOG_INTERVAL=1` rather than client wall times.
- Eager prefill is CPU-launch-bound and lets ranks drift apart, which shows up as time spent in
  the next TP reduce-scatter's barrier. `CUDA_GRAPH_MAX_TOKENS=8192` graphs prefill steps too.
- The ShareGPT dataset is read from `sharegpt_filtered_benchmark.json` at the repo root
  (entries `[prompt, prompt_tokens, output_tokens]`); it is not checked in.

## Profiling

`PROFILE=1` runs every rank under nsys with `--capture-range=cudaProfilerApi`. The client
brackets the timed iterations of the first batch size / prompt length with `/start_profile` and
`/stop_profile`, after which the server exits and each rank writes
`<exp dir>/trace_rank<r>.nsys-rep`. The first request after `start_profile` can include a
one-time spike, so profile at least two iterations.

```bash
sbatch --export=ALL,EXP_NAME=trace,MODEL=superv3,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16,CUDA_GRAPH_MAX_TOKENS=8192,PROFILE=1,LOG_INTERVAL=1,BENCH=1,BENCH_DATASET=sharegpt,BENCH_EXACT_ISLS=4096,BENCH_OSL=1,BENCH_WARMUP=2,BENCH_ITERS=2 inference-bench/server/launch.sh
```

## Output

Each run writes `inference-bench/experiments/<timestamp>_<EXP_NAME>/` with `config.env` (knobs and
the full server command), `server_rank*.log`, the `launch.sh` copy the ranks ran (so editing the
script cannot affect running jobs), and `bench.*` / `trace_rank*.nsys-rep` when enabled. Run
outputs are git-ignored.
