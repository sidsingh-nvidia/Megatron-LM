# moe-bench

Single-node tooling for benchmarking and profiling the mcore dynamic inference server on
Nemotron MoE models, built to compare the MoE expert-parallel dispatchers (`nvls`, `nvl_a2a`).

- `server/launch.sh`: one sbatch script that serves nano-3.5 or Super v3 on a 4-GPU node and,
  optionally, runs a throughput sweep or an nsys profile inside the same job.
- `server/client.py`: the in-job sweep client (stdlib HTTP plus HF `datasets` / `transformers`).
- `convert/convert_ultra_tp8ep8.sh`: Megatron Bridge HF -> mcore conversion of Nemotron 3 Ultra
  (2 nodes, TP8 x EP8).

Run everything from the Megatron-LM repo root. Every knob and its default is documented at the
top of `server/launch.sh`.

## Setup

Cluster-specific settings (slurm account, partition and QoS; container images and mounts; the
HF cache; checkpoint paths) are not in the scripts. Copy `cluster.env.example` to
`moe-bench/cluster.env` (git-ignored), fill it in, and source it in the shell you submit from:

```bash
cp moe-bench/cluster.env.example moe-bench/cluster.env   # then edit
source moe-bench/cluster.env
```

sbatch reads the `SBATCH_*` variables from the environment; the scripts also source the file
themselves (or `$MOE_BENCH_CONFIG`) for the paths. The examples below assume it is sourced.

## Serving

The server listens on port 5000 (OpenAI API at `/v1`) until the 4 h time limit. The host is
printed in the slurm log and written to `<exp dir>/host`.

```bash
sbatch --export=ALL,EXP_NAME=a2a,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16 moe-bench/server/launch.sh
# Super v3, for evals: prefix caching, and chat responses without prompt token ids
sbatch --export=ALL,EXP_NAME=eval,MODEL=superv3,PREFIX_CACHING=1,EVAL_MODE=1 moe-bench/server/launch.sh
```

## Throughput sweeps

When `BENCH` is set, rank 0 runs `server/client.py` once the server is ready and the job exits
when the sweep finishes. Results go to `<exp dir>/bench.log` and `bench.json`.

```bash
# decode-heavy: short GSM8K prompts, 1024 output tokens, batch sizes 1-128
sbatch --export=ALL,EXP_NAME=decode,BENCH="1 2 4 8 16 32 64 128" moe-bench/server/launch.sh

# prefill-heavy, batched: 512-1024-token ShareGPT prompts, 1 output token
sbatch --export=ALL,EXP_NAME=prefill,BENCH="1 8 32 128",BENCH_DATASET=sharegpt,BENCH_MIN_ISL=512,BENCH_OSL=1 moe-bench/server/launch.sh

# prefill, one request per step: exact prompt lengths, CUDA graphs up to 8k tokens
sbatch --export=ALL,EXP_NAME=prefill1,MODEL=superv3,CUDA_GRAPH_MAX_TOKENS=8192,LOG_INTERVAL=1,BENCH=1,BENCH_DATASET=sharegpt,BENCH_EXACT_ISLS="1024 2048 4096 8000",BENCH_OSL=1,BENCH_WARMUP=2,BENCH_ITERS=5 moe-bench/server/launch.sh
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
sbatch --export=ALL,EXP_NAME=trace,MODEL=superv3,DISPATCHER=nvl_a2a,COMBINE_PRECISION=bf16,CUDA_GRAPH_MAX_TOKENS=8192,PROFILE=1,LOG_INTERVAL=1,BENCH=1,BENCH_DATASET=sharegpt,BENCH_EXACT_ISLS=4096,BENCH_OSL=1,BENCH_WARMUP=2,BENCH_ITERS=2 moe-bench/server/launch.sh
```

## Output

Each run writes `moe-bench/experiments/<timestamp>_<EXP_NAME>/` with `config.env` (knobs and
the full server command), `server_rank*.log`, the `launch.sh` copy the ranks ran (so editing the
script cannot affect running jobs), and `bench.*` / `trace_rank*.nsys-rep` when enabled. Run
outputs are git-ignored.
