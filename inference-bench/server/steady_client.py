"""Steady-state closed-loop load: throughput and interactivity at fixed concurrency.

At each concurrency level c, c workers each keep one chat request in flight on the math blend
(AIME 2025 + HMMT Feb 2025 + BeyondAIME, see rollouts/collect_rollouts.py), sending the next
problem as soon as the previous reply arrives. No artificial output lengths: the model stops at
its own EOS, with server-default sampling, so long-running requests are part of the load. By
default there is no output cap (the server allows up to its max sequence length); --max-tokens
sets one.

Levels ramp up (c sorted ascending) without draining: going from c1 to c2 adds c2 - c1 workers,
and the in-flight requests carry over into the next level's steady state. Per level:
  - warm up for --warmup seconds, then measure for --window seconds;
  - throughput: the server's /metrics generation-token counter (summed over engines), read at
    the start and end of the window; exact to an engine step;
  - interactivity: per request, end-to-end latency / output tokens (TPOT including TTFT, since
    the user waits for the first token too) over the requests that finished inside the window;
    reported as TPOT percentiles and tokens/s/user = 1000 / TPOT. Unlike InferenceX, TTFT is
    not subtracted. The p90 / p99 TPOT (slowest requests) are the low end of tokens/s/user.

    python3 inference-bench/server/steady_client.py --server http://<host>:5000 \
        --concurrency 4 8 16 --out <exp dir>/steady.jsonl
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rollouts"))
from collect_rollouts import PROMPT_SUFFIX, load_rows  # noqa: E402


def read_metrics(server):
    """Summed (unlabeled) values of the server's mcore:* metrics."""
    with urllib.request.urlopen(f"{server}/metrics", timeout=120) as resp:
        text = resp.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("mcore:") and "{" not in line:
            name, value = line.split()
            out[name[len("mcore:"):]] = float(value)
    return out


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", required=True, help="total in-flight requests per level")
    ap.add_argument("--warmup", type=float, default=60.0, help="seconds before measuring each level")
    ap.add_argument("--window", type=float, default=300.0, help="measured seconds per level")
    ap.add_argument("--max-tokens", type=int, default=0, help="output cap; 0 = none (natural EOS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows("hard_math")
    random.Random(args.seed).shuffle(rows)
    lock = threading.Lock()
    next_row = [0]
    completed = []  # (start, end, completion_tokens, prompt_tokens, finish_reason)
    errors = [0]
    consecutive_errors = [0]
    stop = threading.Event()
    # This many request errors in a row with no success means the server is gone: stop instead
    # of hammering it (a dead server refuses connections instantly).
    max_consecutive_errors = 50

    def worker():
        while not stop.is_set():
            with lock:
                row = rows[next_row[0] % len(rows)]
                next_row[0] += 1
            payload = {"messages": [{"role": "user", "content": row["problem"] + PROMPT_SUFFIX}]}
            if args.max_tokens > 0:
                payload["max_tokens"] = args.max_tokens
            req = urllib.request.Request(f"{args.server}/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=4 * 3600) as resp:
                    body = json.loads(resp.read())
                usage = body.get("usage", {})
                rec = (start, time.time(), usage.get("completion_tokens", 0),
                       usage.get("prompt_tokens", 0), body["choices"][0].get("finish_reason"))
                with lock:
                    completed.append(rec)
                    consecutive_errors[0] = 0
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                    consecutive_errors[0] += 1
                    if consecutive_errors[0] >= max_consecutive_errors:
                        stop.set()
                print(f"request error: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(1)

    # Fail fast if the server is not up (e.g. its job ended before this client started).
    try:
        urllib.request.urlopen(f"{args.server}/health", timeout=30).close()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"server {args.server} is not reachable: {type(e).__name__}: {e}")
    print(f"{len(rows)} problems; levels {sorted(args.concurrency)}; warmup {args.warmup:.0f} s, "
          f"window {args.window:.0f} s, max_tokens {args.max_tokens}", flush=True)
    threads = []
    out = open(args.out, "a")
    for c in sorted(args.concurrency):
        while len(threads) < c:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
        level_start = time.time()
        print(f"{level_start:.3f} level c={c}: warming up", flush=True)
        if stop.wait(args.warmup):
            break
        m0, t0 = read_metrics(args.server), time.time()
        if stop.wait(args.window):
            break
        m1, t1 = read_metrics(args.server), time.time()
        dt = t1 - t0
        with lock:
            done = [r for r in completed if t0 <= r[1] <= t1]
        tpot_ms = [1000.0 * (r[1] - r[0]) / r[2] for r in done if r[2] > 0]
        rec = {
            "concurrency": c,
            "window_s": round(dt, 2),
            "output_tok_s": (m1["generation_tokens_total"] - m0["generation_tokens_total"]) / dt,
            "prompt_tok_s": (m1["prompt_tokens_total"] - m0["prompt_tokens_total"]) / dt,
            "cached_prompt_tok_s": (m1["prompt_tokens_cached_total"] - m0["prompt_tokens_cached_total"]) / dt,
            "requests_finished": int(m1["requests_finished_total"] - m0["requests_finished_total"]),
            "engine_steps": int(m1["engine_steps_total"] - m0["engine_steps_total"]),
            "running_at_end": int(m1["requests_running"]),
            "waiting_at_end": int(m1["requests_waiting"]),
            "client_finished": len(done),
            "tpot_ms_p50": percentile(tpot_ms, 0.5),
            "tpot_ms_mean": sum(tpot_ms) / len(tpot_ms) if tpot_ms else float("nan"),
            "tpot_ms_p90": percentile(tpot_ms, 0.9),
            "tpot_ms_p99": percentile(tpot_ms, 0.99),
            "output_tokens_p50": percentile([r[2] for r in done], 0.5),
            "truncated": sum(r[4] == "length" for r in done),
            "errors": errors[0],
        }
        # Tokens/s/user = 1000 / TPOT, InferenceX style (median, and the slow tail at p90).
        rec["user_tok_s_p50"] = 1000.0 / rec["tpot_ms_p50"]
        rec["user_tok_s_p90_slow"] = 1000.0 / rec["tpot_ms_p90"]
        # Cross-check only: equals the per-user rate only if every worker is always decoding.
        rec["output_tok_s_over_c"] = rec["output_tok_s"] / c
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"c={c:5d}  output {rec['output_tok_s']:9.1f} tok/s  "
              f"interactivity p50 {rec['user_tok_s_p50']:6.1f} tok/s/user "
              f"(TPOT p50 {rec['tpot_ms_p50']:.2f} ms, p90 {rec['tpot_ms_p90']:.2f} ms; "
              f"{len(done)} finished)  "
              f"running {rec['running_at_end']} waiting {rec['waiting_at_end']}  "
              f"truncated {rec['truncated']}  errors {errors[0]}", flush=True)
    failed = stop.is_set()
    stop.set()
    out.close()
    if failed:
        sys.exit(f"stopped after {max_consecutive_errors} consecutive request errors; is the server up?")
    print("done", flush=True)


if __name__ == "__main__":
    main()
