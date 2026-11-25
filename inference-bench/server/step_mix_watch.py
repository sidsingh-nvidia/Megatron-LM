"""Watch an mcore server's step mix while an external load runs against it.

Every --interval seconds, reads /metrics and prints one row for the window just ended:
  - running / waiting requests (gauges at the end of the window);
  - prefill steps (prefill_only + mixed) vs decode steps, as counts and shares of steps;
  - share of engine wall time in prefill vs decode steps;
  - mean input tokens and mean time of a prefill step and of a decode step;
  - step TPOT: engine wall time / steps (every running request gets one token per step,
    prefill or decode, so this is the mean inter-token time the engine delivers);
  - generated tokens/s and the prefix-cache hit rate (cached / (cached + computed) prompt tokens).
Counters are summed over engines, so with several engines the step counts are engine-steps.
TTFT and client-side TPOT come from the load harness, not from here.

Each window is also appended as a JSON line to --out. Run from the login node:
    python3 inference-bench/server/step_mix_watch.py --server http://<host>:5000 \
        --interval 60 --out step_mix.jsonl
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steady_client import diff_step_mix, read_metrics  # noqa: E402

PREFILL_CLASSES = ("prefill_only", "mixed")


def summarize(m0, m1, seconds):
    """One window's stats from two read_metrics snapshots `seconds` apart."""
    mix = diff_step_mix(m0["step_mix"], m1["step_mix"]).get("all", {})
    prefill = [mix[c] for c in PREFILL_CLASSES if c in mix]
    decode = [d for c, d in mix.items() if c not in PREFILL_CLASSES]

    def total(group, field):
        return sum(d[field] for d in group)

    steps = total(prefill, "steps") + total(decode, "steps")
    wall = total(prefill, "wall_ms") + total(decode, "wall_ms")
    # mcore's prompt_tokens_total counts only the computed (uncached) prompt tokens.
    computed = m1["prompt_tokens_total"] - m0["prompt_tokens_total"]
    cached = m1["prompt_tokens_cached_total"] - m0["prompt_tokens_cached_total"]

    def per_step(group, field):
        n = total(group, "steps")
        return total(group, field) / n if n else float("nan")

    return {
        "time": time.strftime("%H:%M:%S"),
        "seconds": seconds,
        "running": m1["requests_running"],
        "waiting": m1["requests_waiting"],
        "prefill_steps": total(prefill, "steps"),
        "decode_steps": total(decode, "steps"),
        "prefill_step_share": total(prefill, "steps") / steps if steps else float("nan"),
        "prefill_time_share": total(prefill, "wall_ms") / wall if wall else float("nan"),
        "prefill_tokens_per_step": per_step(prefill, "tokens"),
        "decode_tokens_per_step": per_step(decode, "tokens"),
        "prefill_step_ms": per_step(prefill, "wall_ms"),
        "decode_step_ms": per_step(decode, "wall_ms"),
        "step_tpot_ms": wall / steps if steps else float("nan"),
        "gen_tok_s": (m1["generation_tokens_total"] - m0["generation_tokens_total"]) / seconds,
        "prompt_tokens_computed": computed,
        "prompt_tokens_cached": cached,
        "cache_hit_rate": cached / (cached + computed) if cached + computed else float("nan"),
        "step_mix": mix,
    }


HEADER = (f"{'time':>8} {'run':>5} {'wait':>5} {'P steps':>8} {'D steps':>8} {'P %st':>6} "
          f"{'P %time':>7} {'P tok/st':>9} {'D tok/st':>9} {'P ms':>7} {'D ms':>6} "
          f"{'TPOT ms':>8} {'gen tok/s':>9} {'hit %':>6}")


def row(s):
    return (f"{s['time']:>8} {s['running']:>5.0f} {s['waiting']:>5.0f} {s['prefill_steps']:>8.0f} "
            f"{s['decode_steps']:>8.0f} {100 * s['prefill_step_share']:>6.1f} "
            f"{100 * s['prefill_time_share']:>7.1f} {s['prefill_tokens_per_step']:>9.0f} "
            f"{s['decode_tokens_per_step']:>9.1f} {s['prefill_step_ms']:>7.1f} "
            f"{s['decode_step_ms']:>6.1f} {s['step_tpot_ms']:>8.2f} {s['gen_tok_s']:>9.0f} "
            f"{100 * s['cache_hit_rate']:>6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="e.g. http://10.109.22.89:5000")
    ap.add_argument("--interval", type=float, default=60.0, help="window length in seconds")
    ap.add_argument("--out", help="append each window as a JSON line here")
    args = ap.parse_args()

    print(HEADER, flush=True)
    m0, t0 = read_metrics(args.server), time.monotonic()
    while True:
        time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
        m1, t1 = read_metrics(args.server), time.monotonic()
        s = summarize(m0, m1, t1 - t0)
        print(row(s), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(s) + "\n")
        m0, t0 = m1, t1


if __name__ == "__main__":
    main()
