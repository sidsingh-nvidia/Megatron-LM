"""One row per level of a ramp_agents.sh run: the server's step mix plus the harness's latencies.

Step mix: the step_mix.jsonl windows of the level minus the first --skip (cache warmup),
summed per step class before dividing, so the long windows weigh more than short ones.
Latencies and throughput: the harness's "Full run" summary in harness.log (all calls of the
level, including the warmup minute).

    python3 inference-bench/server/ramp_summary.py <ramp dir> [--skip 1]
"""

import argparse
import glob
import json
import os
import re

PREFILL_CLASSES = ("prefill_only", "mixed")


def step_mix(path, skip):
    rows = [json.loads(line) for line in open(path)][skip:]
    totals = {"P": dict(steps=0.0, tokens=0.0, wall_ms=0.0),
              "D": dict(steps=0.0, tokens=0.0, wall_ms=0.0)}
    cached = []
    for r in rows:
        for step_class, d in r["step_mix"].items():
            group = totals["P" if step_class in PREFILL_CLASSES else "D"]
            for f in group:
                group[f] += d[f]
        rate = r["cache_hit_rate"]
        if "prompt_tokens_cached" not in r:
            rate = rate / (1 + rate)  # rows from before the fix: cached / computed
        cached.append(rate)
    p, d = totals["P"], totals["D"]
    steps, wall = p["steps"] + d["steps"], p["wall_ms"] + d["wall_ms"]
    nan = float("nan")
    return {
        "windows": len(rows),
        "running": sum(r["running"] for r in rows) / len(rows) if rows else nan,
        "P_steps": p["steps"], "D_steps": d["steps"],
        "P_step_pct": 100 * p["steps"] / steps if steps else nan,
        "P_time_pct": 100 * p["wall_ms"] / wall if wall else nan,
        "P_tok": p["tokens"] / p["steps"] if p["steps"] else nan,
        "D_tok": d["tokens"] / d["steps"] if d["steps"] else nan,
        "P_ms": p["wall_ms"] / p["steps"] if p["steps"] else nan,
        "D_ms": d["wall_ms"] / d["steps"] if d["steps"] else nan,
        "step_tpot": wall / steps if steps else nan,
        "hit_pct": 100 * sum(cached) / len(cached) if cached else nan,
    }


def harness(path):
    """{metric: (mean, p50, p90)} and throughput from the harness's Full run summary."""
    text = open(path).read()
    if "=== Full run ===" not in text:
        return {}
    full = text.split("=== Full run ===", 1)[1].split("=== Throughput by", 1)[0]
    out = {}
    m = re.search(r"Throughput\s*:\s*([\d.]+)", full)
    if m:
        out["tok_s"] = float(m.group(1))
    for name in ("TTFT", "TPOT"):
        m = re.search(rf"^{name} \(ms\)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)",
                      full, re.M)
        if m:
            out[name] = tuple(float(x) for x in m.groups())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ramp_dir")
    ap.add_argument("--skip", type=int, default=1, help="leading windows to drop per level")
    args = ap.parse_args()

    levels = sorted(glob.glob(os.path.join(args.ramp_dir, "w*")),
                    key=lambda p: int(os.path.basename(p)[1:]))
    print(f"{'agents':>6} {'run':>5} {'P steps':>8} {'D steps':>8} {'P %st':>6} {'P %time':>7} "
          f"{'P tok/st':>8} {'D tok/st':>8} {'P ms':>6} {'D ms':>5} {'stepTPOT':>8} {'hit %':>6} "
          f"{'TTFT p50':>8} {'TTFT p90':>8} {'TPOT p50':>8} {'TPOT p90':>8} {'tok/s':>7}")
    for level in levels:
        s = step_mix(os.path.join(level, "step_mix.jsonl"), args.skip)
        h = harness(os.path.join(level, "harness.log"))
        nan = (float("nan"),) * 3
        ttft, tpot = h.get("TTFT", nan), h.get("TPOT", nan)
        print(f"{os.path.basename(level)[1:]:>6} {s['running']:>5.0f} {s['P_steps']:>8.0f} "
              f"{s['D_steps']:>8.0f} {s['P_step_pct']:>6.1f} {s['P_time_pct']:>7.1f} "
              f"{s['P_tok']:>8.0f} {s['D_tok']:>8.1f} {s['P_ms']:>6.1f} {s['D_ms']:>5.1f} "
              f"{s['step_tpot']:>8.2f} {s['hit_pct']:>6.1f} {ttft[1]:>8.0f} {ttft[2]:>8.0f} "
              f"{tpot[1]:>8.1f} {tpot[2]:>8.1f} {h.get('tok_s', float('nan')):>7.0f}")


if __name__ == "__main__":
    main()
