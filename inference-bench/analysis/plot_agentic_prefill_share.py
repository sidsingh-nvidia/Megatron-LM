"""Agentic load: how prefill steps take over the engine as the number of agents grows.

Reads the per-level outputs of server/ramp_agents.sh (step_mix.jsonl from step_mix_watch.py, and
the harness's harness.log) for one or more models and writes:

  prefill_share.png   (a) share of engine steps that are prefill steps, (b) share of engine time
                      in them, (c) time and (d) input tokens per prefill step, per model (e, f)
                      step TPOT split into its decode-step and prefill-step parts with the
                      client's p50 TPOT on top, (g) output tok/s, (h) prefill step time against
                      its tokens
  summary.csv         every number behind the plot, per model and level

Per level, the first minute (agents starting, cache cold) and the drain minutes at the end (fewer
than half of the level's peak requests running) are dropped; the rest are summed per step class
before dividing. Step TPOT is engine wall time / steps: every running request gets one token per
step, prefill or decode, so it splits into wall time in decode steps / steps plus wall time in
prefill steps / steps.

    python3 inference-bench/analysis/plot_agentic_prefill_share.py \\
        --ramp "Nano 3.5 (30B-A3B)=<ramp dir>" --ramp "Super 3 (120B-A12B)=<ramp dir>" \\
        --out inference-bench/analysis/agentic_prefill
"""

import argparse
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
from ramp_summary import harness  # noqa: E402

# Reference categorical palette (as in plot_pd_interference.py), plus neutral ink.
BLUE, ORANGE, YELLOW = "#2a78d6", "#eb6834", "#eda100"
DECODE_GRAY = "#9c9a95"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_STYLE = [(BLUE, "o", "-"), (ORANGE, "^", "--")]  # per --ramp, in order
PREFILL_CLASSES = ("prefill_only", "mixed")


def level_stats(level_dir):
    """Summed step mix and means over a level's kept windows, plus the harness's TPOT/TTFT."""
    rows = [json.loads(line) for line in open(os.path.join(level_dir, "step_mix.jsonl"))][1:]
    peak = max((r["running"] for r in rows), default=0)
    rows = [r for r in rows if r["running"] >= 0.5 * peak]  # drop the drain minutes
    p = dict(steps=0.0, tokens=0.0, wall_ms=0.0)
    d = dict(steps=0.0, tokens=0.0, wall_ms=0.0)
    for r in rows:
        for step_class, c in r["step_mix"].items():
            group = p if step_class in PREFILL_CLASSES else d
            for f in group:
                group[f] += c[f]
    steps, wall = p["steps"] + d["steps"], p["wall_ms"] + d["wall_ms"]
    h = harness(os.path.join(level_dir, "harness.log"))
    nan = float("nan")
    mean = lambda key: sum(r[key] for r in rows) / len(rows) if rows else nan  # noqa: E731
    return {
        "agents": int(os.path.basename(level_dir)[1:]),
        "windows": len(rows),
        "running": mean("running"),
        "waiting": mean("waiting"),
        "prefill_step_pct": 100 * p["steps"] / steps if steps else nan,
        "prefill_time_pct": 100 * p["wall_ms"] / wall if wall else nan,
        "prefill_tokens_per_step": p["tokens"] / p["steps"] if p["steps"] else nan,
        "prefill_step_ms": p["wall_ms"] / p["steps"] if p["steps"] else nan,
        "decode_step_ms": d["wall_ms"] / d["steps"] if d["steps"] else nan,
        "step_tpot_ms": wall / steps if steps else nan,
        "tpot_decode_part_ms": d["wall_ms"] / steps if steps else nan,
        "tpot_prefill_part_ms": p["wall_ms"] / steps if steps else nan,
        "output_tok_s": mean("gen_tok_s"),
        "cache_hit_pct": 100 * mean("cache_hit_rate"),
        "client_tpot_p50_ms": h.get("TPOT", (nan,) * 3)[1],
        "client_ttft_p50_ms": h.get("TTFT", (nan,) * 3)[1],
    }


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp", action="append", required=True, help="LABEL=<ramp_agents.sh dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--setup", default="Aggregated MInf, one GB200 node per model: TP4 x EP4, NVLS "
                    "dispatch, TRT-LLM BF16 routed experts, LRU prefix caching, streaming",
                    help="second title line: the serving setup")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = []
    for spec in args.ramp:
        label, path = spec.split("=", 1)
        levels = [level_stats(d) for d in glob.glob(os.path.join(path, "w*"))
                  if os.path.exists(os.path.join(d, "step_mix.jsonl"))]
        levels = sorted((s for s in levels if s["windows"]), key=lambda s: s["agents"])
        models.append((label, levels))

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        fields = ["model"] + list(models[0][1][0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, levels in models:
            for s in levels:
                writer.writerow({"model": label, **{k: round(v, 3) if isinstance(v, float) else v
                                                    for k, v in s.items()}})

    fig, axes = plt.subplots(2, 4, figsize=(19, 7.8))
    (ax_steps, ax_share, ax_pms, ax_tok), (ax_n, ax_s, ax_tput, ax_fit) = axes
    stacked_axes = (ax_n, ax_s)
    line_panels = [
        (ax_steps, "prefill_step_pct", "(a) Engine steps that are prefill steps", "% of steps"),
        (ax_share, "prefill_time_pct", "(b) Engine time spent in prefill steps", "% of engine time"),
        (ax_pms, "prefill_step_ms", "(c) Time per prefill step", "ms per prefill step"),
        (ax_tok, "prefill_tokens_per_step", "(d) Tokens per prefill step (calls batched together)",
         "input tokens per prefill step"),
        (ax_tput, "output_tok_s", "(g) Output throughput", "output tok/s (one node, 4 GPUs)"),
    ]
    for (label, levels), (color, marker, line) in zip(models, MODEL_STYLE):
        x = [s["agents"] for s in levels]
        for ax, key, _, _ in line_panels:
            ax.plot(x, [s[key] for s in levels], color=color, marker=marker, linestyle=line,
                    linewidth=2, markersize=8, label=label)
        # (h) prefill step time against its tokens, one point per level, labeled by agents.
        toks = [s["prefill_tokens_per_step"] for s in levels]
        ax_fit.plot(toks, [s["prefill_step_ms"] for s in levels], color=color, marker=marker,
                    linestyle=line, linewidth=2, markersize=8, label=label)
        for i, s in enumerate(levels):  # alternate below / above so close points stay legible
            ax_fit.annotate(str(s["agents"]), (s["prefill_tokens_per_step"], s["prefill_step_ms"]),
                            textcoords="offset points", xytext=(6, -12 if i % 2 == 0 else 6),
                            fontsize=8, color=MUTED)
    agents = sorted({s["agents"] for _, levels in models for s in levels})
    for ax, key, title, ylabel in line_panels:
        style_axis(ax, title, ylabel)
        ax.set_xlabel("concurrent agents", color=MUTED)
        ax.set_xticks(agents)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0, 100) if key.endswith("_pct") else ax.set_ylim(bottom=0)
    style_axis(ax_fit, "(h) Prefill step time vs its tokens", "ms per prefill step")
    ax_fit.set_xlabel("input tokens per prefill step (point label: agents)", color=MUTED)
    ax_fit.set_xlim(left=0)
    ax_fit.set_ylim(bottom=0)
    ax_fit.legend(frameon=False, fontsize=9, loc="upper left")

    for ax, (label, levels), tag in zip(stacked_axes, models, "ef"):
        pos = range(len(levels))
        dec = [s["tpot_decode_part_ms"] for s in levels]
        pre = [s["tpot_prefill_part_ms"] for s in levels]
        ax.bar(pos, dec, width=0.6, color=DECODE_GRAY, hatch="\\\\\\", edgecolor="white",
               linewidth=0.5, label="decode steps")
        ax.bar(pos, pre, width=0.6, bottom=dec, color=YELLOW, hatch="///", edgecolor="white",
               linewidth=0.5, label="prefill steps")
        ax.plot(pos, [s["client_tpot_p50_ms"] for s in levels], linestyle="none", marker="D",
                markersize=7, color=TEXT, label="client p50 TPOT")
        for i, s in enumerate(levels):
            ax.text(i, s["step_tpot_ms"] + 0.4, f"{s['prefill_time_pct']:.0f}%", ha="center",
                    va="bottom", fontsize=8, color=MUTED)
        ax.set_xticks(list(pos), [str(s["agents"]) for s in levels])
        ax.set_xlabel("concurrent agents (label: prefill share of engine time)", color=MUTED)
        style_axis(ax, f"({tag}) {label}: step TPOT by step type", "ms per output token")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_ylim(0, 1.18 * max(s["step_tpot_ms"] for s in levels))

    fig.suptitle("Agentic SWE load (OpenSandbox, ~96% prefix-cache hits): prefill steps take over "
                 "the engine as agents grow\n" + args.setup, fontsize=12, color=TEXT, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "prefill_share.png"), dpi=160)
    print(f"wrote {args.out}/prefill_share.png and summary.csv")


if __name__ == "__main__":
    main()
