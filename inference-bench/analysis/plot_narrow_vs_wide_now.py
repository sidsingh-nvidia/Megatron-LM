"""Narrow EP vs wide EP on the same code: per-GPU Pareto and TPOT, one row per model. MInf, and
optionally vLLM (two more runs per model).

Narrow and wide runs are compared at the same load per GPU (concurrency / GPUs). The
one-request-per-GPU level is dropped, as in paper_figures.py.

    python3 inference-bench/analysis/plot_narrow_vs_wide_now.py \
        --superv3 <wide run> <narrow run> --ultrav3 <wide run> <narrow run> \
        [<vLLM wide run> <vLLM narrow run>]
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_figures as pf

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
MINF, VLLM = "#0072B2", "#D55E00"
# Extra runs (--extra-<model>), in order: bluish green, reddish purple, orange (Okabe-Ito).
EXTRA_STYLES = [("#009E73", "s"), ("#CC79A7", "D"), ("#E69F00", "v")]
TITLES = {"superv3": "Super", "ultrav3": "Ultra"}


def layout(run):
    """Parallelism of a run from its config.env, as in the paper: 'TP8×DP8, EP64'."""
    cfg = open(os.path.join(pf.EXPERIMENTS, run, "config.env")).read()
    tp, ep = (int(re.search(rf"\b{k}=(\d+)", cfg).group(1)) for k in ("TP", "EP"))
    dp = pf.gpus_of(run) // tp
    return f"TP{tp}" + (f"×DP{dp}" if dp > 1 else "") + f", EP{ep}"


def load(run):
    """Per-level records: load per GPU, throughput per GPU, p90 interactivity, median TPOT, and
    the mean clean decode step (decode-only steps, all coordinators; None if not recorded)."""
    gpus = pf.gpus_of(run)
    out = []
    for line in open(os.path.join(pf.EXPERIMENTS, run, "steady.jsonl")):
        r = json.loads(line)
        if r["concurrency"] == gpus:  # one request per GPU: dropped, as in the paper's figures
            continue
        clean = r.get("step_time_ms", {}).get("all", {}).get("decode_clean", {}).get("mean_ms")
        out.append((r["concurrency"] / gpus, r["output_tok_s"] / gpus, r[pf.INTERACTIVITY],
                    r["tpot_ms_p50"], clean))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    for model in ("superv3", "ultrav3"):
        ap.add_argument(f"--{model}", nargs="+",
                        help="MInf wide and narrow runs, then optionally vLLM wide [and narrow]")
        ap.add_argument(f"--extra-{model}", nargs="*", default=[], metavar="LABEL=RUN",
                        help="more runs, e.g. vLLM variants, each drawn as its own series")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "narrow_vs_wide_now.png"))
    args = ap.parse_args()
    models = [(m, getattr(args, m)) for m in ("superv3", "ultrav3") if getattr(args, m)]

    fig, axes = plt.subplots(len(models), 3, figsize=(16, 4.2 * len(models)), squeeze=False)
    for row, (model, runs) in enumerate(models):
        assert 2 <= len(runs) <= 4, f"--{model}: MInf wide narrow [vLLM wide [narrow]]"
        series = []
        for name, color, pair in (("MInf", MINF, runs[:2]), ("vLLM", VLLM, runs[2:])):
            for kind, run, marker, ls in zip(("wide", "narrow"), pair, ("o", "^"), ("-", "--")):
                series.append((f"{name}, {kind} EP: {layout(run)} ({pf.gpus_of(run)} GPUs)", load(run), color,
                               marker, ls))
        for (color, marker), spec in zip(EXTRA_STYLES, getattr(args, f"extra_{model}")):
            label, run = spec.split("=", 1)
            series.append((f"{label}: {layout(run)} ({pf.gpus_of(run)} GPUs)", load(run), color,
                           marker, "-"))
        pareto, tpot, step = axes[row]
        for label, pts, color, marker, ls in series:
            per_gpu, tput, inter, t, clean = zip(*pts)
            pareto.plot(inter, tput, color=color, marker=marker, ls=ls, lw=2, ms=7, label=label)
            for x, y, c in zip(inter, tput, per_gpu):
                pareto.annotate(f"{c:g}/GPU", (x, y), xytext=(5, -2), textcoords="offset points",
                                fontsize=7, color="#555555")
            tpot.plot(per_gpu, t, color=color, marker=marker, ls=ls, lw=2, ms=7, label=label)
            measured = [(c, v) for c, v in zip(per_gpu, clean) if v is not None]
            if measured:
                step.plot(*zip(*measured), color=color, marker=marker, ls=ls, lw=2, ms=7,
                          label=label)
        if len(series) == 2:  # MInf only: narrow / wide at each shared load
            for ax, col in ((tpot, 3), (step, 4)):
                wide_v = {p[0]: p[col] for p in series[0][1] if p[col] is not None}
                for p in series[1][1]:
                    if p[0] in wide_v and p[col] is not None:
                        ax.annotate(f"{p[col] / wide_v[p[0]]:.2f}×", (p[0], p[col]),
                                    xytext=(0, 8), textcoords="offset points", ha="center",
                                    fontsize=8, color="#555555")
        pareto.set_title(f"{TITLES[model]}: throughput vs p90 interactivity", fontsize=10,
                         loc="left")
        pareto.set_xlabel("p90 interactivity (tokens/s/user)")
        pareto.set_ylabel("Throughput (tokens/s/GPU)")
        ratio = " (labels: narrow / wide)" if len(series) == 2 else ""
        tpot.set_title(f"{TITLES[model]}: median TPOT{ratio}", fontsize=10, loc="left")
        step.set_title(f"{TITLES[model]}: MInf clean decode step{ratio}", fontsize=10, loc="left")
        step.set_ylabel("Mean clean decode step (ms)")
        loads = sorted({c for _, pts, *_ in series for c, *_ in pts})
        for ax in (tpot, step):
            ax.set_xscale("log", base=2)
            ax.set_xticks(loads, [f"{c:g}" for c in loads])
            ax.set_xlabel("Requests in flight per GPU")
        tpot.set_ylabel("Median TPOT (ms)")
        for ax in (pareto, tpot, step):
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.grid(color="#EEEEEE", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    # One legend for every panel, below them: series labels are too long to sit inside a panel.
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
               ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 0))
    legend_rows = -(-len(labels) // 3)
    fig.tight_layout(rect=(0, 0.06 * legend_rows / len(models), 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
