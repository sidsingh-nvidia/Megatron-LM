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


def prefill_gpus(run):
    """GPUs of a PD run's prefill pool (0 for aggregated runs); NODES counts only the decode pool."""
    m = re.search(r"\bPREFILL_GPUS=(\d+)", open(os.path.join(pf.EXPERIMENTS, run, "config.env")).read())
    return int(m.group(1)) if m else 0


def layout(run):
    """Parallelism of a run from its config.env, as in the paper: 'TP8×DP8, EP64'; PD runs give the
    prefill pool too."""
    cfg = open(os.path.join(pf.EXPERIMENTS, run, "config.env")).read()
    tp, ep = (int(re.search(rf"\b{k}=(\d+)", cfg).group(1)) for k in ("TP", "EP"))
    dp = pf.gpus_of(run) // tp
    text = f"TP{tp}" + (f"×DP{dp}" if dp > 1 else "") + f", EP{ep}"
    shards = re.search(r"\bSHARDS=tp=(\d+),dp=\d+,ep=(\d+),[^ ]*role=prefill", cfg)
    if prefill_gpus(run) and shards:
        text = f"prefill TP{shards.group(1)}, EP{shards.group(2)} + decode {text}"
    return text


def total_gpus(run):
    """GPUs the run holds: the decode (or aggregated) GPUs plus any prefill pool."""
    return pf.gpus_of(run) + prefill_gpus(run)


def load(run):
    """Per-level records: load per GPU, throughput per GPU, p90 interactivity, median TPOT, and
    the mean clean decode step (decode-only steps, all coordinators; None if not recorded).
    Load is per decode (or aggregated) GPU; throughput is per GPU of the whole run, prefill pool
    included. A PD decode pool's steps are all clean decodes ("decode")."""
    gpus, all_gpus = pf.gpus_of(run), total_gpus(run)
    out = []
    for line in open(os.path.join(pf.EXPERIMENTS, run, "steady.jsonl")):
        r = json.loads(line)
        if r["concurrency"] == gpus:  # one request per GPU: dropped, as in the paper's figures
            continue
        steps = r.get("step_time_ms", {}).get("all", {})
        clean = (steps.get("decode_clean") or steps.get("decode") or {}).get("mean_ms")
        out.append((r["concurrency"] / gpus, r["output_tok_s"] / all_gpus, r[pf.INTERACTIVITY],
                    r["tpot_ms_p50"], clean))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    for model in ("superv3", "ultrav3"):
        ap.add_argument(f"--{model}", nargs="+",
                        help="MInf wide and narrow runs, then optionally vLLM wide [and narrow]; \"-\" skips a slot")
        ap.add_argument(f"--extra-{model}", nargs="*", default=[], metavar="LABEL=RUN",
                        help="more runs, e.g. vLLM variants, each drawn as its own series")
    ap.add_argument("--no-step", action="store_true",
                    help="drop the clean-decode-step panel (it needs the EP-consensus step classes)")
    ap.add_argument("--agg-name", default="MInf", help="series name of the first two (MInf) runs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "narrow_vs_wide_now.png"))
    args = ap.parse_args()
    models = [(m, getattr(args, m)) for m in ("superv3", "ultrav3") if getattr(args, m)]

    ncols = 2 if args.no_step else 3
    fig, axes = plt.subplots(len(models), ncols, figsize=(5.4 * ncols, 4.2 * len(models)), squeeze=False)
    for row, (model, runs) in enumerate(models):
        assert 2 <= len(runs) <= 4, f"--{model}: MInf wide narrow [vLLM wide [narrow]]"
        series = []
        for name, color, pair in ((args.agg_name, MINF, runs[:2]), ("vLLM", VLLM, runs[2:])):
            for kind, run, marker, ls in zip(("wide", "narrow"), pair, ("o", "^"), ("-", "--")):
                if run == "-":  # placeholder: no run for this slot
                    continue
                series.append((f"{name}, {kind} EP: {layout(run)} ({total_gpus(run)} GPUs)", load(run), color,
                               marker, ls))
        for (color, marker), spec in zip(EXTRA_STYLES, getattr(args, f"extra_{model}")):
            label, run = spec.split("=", 1)
            series.append((f"{label}: {layout(run)} ({total_gpus(run)} GPUs)", load(run), color,
                           marker, "-"))
        pareto, tpot, step = (*axes[row], None) if args.no_step else axes[row]
        for label, pts, color, marker, ls in series:
            per_gpu, tput, inter, t, clean = zip(*pts)
            pareto.plot(inter, tput, color=color, marker=marker, ls=ls, lw=2, ms=7, label=label)
            for x, y, c in zip(inter, tput, per_gpu):
                pareto.annotate(f"{c:g}/GPU", (x, y), xytext=(5, -2), textcoords="offset points",
                                fontsize=7, color="#555555")
            tpot.plot(per_gpu, t, color=color, marker=marker, ls=ls, lw=2, ms=7, label=label)
            measured = [(c, v) for c, v in zip(per_gpu, clean) if v is not None]
            if measured and step is not None:
                step.plot(*zip(*measured), color=color, marker=marker, ls=ls, lw=2, ms=7,
                          label=label)
        if len(series) == 2:  # MInf only: narrow / wide at each shared load
            for ax, col in [(tpot, 3)] + ([(step, 4)] if step is not None else []):
                wide_v = {p[0]: p[col] for p in series[0][1] if p[col] is not None}
                for p in series[1][1]:
                    if p[0] in wide_v and p[col] is not None:
                        ax.annotate(f"{p[col] / wide_v[p[0]]:.2f}×", (p[0], p[col]),
                                    xytext=(0, 8), textcoords="offset points", ha="center",
                                    fontsize=8, color="#555555")
        pareto.set_title(f"{TITLES[model]}: throughput vs p90 interactivity", fontsize=10,
                         loc="left")
        pareto.set_xlabel("p90 interactivity (tokens/s/user)")
        pareto.set_ylabel("Throughput (tokens/s/GPU, all GPUs)")
        ratio = " (labels: narrow / wide)" if len(series) == 2 else ""
        tpot.set_title(f"{TITLES[model]}: median TPOT{ratio}", fontsize=10, loc="left")
        if step is not None:
            step.set_title(f"{TITLES[model]}: MInf clean decode step{ratio}", fontsize=10, loc="left")
            step.set_ylabel("Mean clean decode step (ms)")
        loads = sorted({c for _, pts, *_ in series for c, *_ in pts})
        for ax in [a for a in (tpot, step) if a is not None]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(loads, [f"{c:g}" for c in loads])
            ax.set_xlabel("Requests in flight per GPU (PD: per decode GPU)")
        tpot.set_ylabel("Median TPOT (ms)")
        for ax in [a for a in (pareto, tpot, step) if a is not None]:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
            ax.grid(color="#EEEEEE", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    # One legend for every panel, below them: series labels are too long to sit inside a panel.
    handles, labels = axes[0][1].get_legend_handles_labels()
    ncol_legend = 1 if args.no_step else min(len(labels), 3)  # two panels are too narrow for three
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center",
               ncol=ncol_legend, bbox_to_anchor=(0.5, 0))
    legend_rows = -(-len(labels) // ncol_legend)
    fig.tight_layout(rect=(0, 0.06 * legend_rows / len(models), 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
