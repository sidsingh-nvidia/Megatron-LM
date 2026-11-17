"""Decode-step breakdown from nsys traces: per-category GPU kernel time per decode step, MInf vs
vLLM, one GPU each, as grouped horizontal bars, with the wall time per step.

Input: SQLite exports of the traces (nsys export --type sqlite). Decode steps are counted with a
per-layer kernel common to both engines (_selective_scan_update_kernel, once per Mamba layer):
steps = its instances / its per-step count in the MInf trace, where _fused_metadata_kernel runs
once per step.

    python3 inference-bench/analysis/plot_decode_step_breakdown.py \\
        --minf minf_rank0.sqlite --vllm vllm_node0.sqlite --vllm-device 0 --out breakdown.png
"""

import argparse
import collections
import re
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Gill Sans", "Gill Sans MT", "URW Gothic", "Open Sans", "DejaVu Sans"]
MINF, VLLM = "#0072B2", "#D55E00"  # Okabe-Ito blue / vermillion, as in the other figures

# First match wins.
CATEGORIES = [
    ("EP dispatch / combine", r"moeA2A|_multimem_all_gatherv|_multimem_reduce_scatter_v|_fused_metadata"),
    ("TP / SP collectives", r"multimem_all_reduce|_multimem_all_gather_kernel|_multimem_reduce_scatter_kernel|nccl|[Aa]llreduce|lamport"),
    ("MoE routing", r"routing|topk|[Tt]opK"),
    ("MoE expert GEMM", r"bmm_|[Bb]atchedGemm|finalize|[Pp]ermute"),
    ("Attention", r"fmha|[Aa]ttention|attn|BatchPrefill|BatchDecode|flash_fwd|merge_state"),
    ("Mamba", r"ssd|mamba|chunk_scan|chunk_state|state_passing|selective|conv1d|causal_conv|bmm_chunk|chunk_cumsum"),
    ("Dense GEMM", r"nvjet|gemm|Gemm|cutlass|xmma|cublas|splitK"),
    ("Norm", r"[Nn]orm|rms"),
]
SCAN, STEP = "_selective_scan_update_kernel", "_fused_metadata_kernel"


def category(name):
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "Other"


def load(path, device=None):
    """{category: ns}, {kernel: instances}, window ns, for one device of a trace."""
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    devices = [d for (d,) in db.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    device = sorted(devices)[0] if device is None else device
    per, count, lo, hi = collections.Counter(), collections.Counter(), None, None
    for start, end, dn in db.execute(
        "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ?", (device,)
    ):
        name = names.get(dn) or "?"
        per[category(name)] += end - start
        count[name] += 1
        lo = start if lo is None else min(lo, start)
        hi = end if hi is None else max(hi, end)
    return per, count, hi - lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minf", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--vllm-device", type=int, default=None)
    ap.add_argument("--title", default="Ultra wide EP (TP8 x DP8, EP64), steady load c=512: one decode step, GPU 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m_per, m_count, m_win = load(args.minf)
    v_per, v_count, v_win = load(args.vllm, args.vllm_device)
    m_steps = m_count[STEP]
    scans_per_step = m_count[SCAN] / m_steps
    v_steps = v_count[SCAN] / scans_per_step
    ms = lambda ns, steps: ns / 1e6 / steps
    rows = {c: (ms(v_per[c], v_steps), ms(m_per[c], m_steps)) for c in set(v_per) | set(m_per)}
    order = sorted(rows, key=lambda c: rows[c][0] - rows[c][1])  # largest vLLM excess at the top
    labels = order + ["Decode step (wall)"]
    vals = [rows[c] for c in order] + [(ms(v_win, v_steps), ms(m_win, m_steps))]
    print(f"steps: MInf {m_steps}, vLLM {v_steps:.0f}")
    for label, (v, m) in zip(labels, vals):
        print(f"{label:24s} vLLM {v:6.2f}  MInf {m:6.2f}  diff {v - m:+6.2f} ms")

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labels) + 1.4))
    y = range(len(labels))
    h = 0.38
    vb = ax.barh([i + h / 2 for i in y], [v for v, _ in vals], h, color=VLLM, hatch="//",
                 edgecolor="white", linewidth=0, label="vLLM (stock: one-sided all-to-all, TRT-LLM-gen attention)")
    mb = ax.barh([i - h / 2 for i in y], [m for _, m in vals], h, color=MINF, edgecolor="white",
                 linewidth=0, label="MInf (NVLS all-gather-v / reduce-scatter-v, TRT-LLM BF16 MoE)")
    for bars in (vb, mb):
        for b in bars:
            ax.text(b.get_width() + 0.08, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}",
                    va="center", fontsize=8, color="#333333")
    ax.axhline(len(labels) - 1.5, color="#BBBBBB", lw=0.8)  # kernel categories below, wall step above
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("ms per decode step (GPU kernel time per category; wall time on top)")
    ax.set_title(args.title, fontsize=10, loc="left")
    ax.grid(axis="x", color="#EEEEEE", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(max(v, m) for v, m in vals) * 1.15)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, "Under nsys. vLLM kernels overlap across streams (sum > wall); collective "
             "kernels include waiting on peer ranks.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(args.out)


if __name__ == "__main__":
    main()
