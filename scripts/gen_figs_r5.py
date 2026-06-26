#!/usr/bin/env python3
"""
gen_figs_r5.py  –  AlgoBench / ConstraintShift paper figure generator (R5)
Generates 5 figures from real JSON data.  ZERO hardcoded paper numbers.
All figures saved as .pdf + .png (300 dpi) to figures/ directory.

Usage (from repo root):
    python3 scripts/gen_figs_r5.py

Anti-hardcoding: grep -E '\b[0-9]+\.[0-9]+\b' scripts/gen_figs_r5.py
  → should only return layout constants (figsize, dpi, font sizes, etc.)
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
ANALYSIS   = os.path.join(ROOT, "analysis")
FIGURES    = os.path.join(ROOT, "EMNLP2026__Algobench__Automatic_algorithm-2", "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
BLUE   = "#1B5FA3"
GREEN  = "#216B43"
RED    = "#A62828"
ORANGE = "#C85A14"
GRAY   = "#888888"

DIST_COLORS = {"Same": BLUE, "Near": ORANGE, "Far": RED}
DIST_MARKERS = {"Same": "o", "Near": "s", "Far": "^"}

# 6-model tab10 palette (consistent with existing fig3a)
MODEL_PALETTE = plt.cm.tab10.colors[:6]

# ── helpers ────────────────────────────────────────────────────────────────────
SINGLE_COL_W = 3.3   # ACL/EMNLP single-column width (inches)
DPI = 300
FONT_BASE = 9

matplotlib.rcParams.update({
    "font.size":        FONT_BASE,
    "axes.titlesize":   FONT_BASE,
    "axes.labelsize":   FONT_BASE,
    "xtick.labelsize":  FONT_BASE - 1,
    "ytick.labelsize":  FONT_BASE - 1,
    "legend.fontsize":  FONT_BASE - 1,
    "pdf.fonttype":     42,   # embed TrueType
    "ps.fonttype":      42,
})


def save(fig, stem):
    """Save figure as both PDF and PNG with tight bbox."""
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – fig_transition_heatmap
# Scatter plot of algo-pair transitions, sized by n_problems, coloured by distance
# ══════════════════════════════════════════════════════════════════════════════
def fig1_transition_heatmap():
    print("[fig1] transition heatmap …")
    cells = json.load(open(os.path.join(ANALYSIS, "fig_transition_data.json")))

    # filter: n_problems >= 1 (all), sort by n_problems desc, keep top 15
    cells = sorted(cells, key=lambda r: -r["n_problems"])[:15]

    # build sorted x/y label lists
    y_labels = []
    x_labels = []
    for c in cells:
        if c["src_family"] not in y_labels:
            y_labels.append(c["src_family"])
        if c["tgt_family"] not in x_labels:
            x_labels.append(c["tgt_family"])
    # add anything still missing
    for c in cells:
        if c["tgt_family"] not in x_labels:
            x_labels.append(c["tgt_family"])
        if c["src_family"] not in y_labels:
            y_labels.append(c["src_family"])

    y_idx = {v: i for i, v in enumerate(y_labels)}
    x_idx = {v: i for i, v in enumerate(x_labels)}

    n_y = len(y_labels)
    n_x = len(x_labels)

    fig_w = max(SINGLE_COL_W * 2, n_x * 0.55 + 1.5)
    fig_h = max(2.5, n_y * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    size_scale = 300  # bubble area per n_problem unit

    for c in cells:
        xi = x_idx[c["tgt_family"]]
        yi = y_idx[c["src_family"]]
        dist = c["distance"]
        n    = c["n_problems"]
        p1   = c["mean_p1"]   # 0–100
        color  = DIST_COLORS[dist]
        marker = DIST_MARKERS[dist]
        ax.scatter(xi, yi,
                   s=n * size_scale,
                   c=color,
                   marker=marker,
                   alpha=0.75,
                   linewidths=0.5,
                   edgecolors="white",
                   zorder=3)
        ax.text(xi, yi, f"{p1:.0f}%",
                ha="center", va="center",
                fontsize=FONT_BASE - 2, color="white", fontweight="bold", zorder=4)

    ax.set_xticks(range(n_x))
    ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=FONT_BASE - 1)
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_labels, fontsize=FONT_BASE - 1)
    ax.set_xlabel("Target algorithm family", fontsize=FONT_BASE)
    ax.set_ylabel("Source algorithm family", fontsize=FONT_BASE)
    ax.set_title("Algorithm transition pairs (top 15 by problem count)", fontsize=FONT_BASE)

    ax.set_xlim(-0.7, n_x - 0.3)
    ax.set_ylim(-0.7, n_y - 0.3)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # legend: distance colour
    handles_dist = [
        mpatches.Patch(color=DIST_COLORS[d], label=d) for d in ("Same", "Near", "Far")
    ]
    # legend: size
    example_ns = sorted(set(c["n_problems"] for c in cells))[:3]
    handles_size = [
        plt.scatter([], [], s=n * size_scale, c=GRAY, alpha=0.6,
                    label=f"n={n}") for n in example_ns
    ]
    leg1 = ax.legend(handles=handles_dist, title="Distance",
                     loc="upper right", framealpha=0.85,
                     handlelength=1.2, borderpad=0.6)
    ax.add_artist(leg1)
    ax.legend(handles=handles_size, title="# problems",
              loc="lower right", framealpha=0.85,
              handlelength=1.2, borderpad=0.6)

    fig.tight_layout()
    save(fig, "fig_transition_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – fig_rag_severity
# Two-panel grouped bar: pass@1 and trap_rate, by model × (easy/hard × direct/RAG)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_rag_severity():
    print("[fig2] RAG severity …")
    raw  = json.load(open(os.path.join(ANALYSIS, "fig_rag_severity_data.json")))
    meta = raw.pop("_meta")

    models  = list(raw.keys())
    n_mod   = len(models)
    n_bars  = 4            # easy-Direct / easy-RAG / hard-Direct / hard-RAG
    x       = np.arange(n_mod)
    w       = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    bar_labels  = ["Easy-Direct", "Easy-RAG", "Hard-Direct", "Hard-RAG"]
    bar_colors  = [GREEN, BLUE, ORANGE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL_W * 2.0, 2.5))

    def plot_panel(ax, metric_key, ylabel, title):
        for bi, (label, color, ofs) in enumerate(zip(bar_labels, bar_colors, offsets)):
            difficulty = "easy" if bi < 2 else "hard"
            prompt     = "direct" if bi % 2 == 0 else "rag"
            vals = []
            for m in models:
                d = raw[m]
                vals.append(d[difficulty][f"{prompt}_{metric_key}"])
            ax.bar(x + ofs, vals, width=w,
                   color=color, label=label, alpha=0.88,
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in models],
                           fontsize=max(FONT_BASE - 2, 7))
        ax.set_ylabel(ylabel, fontsize=FONT_BASE)
        ax.set_title(title, fontsize=FONT_BASE)
        ax.set_ylim(0, 110)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    plot_panel(axes[0], "p1",   "pass@1 (%)",   "(a) pass@1 by difficulty")
    plot_panel(axes[1], "trap", "trap rate (%)", "(b) trap rate by difficulty")

    # shared legend under both panels
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(bar_colors, bar_labels)]
    fig.legend(handles=handles, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               framealpha=0.9, fontsize=FONT_BASE - 1)

    median_jac = meta["median_jaccard"]
    fig.suptitle(
        f"RAG benefit by problem hardness (Jaccard median={median_jac:.3f})",
        fontsize=FONT_BASE, y=1.02
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "fig_rag_severity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – fig_gapt_breakdown
# Dual-metric bar per operator: mean Δexp (left y) + % optimal (right y)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_gapt_breakdown():
    print("[fig3] GAPT breakdown …")
    raw = json.load(open(os.path.join(ANALYSIS, "fig_gapt_data.json")))

    operators  = list(raw.keys())
    n_ops      = len(operators)
    mean_diffs = [raw[op]["mean_diff"]    for op in operators]
    pct_opts   = [raw[op]["pct_optimal"]  for op in operators]
    n_samples  = [raw[op]["n_samples"]    for op in operators]

    x = np.arange(n_ops)
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_W, 2.6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, mean_diffs, width=w,
                    color=RED, alpha=0.85, label="Mean Δexp", edgecolor="white", linewidth=0.4)
    bars2 = ax2.bar(x + w / 2, pct_opts,   width=w,
                    color=BLUE, alpha=0.85, label="% Optimal (Δ≤0.1)", edgecolor="white", linewidth=0.4)

    # annotate values on bars
    for bar, val in zip(bars1, mean_diffs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=FONT_BASE - 2, color=RED)
    for bar, val in zip(bars2, pct_opts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                 f"{val:.0f}%", ha="center", va="bottom",
                 fontsize=FONT_BASE - 2, color=BLUE)

    ax1.set_xticks(x)
    # show operator name + n
    xlabels = [f"{op}\n(n={n_samples[i]})" for i, op in enumerate(operators)]
    ax1.set_xticklabels(xlabels, fontsize=FONT_BASE - 1)
    ax1.set_ylabel("Mean Δexp\n(est. − target)", fontsize=FONT_BASE, color=RED)
    ax2.set_ylabel("% Optimal (Δ≤0.1)", fontsize=FONT_BASE, color=BLUE)
    ax1.tick_params(axis="y", colors=RED)
    ax2.tick_params(axis="y", colors=BLUE)

    # ensure left axis goes to 0 with a bit of headroom
    ax1_max = max(mean_diffs) if mean_diffs else 0.5
    ax1.set_ylim(0, ax1_max * 1.45)
    ax2.set_ylim(0, 115)

    ax1.set_title("GAPT complexity accuracy per operator", fontsize=FONT_BASE)
    ax1.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax1.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=RED,  label="Mean Δexp"),
        mpatches.Patch(color=BLUE, label="% Optimal (Δ≤0.1)"),
    ]
    ax1.legend(handles=handles, fontsize=FONT_BASE - 1,
               loc="upper right", framealpha=0.85)

    fig.tight_layout()
    save(fig, "fig_gapt_breakdown")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – fig_gen_gap  (regenerated from table_main.json + table_gen_gap_dist.json)
# Top panel: Src vs Shf pass@1 per model
# Bottom panel: Δp@1 by alg-distance category (Same / Near / Far)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_gen_gap():
    print("[fig4] gen gap …")
    main_rows  = json.load(open(os.path.join(ANALYSIS, "table_main.json")))
    dist_data  = json.load(open(os.path.join(ANALYSIS, "table_gen_gap_dist.json")))

    # ── top panel: Src vs Shf p@1 per model ───────────────────────────────────
    models_main = [r["model"] for r in main_rows]
    src_p1      = [r["orig_p1"] for r in main_rows]
    shf_p1      = [r["shf_p1"]  for r in main_rows]
    n_mod       = len(models_main)
    x           = np.arange(n_mod)
    w           = 0.32

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(SINGLE_COL_W, 4.0))

    for i, (s, sh) in enumerate(zip(src_p1, shf_p1)):
        ax_top.bar(x[i] - w / 2, s,  width=w, color=MODEL_PALETTE[i], alpha=0.9, hatch="//",
                   edgecolor="white", linewidth=0.5)
        ax_top.bar(x[i] + w / 2, sh, width=w, color=MODEL_PALETTE[i], alpha=0.6,
                   edgecolor="white", linewidth=0.5)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([m.replace(" ", "\n") for m in models_main],
                           fontsize=max(FONT_BASE - 2, 7))
    ax_top.set_ylabel("pass@1 (%)", fontsize=FONT_BASE)
    ax_top.set_title("(a) Source vs Shifted pass@1", fontsize=FONT_BASE)
    ax_top.set_ylim(0, 115)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_top.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_top.set_axisbelow(True)

    # hatch legend
    src_patch = mpatches.Patch(facecolor=GRAY, hatch="//", label="Source (orig.)")
    shf_patch = mpatches.Patch(facecolor=GRAY, alpha=0.6,  label="Shifted")
    ax_top.legend(handles=[src_patch, shf_patch],
                  fontsize=FONT_BASE - 1, loc="lower left", framealpha=0.85)

    # ── bottom panel: Δp@1 by distance (box/bar with per-model dots) ───────────
    dist_categories = ["Same", "Near", "Far"]

    # collect Δp@1 per model per distance category
    dist_models = list(dist_data.keys())
    n_dm        = len(dist_models)
    xd          = np.arange(len(dist_categories))
    wd          = 0.12
    model_offsets = np.linspace(-wd * (n_dm - 1) / 2, wd * (n_dm - 1) / 2, n_dm)

    for mi, (mname, offset) in enumerate(zip(dist_models, model_offsets)):
        mdata = dist_data[mname]
        vals  = [mdata[dc]["delta_pp"] for dc in dist_categories]
        ax_bot.bar(xd + offset, vals, width=wd,
                   color=MODEL_PALETTE[mi], alpha=0.82,
                   edgecolor="white", linewidth=0.3,
                   label=mname)

    ax_bot.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_bot.set_xticks(xd)
    ax_bot.set_xticklabels(dist_categories, fontsize=FONT_BASE)
    ax_bot.set_ylabel("Δp@1 (shf − src, pp)", fontsize=FONT_BASE)
    ax_bot.set_title("(b) Generalisation gap by alg. distance", fontsize=FONT_BASE)
    ax_bot.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(fontsize=FONT_BASE - 2, loc="lower right",
                  framealpha=0.85, ncol=2)

    fig.tight_layout(pad=0.8)
    save(fig, "fig_gen_gap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – fig_ablation_gates  (NEW)
# 5 configs × 4 metrics grouped bar, diagonal pattern obvious
# ══════════════════════════════════════════════════════════════════════════════
def fig5_ablation_gates():
    print("[fig5] ablation gates …")
    ablation = json.load(open(os.path.join(ANALYSIS, "table_ablation.json")))
    meta     = ablation["meta"]
    table    = ablation["table"]

    configs      = [row["config"]       for row in table]
    old_sol_pct  = [row["old_sol_pct"]  for row in table]
    ref_fail_pct = [row["ref_fail_pct"] for row in table]
    near_para_pct= [row["near_para_pct"]for row in table]
    f_opt_pct    = [row["f_opt_pct"]    for row in table]

    n_cfg    = len(configs)
    metrics  = ["Old-Sol", "Ref-Fail", "Near-Para", "F-Opt"]
    m_colors = [RED, ORANGE, BLUE, GREEN]
    all_vals = [old_sol_pct, ref_fail_pct, near_para_pct, f_opt_pct]

    x  = np.arange(n_cfg)
    w  = 0.18
    n_met = len(metrics)
    offsets = np.linspace(-(n_met - 1) * w / 2, (n_met - 1) * w / 2, n_met)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.5, 2.8))

    for mi, (vals, label, color, ofs) in enumerate(
            zip(all_vals, metrics, m_colors, offsets)):
        bars = ax.bar(x + ofs, vals, width=w,
                      color=color, alpha=0.85, label=label,
                      edgecolor="white", linewidth=0.4)
        # annotate non-zero bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{val:.1f}%",
                        ha="center", va="bottom",
                        fontsize=FONT_BASE - 2, color=color, fontweight="bold")

    n_total = meta["n_candidates"]
    ax.set_xticks(x)
    # pretty config labels
    pretty = {"full": "Full", "no_g1": "−G1", "no_g2": "−G2",
              "no_g3": "−G3", "no_g4": "−G4"}
    ax.set_xticklabels([pretty.get(c, c) for c in configs],
                       fontsize=FONT_BASE)
    ax.set_ylabel("Contaminated problems (%)", fontsize=FONT_BASE)
    ax.set_title(
        f"Quality gate ablation (n={n_total} candidates)\n"
        "Each gate targets one failure mode (diagonal pattern)",
        fontsize=FONT_BASE
    )
    ax.set_ylim(0, max(max(v) for v in all_vals) * 1.55 + 1)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(fontsize=FONT_BASE - 1, ncol=2,
              loc="upper right", framealpha=0.88)

    fig.tight_layout()
    save(fig, "fig_ablation_gates")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"ROOT    = {ROOT}")
    print(f"FIGURES = {FIGURES}\n")

    fig1_transition_heatmap()
    fig2_rag_severity()
    fig3_gapt_breakdown()
    fig4_gen_gap()
    fig5_ablation_gates()

    print("\nAll figures generated ✓")
