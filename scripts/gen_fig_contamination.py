#!/usr/bin/env python3
"""
gen_fig_contamination.py
Generates fig_contamination.{pdf,png} from real complexity-tier data.

Data source: analysis/fig_contamination_by_tier.json
Output:      EMNLP2026__Algobench__Automatic_algorithm-2/figures/fig_contamination.{pdf,png}

Anti-hardcoding: all bar values are read from JSON at runtime.
Skip O(n^3+) tier (n=1, statistically unreliable).
Skip O(1) tier (no data in JSON).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

DATA_PATH  = os.path.join(REPO_ROOT, "analysis", "fig_contamination_by_tier.json")
FIG_DIR    = os.path.join(REPO_ROOT, "EMNLP2026__Algobench__Automatic_algorithm-2", "figures")
OUT_PDF    = os.path.join(FIG_DIR, "fig_contamination.pdf")
OUT_PNG    = os.path.join(FIG_DIR, "fig_contamination.png")

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)

SKIP_TIERS = {"O(1)", "O(n^3+)"}   # no-data / n=1 unreliable
tier_data  = raw["data"]
tiers_all  = raw["tiers_ordered"]
tiers_plot = [t for t in tiers_all if t in tier_data and t not in SKIP_TIERS]

# Pull values from JSON — NO hardcoding
src_vals   = [tier_data[t]["src_mean"]  for t in tiers_plot]
shf_vals   = [tier_data[t]["shf_mean"]  for t in tiers_plot]
delta_vals = [tier_data[t]["delta_pp"]  for t in tiers_plot]
n_vals     = [tier_data[t]["n_pids"]    for t in tiers_plot]

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE         = "#1B5FA3"
ORANGE       = "#C85A14"
SINGLE_COL_W = 3.5    # inches
FIG_H        = 3.0    # inches
FONT_SIZE    = 9
ANN_FONT     = 6.5

TIER_LABELS = {
    "O(log n)":   r"$O(\log n)$",
    "O(n)":       r"$O(n)$",
    "O(n log n)": r"$O(n \log n)$",
    "O(n^2)":     r"$O(n^2)$",
    "O(n^3+)":    r"$O(n^3{+})$",
    "exp":        r"exp",
    "other":      r"other",
}

# ── Compute annotation y-positions (greedy, ensures ≥11pp gap) ────────────────
def compute_ann_ys(bar_tops, base_pad=2.5, min_sep=11.0):
    """
    Place annotation y-positions at bar_top+base_pad, but ensure that each
    adjacent annotation is ≥ min_sep away from its left neighbour.
    If a conflict exists, try above the neighbour first; fall back to below
    only if below ≥ bar_top+1.
    """
    ys = []
    for i, top in enumerate(bar_tops):
        candidate = top + base_pad
        if ys and abs(candidate - ys[-1]) < min_sep:
            above = ys[-1] + min_sep
            below = ys[-1] - min_sep
            candidate = below if below >= top + 1.0 else above
        ys.append(candidate)
    return ys

bar_tops = [max(s, h) for s, h in zip(src_vals, shf_vals)]
ann_ys   = compute_ann_ys(bar_tops)

# ── Build figure ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE - 1,
    "ytick.labelsize":  FONT_SIZE - 1,
    "legend.fontsize":  FONT_SIZE - 1,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

fig, ax = plt.subplots(figsize=(SINGLE_COL_W, FIG_H))

n_tiers = len(tiers_plot)
x       = np.arange(n_tiers)
bar_w   = 0.32
gap     = 0.04

x_src = x - bar_w / 2 - gap / 2
x_shf = x + bar_w / 2 + gap / 2

# Source bars — ORANGE + hatching
ax.bar(x_src, src_vals,
       width=bar_w, color=ORANGE, edgecolor="white",
       hatch="///", linewidth=0.5,
       zorder=3, label="Source (original)")

# Shifted bars — BLUE solid
ax.bar(x_shf, shf_vals,
       width=bar_w, color=BLUE, edgecolor="white",
       linewidth=0.5,
       zorder=3, label="Shifted")

# ── Δ annotations — non-overlapping via greedy placement ──────────────────────
for i, (d, n, ann_y) in enumerate(zip(delta_vals, n_vals, ann_ys)):
    sign  = "+" if d >= 0 else ""
    line1 = f"Δ={sign}{d:.1f}pp"
    line2 = f"(n={n})"
    ax.text(x[i], ann_y, f"{line1}\n{line2}",
            ha="center", va="bottom",
            fontsize=ANN_FONT, color="#333333",
            linespacing=1.3,
            multialignment="center")

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([TIER_LABELS.get(t, t) for t in tiers_plot],
                   rotation=30, ha="right")
ax.set_ylabel("pass@1 (%)", fontsize=FONT_SIZE)
ax.set_ylim(0, 125)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend — lower right to stay away from annotation cluster
ax.legend(loc="lower right", frameon=False,
          handlelength=1.4, handleheight=0.9,
          borderpad=0.3, labelspacing=0.3)

fig.tight_layout(pad=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
fig.savefig(OUT_PNG, format="png", dpi=300, bbox_inches="tight")
print(f"Saved PDF : {OUT_PDF}")
print(f"Saved PNG : {OUT_PNG}")

# Debug: print computed annotation y-positions for sanity check
for t, y in zip(tiers_plot, ann_ys):
    print(f"  {t}: ann_y={y:.1f}")
