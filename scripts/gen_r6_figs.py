#!/usr/bin/env python3
"""R6: Regenerate fig_contamination + fig_consens to match human-authoritative numbers in R6 tables.
All data loaded from JSON. No hardcoded paper numbers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANA  = ROOT / 'analysis'
FIGS = ROOT / 'EMNLP2026__Algobench__Automatic_algorithm-2/figures'
FIGS.mkdir(parents=True, exist_ok=True)

BLUE   = '#1B5FA3'
ORANGE = '#C85A14'
GREEN  = '#216B43'
RED    = '#A62828'
GRAY   = '#777777'
SINGLE_COL_W = 3.3
FONT_BASE = 9

plt.rcParams.update({
    'font.size': FONT_BASE,
    'axes.titlesize': FONT_BASE,
    'axes.labelsize': FONT_BASE,
    'xtick.labelsize': FONT_BASE - 1,
    'ytick.labelsize': FONT_BASE - 1,
    'legend.fontsize': FONT_BASE - 1,
})

# ---------------------------------------------------------------- fig_contamination
def fig_contamination():
    data = json.load(open(ANA / 'fig_contamination_by_tier.json'))
    tiers_def = data['tiers']
    order = ['O(log n)', 'O(n)', 'O(n log n)', 'O(n^2)', 'other']
    labels_disp = {
        'O(log n)':   r'$O(\log n)$',
        'O(n)':       r'$O(n)$',
        'O(n log n)': r'$O(n\log n)$',
        'O(n^2)':     r'$O(n^2)$',
        'other':      'other',
    }

    tiers = [t for t in order if t in tiers_def]
    src   = [tiers_def[t]['src_mean']  for t in tiers]
    shf   = [tiers_def[t]['shf_mean']  for t in tiers]
    delta = [tiers_def[t]['delta_pp']  for t in tiers]
    ns    = [tiers_def[t]['n']         for t in tiers]
    x_disp= [labels_disp[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.2, 2.4))
    x = np.arange(len(tiers))
    w = 0.36

    bars_src = ax.bar(x - w/2, src, w, color=BLUE,   alpha=0.92, label='Source',  edgecolor='white', linewidth=0.4)
    bars_shf = ax.bar(x + w/2, shf, w, color=ORANGE, alpha=0.92, label='Shifted', edgecolor='white', linewidth=0.4)

    # Δ annotation above Shf bar
    for i, (sb, d) in enumerate(zip(bars_shf, delta)):
        h = sb.get_height()
        ax.annotate(
            f'$\\Delta{{=}}{d:+.1f}$',
            xy=(sb.get_x() + sb.get_width()/2, h),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom',
            fontsize=FONT_BASE - 2,
            color=RED if d <= -10 else GRAY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f'{lbl}\n($n{{=}}{n}$)' for lbl, n in zip(x_disp, ns)],
                       fontsize=FONT_BASE - 1)
    ax.set_ylabel('pass@1 (%)')
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', framealpha=0.85, ncol=2)

    fig.tight_layout(pad=0.4)
    for ext in ['pdf','png']:
        fig.savefig(FIGS / f'fig_contamination.{ext}', dpi=300 if ext=='png' else None, bbox_inches='tight')
    plt.close(fig)
    print('[fig_contamination] tiers:', tiers)
    print('[fig_contamination] src:', src)
    print('[fig_contamination] shf:', shf)
    print('[fig_contamination] saved to', FIGS / 'fig_contamination.{pdf,png}')

# ---------------------------------------------------------------- fig_consens
def fig_consens():
    ckpt = json.load(open(ANA / 'sweep_constraints_checkpoint.json'))
    # aggregate pass@5 per (model, N)
    from collections import defaultdict
    agg = defaultdict(list)
    levels_seen = set()
    for cell in ckpt.values():
        m = cell['model_label']
        n = cell['N']
        levels_seen.add(n)
        samples = cell.get('sample_results', [])
        if not samples:
            continue
        p5 = 1.0 if any(s['correct'] and s['tle_ok'] for s in samples) else 0.0
        agg[(m, n)].append(p5)

    LEVELS = sorted(levels_seen)
    MODELS_DISP = [
        ('gpt-4o-mini',      'GPT-4o-mini',      BLUE,   '-',  'o'),
        ('gpt-4o',           'GPT-4o',           GREEN,  '--', 's'),
        ('claude-haiku-4-5', 'Haiku 4.5',        ORANGE, '-',  '^'),
        ('gpt-5.4',          'GPT-5.4 (4.1)',    RED,    '--', 'D'),
        ('claude-opus-4-5',  'Opus 4.5',         '#6A329F','-.','v'),
    ]
    EXCLUDED = {'gemini-2.5-flash'}

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.4))

    plotted = []
    for mkey, disp, color, ls, marker in MODELS_DISP:
        if mkey in EXCLUDED: continue
        vals = []
        for l in LEVELS:
            vs = agg.get((mkey, l), [])
            if not vs:
                vals.append(None)
            else:
                vals.append(sum(vs)/len(vs)*100)
        # jitter overlapping 100% lines for visibility
        if all(v == 100 for v in vals if v is not None):
            jitter = {'GPT-5.4 (4.1)': +1.2, 'Opus 4.5': -1.2}.get(disp, 0)
            vals_plot = [v + jitter if v is not None else None for v in vals]
        else:
            vals_plot = vals
        ax.plot(LEVELS, vals_plot, color=color, linestyle=ls, marker=marker,
                markersize=4.5, linewidth=1.6, label=disp)
        plotted.append((disp, vals))

    ax.set_xscale('log')
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f'{int(l/1000)}K' for l in LEVELS])
    ax.minorticks_off()
    ax.set_xlabel('Constraint magnitude $N$')
    ax.set_ylabel('pass@5 (%)')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', framealpha=0.88, ncol=2, columnspacing=0.8,
              title='±1.2pp jitter on flat 100% lines')

    fig.tight_layout(pad=0.4)
    for ext in ['pdf','png']:
        fig.savefig(FIGS / f'fig_consens.{ext}', dpi=300 if ext=='png' else None, bbox_inches='tight')
    plt.close(fig)
    print('[fig_consens] models plotted:', [d for d,_ in plotted])
    for d, vals in plotted:
        print(f'  {d}: {vals}')
    print('[fig_consens] saved to', FIGS / 'fig_consens.{pdf,png}')

if __name__ == '__main__':
    fig_contamination()
    print()
    fig_consens()
