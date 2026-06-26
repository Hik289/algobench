#!/usr/bin/env python3
"""Recompute all paper tables from results/multimodel_results.json.
Director utility script: pure aggregation, no ML code.

Produces:
  analysis/table_main.json          # tab:main
  analysis/table_optt_opts.json     # tab:optt_opts
  analysis/table_strategy.json      # tab:strategy_cross
  analysis/table_breakdown.json     # tab:breakdown (GPT-4o per-operator on ALL52)
  analysis/tier_analysis.json       # Orig-11 / Hard-9 / Classic-32 gen gaps
  analysis/no_gap_problems.json     # Classic-32 problems where Src≈Shf
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RES  = json.load(open(ROOT / 'results/multimodel_results.json'))
OUT  = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)

MODELS = [
    'gpt-4o-mini',
    'gpt-4o',
    'claude-haiku-4-5',
    'gemini-2.5-flash',
    'gpt-5.4',
    'claude-opus-4-5',
]
DISPLAY = {
    'gpt-4o-mini':       'GPT-4o-mini',
    'gpt-4o':            'GPT-4o',
    'claude-haiku-4-5':  'Claude Haiku 4.5',
    'gemini-2.5-flash':  'Gemini 2.5 Flash',
    'gpt-5.4':           'GPT-5.4',
    'claude-opus-4-5':   'Claude Opus 4.5',
}

ORIG11 = ['CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002','GT001','GT002','GT003']
HARD9  = ['CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3']
O20    = ORIG11 + HARD9

# infer all problem IDs
ALL_PIDS = sorted({p for k in RES for p in RES[k]})
CLASSIC = sorted(set(ALL_PIDS) - set(ORIG11) - set(HARD9))

# Load benchmark to find which pids have trap_patterns labeled
import json as _json
TRAP_LABELED = sorted({
    p['id'] for p in (_json.loads(l) for l in open(ROOT / 'data/final_benchmark.jsonl'))
    if p.get('trap_patterns')
})  # 25 pids: ORIG11 + HARD9 + 5 extras (CS013,CS015,GT004,OP007,OP008)

def get_val(key, pid, mode, field):
    e = RES.get(key, {}).get(pid)
    if not isinstance(e, dict): return None
    m = e.get(mode)
    if not isinstance(m, dict): return None
    return m.get(field)

def mean_field(key, mode, field, restrict_pids=None, treat_zero_as_missing=False):
    pids = list(RES.get(key, {}).keys()) if restrict_pids is None else [
        p for p in restrict_pids if p in RES.get(key, {})
    ]
    vals = []
    for p in pids:
        v = get_val(key, p, mode, field)
        if v is None: continue
        vals.append(float(v))
    if not vals: return None, 0
    mean_pct = (sum(vals)/len(vals))*100
    # treat_zero_as_missing: ALL zero → None (full pipeline bug for GPT-5.4 / Opus-4.5)
    if treat_zero_as_missing and field == 'opt_t' and mean_pct == 0.0 and all(v == 0.0 for v in vals):
        return None, len(vals)
    return mean_pct, len(vals)

def get_operator(pid):
    """Map pid prefix to operator. CS_H1 → CS, GT001 → GT, etc."""
    if pid.startswith('CS'): return 'CS'
    if pid.startswith('SD'): return 'SD'
    if pid.startswith('OP'): return 'OP'
    if pid.startswith('GT'): return 'GT'
    return 'Unknown'

# ============================================================
# Table 1: tab:main (Direct, ALL52)
# ============================================================
table_main = []
for m in MODELS:
    key = f'{m}__direct'
    row = {'model': DISPLAY[m]}
    row['orig_p1'],     row['orig_p1_n']     = mean_field(key, 'original', 'pass_at_1')
    row['orig_opt_t'],  row['orig_opt_t_n']  = mean_field(key, 'original', 'opt_t', treat_zero_as_missing=True)
    row['shf_p1'],      row['shf_p1_n']      = mean_field(key, 'shifted', 'pass_at_1')
    row['shf_p5'],      row['shf_p5_n']      = mean_field(key, 'shifted', 'pass_at_5')
    row['shf_opt_t'],   row['shf_opt_t_n']   = mean_field(key, 'shifted', 'opt_t')
    # trap_rate ONLY over labeled subset (25 pids), C27 unlabeled would dilute
    row['shf_trap'],    row['shf_trap_n']    = mean_field(key, 'shifted', 'trap_rate', restrict_pids=TRAP_LABELED)
    table_main.append(row)
with open(OUT / 'table_main.json', 'w') as f:
    json.dump(table_main, f, indent=2)

# ============================================================
# Table 2: tab:optt_opts (Direct, ALL52, shifted only)
# ============================================================
table_oo = []
for m in MODELS:
    key = f'{m}__direct'
    row = {'model': DISPLAY[m]}
    row['p1'],   row['p1_n']   = mean_field(key, 'shifted', 'pass_at_1')
    row['p5'],   row['p5_n']   = mean_field(key, 'shifted', 'pass_at_5')
    row['optt'], row['optt_n'] = mean_field(key, 'shifted', 'opt_t')
    row['opts'], row['opts_n'] = mean_field(key, 'shifted', 'opt_s')
    if row['optt'] is not None and row['opts'] is not None:
        row['gap_p5_optt'] = row['p5'] - row['optt']
        row['delta_t_s']   = row['optt'] - row['opts']
        row['abs_delta']   = abs(row['delta_t_s'])
    table_oo.append(row)
with open(OUT / 'table_optt_opts.json', 'w') as f:
    json.dump(table_oo, f, indent=2)

# ============================================================
# Table 3: tab:strategy_cross (Direct n=O20=20, CoT/RAG n=17)
# ============================================================
table_sc = []
for m in MODELS:
    row = {'model': DISPLAY[m]}
    # Direct restricted to O20 (which is fully trap-labeled)
    p1d, nd = mean_field(f'{m}__direct', 'shifted', 'pass_at_1', restrict_pids=O20)
    trd, _  = mean_field(f'{m}__direct', 'shifted', 'trap_rate', restrict_pids=O20)  # O20 ⊂ TRAP_LABELED
    # CoT and RAG use existing 17-question splits (whatever pids are there)
    p1c, nc = mean_field(f'{m}__cot', 'shifted', 'pass_at_1')
    trc, _  = mean_field(f'{m}__cot', 'shifted', 'trap_rate')
    p1r, nr = mean_field(f'{m}__rag', 'shifted', 'pass_at_1')
    trr, _  = mean_field(f'{m}__rag', 'shifted', 'trap_rate')
    row['dir_p1'] = p1d; row['dir_n'] = nd; row['dir_trap'] = trd
    row['cot_p1'] = p1c; row['cot_n'] = nc; row['cot_trap'] = trc
    row['rag_p1'] = p1r; row['rag_n'] = nr; row['rag_trap'] = trr
    table_sc.append(row)
with open(OUT / 'table_strategy.json', 'w') as f:
    json.dump(table_sc, f, indent=2)

# ============================================================
# Table 4: tab:breakdown (GPT-4o per-operator, ALL52 direct)
# ============================================================
breakdown = []
key = 'gpt-4o__direct'
for op in ['CS', 'SD', 'OP', 'GT']:
    op_pids = [p for p in RES.get(key, {}) if get_operator(p) == op]
    row = {'operator': op, 'n_problems': len(op_pids)}
    row['p1'], _   = mean_field(key, 'shifted', 'pass_at_1', restrict_pids=op_pids)
    row['p5'], _   = mean_field(key, 'shifted', 'pass_at_5', restrict_pids=op_pids)
    row['optt'], _ = mean_field(key, 'shifted', 'opt_t',     restrict_pids=op_pids)
    row['trap'], _ = mean_field(key, 'shifted', 'trap_rate', restrict_pids=op_pids)
    breakdown.append(row)
with open(OUT / 'table_breakdown.json', 'w') as f:
    json.dump(breakdown, f, indent=2)

# ============================================================
# Tier analysis: Orig-11 / Hard-9 / Classic-32
# ============================================================
tier_data = {}
for tier_name, pids in [('Orig-11', ORIG11), ('Hard-9', HARD9), ('Classic-32', CLASSIC)]:
    per_model = []
    src_means = []
    shf_means = []
    for m in MODELS:
        key = f'{m}__direct'
        src, n_src = mean_field(key, 'original', 'pass_at_1', restrict_pids=pids)
        shf, n_shf = mean_field(key, 'shifted',  'pass_at_1', restrict_pids=pids)
        if src is not None and shf is not None:
            per_model.append({'model': m, 'src_p1': src, 'shf_p1': shf,
                              'delta': shf - src, 'n_src': n_src, 'n_shf': n_shf})
            src_means.append(src); shf_means.append(shf)
    tier_data[tier_name] = {
        'pids': pids,
        'per_model': per_model,
        'src_mean_acrossmodels': sum(src_means)/len(src_means) if src_means else None,
        'shf_mean_acrossmodels': sum(shf_means)/len(shf_means) if shf_means else None,
        'delta_mean': (sum(shf_means)/len(shf_means) - sum(src_means)/len(src_means))
                      if src_means else None,
    }
with open(OUT / 'tier_analysis.json', 'w') as f:
    json.dump(tier_data, f, indent=2)

# ============================================================
# No-gap classic problems
# ============================================================
no_gap = []
for p in CLASSIC:
    src_pass, shf_pass, n_seen = 0, 0, 0
    for m in MODELS:
        key = f'{m}__direct'
        if p not in RES.get(key, {}): continue
        n_seen += 1
        if RES[key][p].get('original', {}).get('pass_at_1'): src_pass += 1
        if RES[key][p].get('shifted',  {}).get('pass_at_1'): shf_pass += 1
    if n_seen >= 4 and shf_pass >= n_seen - 1 and src_pass >= n_seen - 1:
        no_gap.append({'pid': p, 'n_models': n_seen,
                       'src_pass': src_pass, 'shf_pass': shf_pass})
with open(OUT / 'no_gap_problems.json', 'w') as f:
    json.dump(no_gap, f, indent=2)

# ============================================================
# Print summary
# ============================================================
def f(v, digits=1):
    if v is None: return '---'
    return f'{v:.{digits}f}'

print("=" * 100)
print("TABLE 1: tab:main (Direct, ALL52)")
print("=" * 100)
print(f"{'Model':<22} {'OrigP1':<10} {'OrigOpT*':<10} {'ShfP1':<10} {'ShfP5':<10} {'ShfOpT':<10} {'Trap':<8}")
for r in table_main:
    print(f"{r['model']:<22} "
          f"{f(r['orig_p1']):<10} "
          f"{f(r['orig_opt_t']):<10} "
          f"{f(r['shf_p1']):<10} "
          f"{f(r['shf_p5']):<10} "
          f"{f(r['shf_opt_t']):<10} "
          f"{f(r['shf_trap']):<8}")
print("* OrigOpT for GPT-5.4 and Claude Opus 4.5 = '---' (pipeline bug, all 0.0)")

print()
print("=" * 100)
print("TABLE 2: tab:optt_opts (Direct, ALL52, shifted)")
print("=" * 100)
print(f"{'Model':<22} {'p@1':<8} {'p@5':<8} {'OptT':<8} {'OptS':<8} {'|Δ(OT-OS)|':<12}")
for r in table_oo:
    print(f"{r['model']:<22} "
          f"{f(r['p1']):<8} {f(r['p5']):<8} {f(r['optt']):<8} {f(r['opts']):<8} {f(r.get('abs_delta')):<12}")

print()
print("=" * 100)
print("TABLE 3: tab:strategy_cross (Direct n=O20, CoT/RAG n=17)")
print("=" * 100)
print(f"{'Model':<22} {'Dir_p@1':<10} {'CoT_p@1':<10} {'RAG_p@1':<10} {'Dir_Trap':<10} {'CoT_Trap':<10} {'RAG_Trap':<10}")
for r in table_sc:
    print(f"{r['model']:<22} "
          f"{f(r['dir_p1'])+'('+str(r['dir_n'])+')':<10} "
          f"{f(r['cot_p1'])+'('+str(r['cot_n'])+')':<10} "
          f"{f(r['rag_p1'])+'('+str(r['rag_n'])+')':<10} "
          f"{f(r['dir_trap']):<10} {f(r['cot_trap']):<10} {f(r['rag_trap']):<10}")

print()
print("=" * 100)
print("TABLE 4: tab:breakdown (GPT-4o per-operator, ALL52 direct, shifted)")
print("=" * 100)
print(f"{'Operator':<22} {'n':<5} {'p@1':<8} {'p@5':<8} {'OptT':<8} {'Trap':<8}")
for r in breakdown:
    print(f"{r['operator']:<22} {r['n_problems']:<5} {f(r['p1']):<8} {f(r['p5']):<8} {f(r['optt']):<8} {f(r['trap']):<8}")

print()
print("=" * 100)
print("TIER ANALYSIS: Generalization Gap (Src vs Shf p@1, averaged across 6 models)")
print("=" * 100)
print(f"{'Tier':<12} {'n_pids':<8} {'Src_mean':<12} {'Shf_mean':<12} {'Δgap':<10}")
for t, info in tier_data.items():
    print(f"{t:<12} {len(info['pids']):<8} "
          f"{f(info['src_mean_acrossmodels']):<12} "
          f"{f(info['shf_mean_acrossmodels']):<12} "
          f"{f(info['delta_mean']):<10}")

print()
print(f"NO-GAP CLASSIC PROBLEMS (Src≈Shf, both high): {len(no_gap)} problems")
for r in no_gap:
    print(f"  {r['pid']}: src_pass={r['src_pass']}/{r['n_models']}, shf_pass={r['shf_pass']}/{r['n_models']}")

print()
print(f"All JSON outputs saved to: {OUT}/")
