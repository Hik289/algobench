#!/usr/bin/env python3
"""Compute structural shift metrics (Director utility — pure aggregation, no ML code).

Outputs:
  analysis/shift_metrics_per_problem.json   # per-problem features
  analysis/table_shift.json                 # tab:shift aggregate
  analysis/table_shift_op.json              # tab:shift_op per-operator
  analysis/table_shift_corr.json            # tab:shift_corr Spearman with perf
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = list(json.loads(l) for l in open(ROOT / 'data/final_benchmark.jsonl'))
RES = json.load(open(ROOT / 'results/multimodel_results.json'))
OUT = ROOT / 'analysis'

ORIG11 = ['CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002','GT001','GT002','GT003']
HARD9 = ['CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3']
MODELS = ['gpt-4o-mini','gpt-4o','claude-haiku-4-5','gemini-2.5-flash','gpt-5.4','claude-opus-4-5']

# ---- Algorithm distance taxonomy ------------------------------------------
# Hand-coded grouping into algorithm families. Same family → Same(0); related
# family → Near(1); different family entirely → Far(2).
FAMILY = {
    # Range queries / static arrays
    'prefix_sum': 'static_range',
    'cumulative_sum': 'static_range',
    # BIT / segment trees (dynamic range)
    'fenwick_tree_BIT': 'dynamic_range',
    'lazy_segment_tree': 'dynamic_range',
    'segment_tree': 'dynamic_range',
    'merge_sort_BIT': 'dynamic_range',
    'prefix_sum_with_merge_sort_or_BIT': 'dynamic_range',
    # DSU
    'dsu': 'dsu',
    'online_dsu_with_size': 'dsu',
    'offline_dynamic_connectivity_dsu_rollback': 'dsu',
    'dsu_rollback': 'dsu',
    # MST / Kruskal
    'kruskal_or_prim': 'mst',
    'kruskal_plus_lca_on_kruskal_tree': 'mst',
    # Shortest path
    'bellman_ford_or_floyd': 'shortest_path',
    'dijkstra_with_heap': 'shortest_path',
    'bfs_per_query': 'graph_traversal',
    'bfs': 'graph_traversal',
    'dfs': 'graph_traversal',
    # DP
    'dp_O(nm)': 'dp_2d',
    'LCS_as_LIS_via_inverse_permutation': 'dp_lis',
    'dp_sorted_intervals_with_switch_count': 'dp_1d',
    'dp_with_type_cooldown_state': 'dp_1d',
    # Greedy
    'kadane': 'greedy_1d',
    'greedy_earliest_finish': 'greedy_1d',
    'greedy_by_profit_with_dsu': 'greedy_dsu',
    'two_pointer': 'two_pointer',
    'sliding_window': 'two_pointer',
}
# Map family-pairs to distance class. Default Far if cross-family.
NEAR_PAIRS = [
    {'static_range', 'dynamic_range'},
    {'graph_traversal', 'shortest_path'},
    {'dp_1d', 'dp_2d'},
    {'dp_1d', 'greedy_1d'},
    {'mst', 'shortest_path'},
    {'dsu', 'mst'},
    {'greedy_1d', 'two_pointer'},
]
def algo_distance(src, tgt):
    fs = FAMILY.get(src); ft = FAMILY.get(tgt)
    if not fs or not ft: return 'Unknown'
    if fs == ft: return 'Same'
    if {fs, ft} in NEAR_PAIRS: return 'Near'
    return 'Far'

# ---- Complexity exponent mapping ------------------------------------------
EXP_MAP = [
    (r'O\(1\)', 0),
    (r'O\(log[\s\S]*n\)', 0.3),
    (r'O\(n \^? ?(?:\^|\*\*)\s*0\.5\)', 0.5),
    (r'O\(\?n\)', 0.5),
    (r"O\(sqrt", 0.5),
    (r'O\(n\)', 1.0),
    (r'O\(n\s*\+\s*[QqMm]\)', 1.0),
    (r'O\(n\s*log\s*n\)', 1.5),
    (r"O\((?:m|n)\s*log\s*(?:m|n)\)", 1.5),
    (r"O\(\(n\s*\+\s*Q\)\s*log\s*n\)", 1.5),
    (r"O\(n\^?2\)", 2.0),
    (r"O\(n\*\*2\)", 2.0),
    (r"O\(n\*\s*m\)", 2.0),
    (r"O\(nm\)", 2.0),
    (r"O\(n\^?3\)", 3.0),
    (r"O\(n!\)", float('inf')),
    (r"O\(2\^n\)", float('inf')),
]
def complexity_to_exp(s):
    if not isinstance(s, str): s = str(s)
    for pat, exp in EXP_MAP:
        if re.search(pat, s, re.IGNORECASE):
            return exp
    # Fallback: count n-power
    s2 = s.lower().replace('log', '_log_')
    if '_log_' in s2 and 'n' in s2:
        return 1.5
    if 'n' in s2 and '^2' in s:
        return 2.0
    return None

# ---- Constraint magnitude (largest numeric upper bound) -------------------
def max_constraint_mag(s):
    """Extract the largest n-bound from a constraint string, return as int.
    Heuristic: only look at the FIRST constraint (typically '1 ≤ n, Q ≤ K')."""
    if not s: return None
    # split on common separators and take the first constraint expression
    first_clause = re.split(r';|\.', s)[0]
    candidates = []
    # forms like "2×10^5", "10^9"
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[×x*]\s*10\s*\^\s*(\d+)', first_clause):
        a, b = float(m.group(1)), int(m.group(2))
        candidates.append(a * 10**b)
    for m in re.finditer(r'10\s*\^\s*(\d+)', first_clause):
        candidates.append(10**int(m.group(1)))
    # plain large integer literal
    for m in re.finditer(r'\b(\d{3,12})\b', first_clause):
        candidates.append(int(m.group(1)))
    if not candidates: return None
    # filter unreasonable values (>10^15 likely an A[i] value bound)
    candidates = [c for c in candidates if c <= 1e10]
    return max(candidates) if candidates else None

# ---- Compute per-problem features -----------------------------------------
per_prob = {}
for p in BENCH:
    src_stmt = p.get('source_statement', '')
    shf_stmt = p.get('shifted_statement', '')
    sa, sb = set(src_stmt.lower().split()), set(shf_stmt.lower().split())
    jaccard = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    len_ratio = len(shf_stmt) / max(1, len(src_stmt))
    src_mag = max_constraint_mag(p.get('source_constraints', ''))
    shf_mag = max_constraint_mag(p.get('shifted_constraints', ''))
    mag_ratio = shf_mag / src_mag if (src_mag and shf_mag) else None
    src_exp = complexity_to_exp(p.get('source_complexity', {}).get('time', ''))
    tgt_exp = complexity_to_exp(p.get('target_complexity', {}).get('time', ''))
    d_exp = (tgt_exp - src_exp) if (src_exp is not None and tgt_exp is not None) else None
    src_algo = p.get('source_reference_algorithm')
    tgt_algo = p.get('target_algorithm')
    dist = algo_distance(src_algo, tgt_algo) if src_algo else None
    per_prob[p['id']] = {
        'operator': p['operator'],
        'jaccard': jaccard,
        'len_ratio': len_ratio,
        'src_mag': src_mag,
        'shf_mag': shf_mag,
        'mag_ratio': mag_ratio,
        'src_exp': src_exp,
        'tgt_exp': tgt_exp,
        'd_exp': d_exp,
        'src_algo': src_algo,
        'tgt_algo': tgt_algo,
        'alg_distance': dist,
        'tier': 'Orig-11' if p['id'] in ORIG11 else ('Hard-9' if p['id'] in HARD9 else 'Classic-32'),
    }

with open(OUT / 'shift_metrics_per_problem.json', 'w') as f:
    json.dump(per_prob, f, indent=2)

# ---- tab:shift (aggregate over all problems) -------------------------------
import math as _math
def agg(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and _math.isinf(v))]
    if not vs: return None, None, 0
    return statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0, len(vs)

vals_jac    = [d['jaccard']    for d in per_prob.values()]
vals_lenr   = [d['len_ratio']  for d in per_prob.values()]
vals_magr   = [d['mag_ratio']  for d in per_prob.values()]
vals_dexp   = [d['d_exp']      for d in per_prob.values()]
vals_dist   = [d['alg_distance'] for d in per_prob.values()]
from collections import Counter
dist_counts = Counter(d['alg_distance'] for d in per_prob.values())

m_j, s_j, n_j = agg(vals_jac)
m_l, s_l, n_l = agg(vals_lenr)
m_m, s_m, n_m = agg(vals_magr)
m_d, s_d, n_d = agg(vals_dexp)
table_shift = {
    'jaccard':  {'mean': m_j, 'std': s_j, 'n': n_j},
    'len_ratio':{'mean': m_l, 'std': s_l, 'n': n_l},
    'mag_ratio':{'mean': m_m, 'std': s_m, 'n': n_m},
    'd_exp':    {'mean': m_d, 'std': s_d, 'n': n_d},
    'distance_distribution': dict(dist_counts),
}
with open(OUT / 'table_shift.json', 'w') as f:
    json.dump(table_shift, f, indent=2)

# ---- tab:shift_op (per-operator) ------------------------------------------
table_shift_op = {}
for op in ['CS','SD','OP','GT']:
    sub = [d for d in per_prob.values() if d['operator'] == op]
    j = agg([d['jaccard'] for d in sub])
    l = agg([d['len_ratio'] for d in sub])
    de = agg([d['d_exp'] for d in sub])
    dist_c = Counter(d['alg_distance'] for d in sub)
    most = dist_c.most_common(1)[0][0] if dist_c else 'Unknown'
    table_shift_op[op] = {
        'jaccard_mean': j[0],
        'len_ratio_mean': l[0],
        'd_exp_mean': de[0],
        'most_common_distance': most,
        'n': len(sub),
    }
with open(OUT / 'table_shift_op.json', 'w') as f:
    json.dump(table_shift_op, f, indent=2)

# ---- tab:shift_corr (Spearman on shift metric vs avg model perf) -----------
def spearman(xs, ys):
    """Return Spearman ρ given paired lists; skip None entries."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5: return None, len(pairs)
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    def rank(arr):
        idx = sorted(range(len(arr)), key=lambda i: arr[i])
        r = [0]*len(arr)
        for i, k in enumerate(idx):
            r[k] = i+1
        return r
    rx = rank(xs); ry = rank(ys)
    n = len(rx)
    d2 = sum((a-b)**2 for a, b in zip(rx, ry))
    rho = 1 - 6*d2 / (n*(n*n-1))
    return rho, n

# Compute per-problem average pass@1, trap_rate, opt_t across models
def avg_perf(pid, field, mode='shifted'):
    vals = []
    for m in MODELS:
        e = RES.get(f'{m}__direct', {}).get(pid, {}).get(mode, {})
        if field in e and e[field] is not None:
            vals.append(float(e[field]))
    return sum(vals)/len(vals) if vals else None

# Encode alg distance as numeric
DIST_NUM = {'Same': 0.0, 'Near': 0.5, 'Far': 1.0, 'Unknown': None}

# Build paired data
pid_list = list(per_prob.keys())
perf_p1  = [avg_perf(p, 'pass_at_1') for p in pid_list]
perf_tr  = [avg_perf(p, 'trap_rate') for p in pid_list]
perf_ot  = [avg_perf(p, 'opt_t') for p in pid_list]

# log(constraint ratio)
import math
xs_jac    = [per_prob[p]['jaccard']    for p in pid_list]
xs_logmag = [math.log(per_prob[p]['mag_ratio']) if per_prob[p]['mag_ratio'] else None for p in pid_list]
xs_dexp   = [per_prob[p]['d_exp']      for p in pid_list]
xs_dist   = [DIST_NUM[per_prob[p]['alg_distance']] if per_prob[p]['alg_distance'] in DIST_NUM else None for p in pid_list]

table_shift_corr = {}
for label, xs in [('jaccard', xs_jac), ('log_mag_ratio', xs_logmag), ('d_exp', xs_dexp), ('alg_distance', xs_dist)]:
    table_shift_corr[label] = {
        'rho_p1':   {'rho': (r:=spearman(xs, perf_p1))[0], 'n': r[1]},
        'rho_trap': {'rho': (r:=spearman(xs, perf_tr))[0], 'n': r[1]},
        'rho_opt_t':{'rho': (r:=spearman(xs, perf_ot))[0], 'n': r[1]},
    }
with open(OUT / 'table_shift_corr.json', 'w') as f:
    json.dump(table_shift_corr, f, indent=2)

# ---- Print summary --------------------------------------------------------
print("=" * 90)
print("TABLE: tab:shift (structural shift metrics, n=52)")
print("=" * 90)
print(f"  Text Jaccard:        mean={m_j:.3f}, std={s_j:.3f}, n={n_j}")
print(f"  Length ratio:        mean={m_l:.2f}×, std={s_l:.2f}, n={n_l}")
print(f"  Constraint mag ratio:mean={m_m:.0f}×, std={s_m:.0f}, n={n_m}")
print(f"  Δexp (complexity):   mean={m_d:+.2f}, std={s_d:.2f}, n={n_d}")
print(f"  Alg distance distribution:")
total_dist = sum(dist_counts.values())
for k, v in sorted(dist_counts.items(), key=lambda x: -(x[1] or 0)):
    label = k if k is not None else 'None(no algo metadata)'
    pct = v/total_dist*100 if total_dist else 0.0
    print(f"    {str(label):<28}: {v}/{total_dist} ({pct:.1f}%)")

print("\n" + "=" * 90)
print("TABLE: tab:shift_op (per-operator profiles)")
print("=" * 90)
print(f"{'Op':<6} {'Jaccard':<10} {'Len':<8} {'Δexp':<8} {'Most Dist':<12} {'n':<4}")
for op, d in table_shift_op.items():
    print(f"{op:<6} {d['jaccard_mean']:.3f}     {d['len_ratio_mean']:.2f}×   {d['d_exp_mean']:+.2f}    {d['most_common_distance']:<12} {d['n']}")

print("\n" + "=" * 90)
print("TABLE: tab:shift_corr (Spearman correlations, n=avg-across-models)")
print("=" * 90)
print(f"{'Metric':<20} {'ρ(p@1)':<10} {'ρ(trap)':<10} {'ρ(opt_t)':<10}")
for label, d in table_shift_corr.items():
    r1 = d['rho_p1']['rho']; r2 = d['rho_trap']['rho']; r3 = d['rho_opt_t']['rho']
    r1s = f'{r1:+.3f}' if r1 is not None else '---'
    r2s = f'{r2:+.3f}' if r2 is not None else '---'
    r3s = f'{r3:+.3f}' if r3 is not None else '---'
    print(f"{label:<20} {r1s:<10} {r2s:<10} {r3s:<10}")

print(f"\n→ JSONs saved to {OUT}/")
