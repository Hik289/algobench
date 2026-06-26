#!/usr/bin/env python3
"""Director Level-1 utility analysis for all missing-data subsections.
Pure data aggregation from benchmark JSON + results JSON. No ML code.

Outputs 6 JSONs in analysis/:
  - algo_taxonomy.json           (algorithm family classification, target coverage ≥80%)
  - table_shift.json             (refined, was already computed v1)
  - table_shift_op.json          (refined per-operator)
  - table_shift_corr.json        (refined Spearman with proper n)
  - table_gen_gap_dist.json      (generalization gap by algorithm distance, all 6 models)
  - fig_transition_data.json     (algo-pair pass@1 / opt_t)
  - fig_rag_severity_data.json   (RAG split by shift severity, n=17)
  - fig_gapt_data.json           (effective exponent gap by operator using est_t)
"""
import json
import re
import math
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
BENCH = list(json.loads(l) for l in open(ROOT / 'data/final_benchmark.jsonl'))
RES = json.load(open(ROOT / 'results/multimodel_results.json'))
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)

ORIG11 = ['CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002','GT001','GT002','GT003']
HARD9  = ['CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3']
O20    = ORIG11 + HARD9
MODELS = ['gpt-4o-mini','gpt-4o','claude-haiku-4-5','gemini-2.5-flash','gpt-5.4','claude-opus-4-5']

# =============================================================
# (1) Algorithm Taxonomy (full coverage, including C27 inference)
# =============================================================
# Hand-curated families. Algos not listed default to 'misc'.
ALGO_TO_FAMILY = {
    # static range / prefix sums
    'prefix_sum': 'static_range',
    'prefix_sum_O_n_plus_q': 'static_range',
    'cumulative_sum': 'static_range',
    'extra_array': 'static_range',
    # dynamic range (BIT/segment trees)
    'fenwick_tree_BIT': 'dynamic_range',
    'fenwick_tree_range_update_range_query': 'dynamic_range',
    'lazy_segment_tree': 'dynamic_range',
    'segment_tree': 'dynamic_range',
    'merge_sort_BIT': 'dynamic_range',
    'prefix_sum_with_merge_sort_or_BIT': 'dynamic_range',
    'sparse_table_rmq': 'dynamic_range',
    'BIT + coordinate compression': 'dynamic_range',
    'Fenwick Tree / BIT': 'dynamic_range',
    'Sparse table': 'dynamic_range',
    'Merge sort inversion count': 'dynamic_range',
    # DSU
    'dsu': 'dsu',
    'online_dsu_with_size': 'dsu',
    'offline_dynamic_connectivity_dsu_rollback': 'dsu',
    'dsu_rollback': 'dsu',
    'Union-Find path compression + rank': 'dsu',
    # MST / Kruskal-based
    'kruskal_or_prim': 'mst',
    'kruskal_plus_lca_on_kruskal_tree': 'mst',
    # Shortest path
    'bellman_ford_or_floyd': 'shortest_path',
    'dijkstra': 'shortest_path',
    'dijkstra_with_heap': 'shortest_path',
    'bellman_ford_O_nm': 'shortest_path',
    'Dijkstra with priority queue': 'shortest_path',
    # Graph traversal / connectivity
    'bfs': 'graph_traversal',
    'bfs_per_query': 'graph_traversal',
    'dfs': 'graph_traversal',
    'dfs_backtrack': 'graph_traversal',
    'bfs_dfs_flood_fill': 'graph_traversal',
    'BFS 2-coloring all components': 'graph_traversal',
    "Tarjan's bridge finding": 'graph_traversal',
    "Kosaraju's SCC + condensation analysis": 'graph_traversal',
    "Kahn's with min-heap": 'graph_traversal',
    # DP 1-D / 2-D / specialized
    'dp_O(nm)': 'dp_2d',
    'dp_grid_paths': 'dp_2d',
    '2D knapsack DP dp[w][v]': 'dp_2d',
    'dp_01_knapsack': 'dp_2d',
    '01_knapsack_dp': 'dp_2d',
    'Unbounded knapsack DP': 'dp_2d',
    'wagner_fischer_dp': 'dp_2d',
    'dynamic_programming_O_kS': 'dp_2d',
    'LCS_as_LIS_via_inverse_permutation': 'dp_lis',
    'dp_sorted_intervals_with_switch_count': 'dp_1d',
    'dp_with_type_cooldown_state': 'dp_1d',
    'dp_house_robber': 'dp_1d',
    'dp_with_binary_search': 'dp_1d',
    'DP + binary search on end times': 'dp_1d',
    'Patience sorting / binary search': 'dp_lis',
    'O(n) DP with precomputed left/right max': 'dp_1d',
    # Greedy
    'kadane': 'greedy_1d',
    'greedy_earliest_finish': 'greedy_1d',
    'greedy_earliest_end': 'greedy_1d',
    'greedy_largest_first': 'greedy_1d',
    'greedy_ratio': 'greedy_1d',
    'greedy_by_profit_with_dsu': 'greedy_dsu',
    'greedy_by_value_per_weight': 'greedy_1d',
    'greedy_alternate': 'greedy_1d',
    'greedy_max_reach': 'greedy_1d',
    'Sweep line + max-heap': 'greedy_heap',
    'Two heaps (max-heap + min-heap)': 'heap',
    'Monotonic deque': 'monotonic',
    'Auxiliary min-stack': 'monotonic',
    # String matching
    'KMP': 'string_match',
    'KMP_O_n_plus_m': 'string_match',
    "Manacher's algorithm": 'string_match',
    # Brute force
    'naive_recursion': 'brute_force',
    'naive_linear_scan': 'brute_force',
    'naive_scan': 'brute_force',
    'brute_force_O_n2': 'brute_force',
    'brute_force_O_nm': 'brute_force',
    'recursive_backtracking': 'brute_force',
    'hash_table_O_n': 'hash',
    # Combinatorial / math
    '3×3 matrix exponentiation': 'matrix_exp',
    'Matrix exponentiation [[3,-1],[1,0]]': 'matrix_exp',
    'Extended GCD + Chinese Remainder Theorem': 'number_theory',
    "König's theorem: min VC = max bipartite matching": 'flow_matching',
    'Binary search on partition': 'binary_search',
    'triple_reverse': 'array_ops',
    'All degrees even + connected': 'graph_property',
}

# Near family pairs — pairs whose distance is "Near", not "Far"
NEAR_FAMILIES = [
    {'static_range', 'dynamic_range'},
    {'graph_traversal', 'shortest_path'},
    {'shortest_path', 'mst'},
    {'mst', 'dsu'},
    {'dp_1d', 'dp_2d'},
    {'dp_1d', 'greedy_1d'},
    {'dp_2d', 'dp_lis'},
    {'dp_1d', 'dp_lis'},
    {'greedy_1d', 'monotonic'},
    {'greedy_1d', 'heap'},
    {'greedy_1d', 'greedy_heap'},
    {'brute_force', 'dp_2d'},
    {'brute_force', 'dp_1d'},
    {'brute_force', 'greedy_1d'},
    {'monotonic', 'heap'},
    {'monotonic', 'greedy_heap'},
    {'dsu', 'greedy_dsu'},
    {'mst', 'greedy_dsu'},
    {'static_range', 'array_ops'},
    {'array_ops', 'monotonic'},
]

def family_of(algo):
    if algo is None: return None
    f = ALGO_TO_FAMILY.get(algo)
    if f is not None: return f
    # Fuzzy match for unknowns
    al = algo.lower()
    if 'segment_tree' in al or 'fenwick' in al or 'BIT' in algo.upper(): return 'dynamic_range'
    if 'prefix' in al or 'cumsum' in al: return 'static_range'
    if 'dsu' in al or 'union_find' in al or 'union-find' in al: return 'dsu'
    if 'kruskal' in al or 'prim' in al: return 'mst'
    if 'dijkstra' in al or 'bellman' in al or 'floyd' in al: return 'shortest_path'
    if 'bfs' in al or 'dfs' in al: return 'graph_traversal'
    if 'knapsack' in al or 'dp_' in al or al.startswith('dp '): return 'dp_2d'
    if 'greedy' in al: return 'greedy_1d'
    if 'brute' in al or 'naive' in al or 'backtrack' in al: return 'brute_force'
    if 'kmp' in al or 'manacher' in al or 'aho' in al: return 'string_match'
    if 'matrix_exp' in al or 'matrix exp' in al: return 'matrix_exp'
    if 'heap' in al: return 'heap'
    if 'sparse_table' in al or 'sparse table' in al: return 'dynamic_range'
    if 'binary_search' in al or 'binary search' in al: return 'binary_search'
    if 'monotonic' in al: return 'monotonic'
    return 'misc'

def algo_distance(src, tgt):
    fs = family_of(src); ft = family_of(tgt)
    if fs is None or ft is None: return 'Unknown'
    if fs == 'misc' or ft == 'misc': return 'Unknown'
    if fs == ft: return 'Same'
    if {fs, ft} in NEAR_FAMILIES: return 'Near'
    return 'Far'

# Infer source algo for C27 from source_complexity + source_statement keywords
def infer_source_algo(p):
    """Heuristic source algo inference for problems without source_reference_algorithm."""
    src_cx = p.get('source_complexity', {}).get('time', '').lower()
    src_stmt = p.get('source_statement', '').lower()
    if 'n+q' in src_cx or 'n + q' in src_cx:
        return 'prefix_sum'  # static_range
    if 'log n' in src_cx and 'n log n' not in src_cx:
        return 'binary_search_or_log'
    if 'n log n' in src_cx:
        return 'sort_or_BIT'
    if re.search(r'n\^?2\b|nm\b|n\*m\b', src_cx):
        if 'dp' in src_stmt or 'subseq' in src_stmt or 'edit distance' in src_stmt:
            return 'dp_O(nm)'
        return 'brute_force_O_n2'
    if 'v+e' in src_cx or 'm+n' in src_cx or '|v|+|e|' in src_cx:
        if 'shortest' in src_stmt or 'distance' in src_stmt: return 'bfs'
        return 'graph_traversal_generic'
    if 'o(n)' in src_cx:
        return 'naive_scan_O_n'
    if '2^n' in src_cx or 'n!' in src_cx or 'exp' in src_cx:
        return 'brute_force'
    return None

# Build per-problem records
per_problem = {}
total_with_algo = 0
total_classified = 0
for p in BENCH:
    pid = p['id']
    src_algo = p.get('source_reference_algorithm')
    if src_algo is None:
        src_algo = infer_source_algo(p)
        inferred = True
    else:
        inferred = False
    tgt_algo = p.get('target_algorithm')
    dist = algo_distance(src_algo, tgt_algo)
    per_problem[pid] = {
        'operator': p['operator'],
        'src_algo': src_algo,
        'src_algo_inferred': inferred,
        'tgt_algo': tgt_algo,
        'src_family': family_of(src_algo),
        'tgt_family': family_of(tgt_algo),
        'distance': dist,
    }
    if src_algo and tgt_algo: total_with_algo += 1
    if dist != 'Unknown': total_classified += 1

coverage = total_classified / len(BENCH)

algo_taxonomy = {
    'method': 'Hand-curated family map for paper algos + heuristic inference for C27',
    'families': sorted(set(ALGO_TO_FAMILY.values())),
    'algo_to_family': ALGO_TO_FAMILY,
    'near_family_pairs': [sorted(s) for s in NEAR_FAMILIES],
    'per_problem': per_problem,
    'coverage_rate': coverage,
    'n_with_both_algos': total_with_algo,
    'n_classified': total_classified,
    'n_total': len(BENCH),
}
with open(OUT / 'algo_taxonomy.json', 'w') as f:
    json.dump(algo_taxonomy, f, indent=2)

# =============================================================
# (2) Distance distribution updates (for tab:shift)
# =============================================================
dist_counts = Counter(d['distance'] for d in per_problem.values())
print(f"Distance distribution (n=52, coverage={coverage*100:.1f}%):")
for k, v in sorted(dist_counts.items(), key=lambda x: -x[1]):
    print(f"  {str(k):<10}: {v} ({v/52*100:.1f}%)")

# Update table_shift.json with new dist
table_shift = json.load(open(OUT / 'table_shift.json'))
table_shift['distance_distribution'] = dict(dist_counts)
table_shift['distance_coverage_rate'] = coverage
with open(OUT / 'table_shift.json', 'w') as f:
    json.dump(table_shift, f, indent=2)

# =============================================================
# (3) tab:gen_gap_dist — Gen gap by algorithm distance, per-model
# =============================================================
# For each model, for each distance class (Same / Near / Far), compute
# mean source pass@1 - mean shifted pass@1.
table_gen_gap_dist = {}
for m in MODELS:
    key = f'{m}__direct'
    by_dist = defaultdict(lambda: {'src': [], 'shf': []})
    for pid, info in per_problem.items():
        d = info['distance']
        if d == 'Unknown': continue
        e = RES.get(key, {}).get(pid, {})
        sv = e.get('original', {}).get('pass_at_1')
        tv = e.get('shifted', {}).get('pass_at_1')
        if sv is not None: by_dist[d]['src'].append(float(sv))
        if tv is not None: by_dist[d]['shf'].append(float(tv))
    row = {}
    for d in ['Same','Near','Far']:
        sv, tv = by_dist[d]['src'], by_dist[d]['shf']
        n = min(len(sv), len(tv))
        if not sv or not tv:
            row[d] = {'n': n, 'src_p1': None, 'shf_p1': None, 'delta_pp': None}
            continue
        s_mean = sum(sv)/len(sv) * 100
        t_mean = sum(tv)/len(tv) * 100
        row[d] = {'n': n, 'src_p1': s_mean, 'shf_p1': t_mean, 'delta_pp': t_mean - s_mean}
    table_gen_gap_dist[m] = row
with open(OUT / 'table_gen_gap_dist.json', 'w') as f:
    json.dump(table_gen_gap_dist, f, indent=2)

# =============================================================
# (4) fig:transition — algo-pair pass@1 / opt_t heatmap
# =============================================================
# Aggregate per (src_family, tgt_family) cell.
transition = defaultdict(lambda: {'pids': [], 'p1_vals': [], 'opt_t_vals': []})
for pid, info in per_problem.items():
    fs, ft = info['src_family'], info['tgt_family']
    if fs is None or ft is None or fs == 'misc' or ft == 'misc': continue
    transition[(fs, ft)]['pids'].append(pid)
    # avg across 6 models
    for m in MODELS:
        e = RES.get(f'{m}__direct', {}).get(pid, {}).get('shifted', {})
        if 'pass_at_1' in e:  transition[(fs, ft)]['p1_vals'].append(float(e['pass_at_1']))
        if 'opt_t' in e:       transition[(fs, ft)]['opt_t_vals'].append(float(e['opt_t']))

fig_transition_data = []
for (fs, ft), v in transition.items():
    fig_transition_data.append({
        'src_family': fs,
        'tgt_family': ft,
        'distance': algo_distance_via_families(fs, ft) if False else (
            'Same' if fs == ft else
            ('Near' if {fs, ft} in NEAR_FAMILIES else 'Far')
        ),
        'n_problems': len(v['pids']),
        'pids': v['pids'],
        'mean_p1':    sum(v['p1_vals'])   / len(v['p1_vals'])   * 100 if v['p1_vals']   else None,
        'mean_opt_t': sum(v['opt_t_vals'])/ len(v['opt_t_vals'])* 100 if v['opt_t_vals'] else None,
    })
# sort by distance then by n_problems
fig_transition_data.sort(key=lambda x: (x['distance'] != 'Same', x['distance'] != 'Near', -x['n_problems']))
with open(OUT / 'fig_transition_data.json', 'w') as f:
    json.dump(fig_transition_data, f, indent=2)

# =============================================================
# (5) fig:rag_severity — RAG split by Jaccard severity
# =============================================================
# Load existing shift_metrics_per_problem.json for Jaccard
per_prob_metrics = json.load(open(OUT / 'shift_metrics_per_problem.json'))
# Use the 17 problems with CoT/RAG data: ORIG11 + 6 hard variants
# Get problems present in cot/rag keys
cot_pids = set()
for m in MODELS:
    cot_pids |= set(RES.get(f'{m}__cot', {}).keys())
cot_pids = sorted(cot_pids)
# median jaccard
jac_vals = [per_prob_metrics[p]['jaccard'] for p in cot_pids if p in per_prob_metrics]
median_jac = statistics.median(jac_vals) if jac_vals else 0.46
print(f"\nCoT/RAG eval set: {len(cot_pids)} problems, median Jaccard = {median_jac:.3f}")

easy_pids = [p for p in cot_pids if p in per_prob_metrics and per_prob_metrics[p]['jaccard'] >= median_jac]
hard_pids = [p for p in cot_pids if p in per_prob_metrics and per_prob_metrics[p]['jaccard'] <  median_jac]

def avg_metric(model, key_suffix, pids, field):
    vs = []
    for p in pids:
        e = RES.get(f'{model}__{key_suffix}', {}).get(p, {}).get('shifted', {})
        if field in e and e[field] is not None: vs.append(float(e[field]))
    if not vs: return None, 0
    return sum(vs)/len(vs)*100, len(vs)

fig_rag_severity = {}
for m in MODELS:
    fig_rag_severity[m] = {}
    for group_name, pids in [('easy', easy_pids), ('hard', hard_pids)]:
        d_p1, n_d  = avg_metric(m, 'direct', pids, 'pass_at_1')
        r_p1, n_r  = avg_metric(m, 'rag',    pids, 'pass_at_1')
        d_tr, _    = avg_metric(m, 'direct', pids, 'trap_rate')
        r_tr, _    = avg_metric(m, 'rag',    pids, 'trap_rate')
        delta_p1   = (r_p1 - d_p1) if (d_p1 is not None and r_p1 is not None) else None
        delta_tr   = (r_tr - d_tr) if (d_tr is not None and r_tr is not None) else None
        fig_rag_severity[m][group_name] = {
            'n': n_d,
            'direct_p1': d_p1, 'rag_p1': r_p1, 'delta_p1': delta_p1,
            'direct_trap': d_tr, 'rag_trap': r_tr, 'delta_trap': delta_tr,
        }
fig_rag_severity['_meta'] = {
    'median_jaccard': median_jac,
    'easy_pids': easy_pids,
    'hard_pids': hard_pids,
    'note': 'easy = Jaccard >= median; hard = Jaccard < median',
}
with open(OUT / 'fig_rag_severity_data.json', 'w') as f:
    json.dump(fig_rag_severity, f, indent=2)

# =============================================================
# (6) fig:gapt — effective exponent gap by operator
# =============================================================
# For each correct sample, parse est_t (symbolic) into exponent, compare to target.
# Output: per-operator mean (exp_effective - exp_target) — limited by symbolic accuracy.

# Reuse complexity_to_exp from compute_shift_metrics
def parse_exp(s):
    if not s: return None
    s = str(s)
    # Look for patterns
    patterns = [
        (r'O\(1\)', 0),
        (r'O\(log[\s\S]{0,8}n\)', 0.3),
        (r'O\(n\^?0\.5\)', 0.5),
        (r'O\(sqrt', 0.5),
        (r'O\(n\s*log\s*n\)', 1.5),
        (r"O\(\(?[nmqQ]+\s*\+?\s*[nmqQ]*\)?\s*log", 1.5),  # O((n+Q)log n), O(m log m)
        (r"O\([nmqQ]\)", 1.0),
        (r"O\([nmqQ]\s*\+\s*[nmqQ]\)", 1.0),  # O(n+Q)
        (r"O\(V\+E\)", 1.0),
        (r"O\([nmqQ]\*[nmqQ]\)", 2.0),
        (r"O\([nmqQ][nmqQ]\)", 2.0),
        (r"O\([nmqQ]\^?2\)", 2.0),
        (r"O\([nmqQ]\^?3\)", 3.0),
        (r'O\(2\^[nN]\)', float('inf')),
        (r'O\([nN]!\)', float('inf')),
    ]
    for pat, val in patterns:
        if re.search(pat, s, re.IGNORECASE):
            return val
    return None

# Aggregate per operator
fig_gapt_data = {}
for op in ['CS','SD','OP','GT']:
    diffs = []
    for p in BENCH:
        if p['operator'] != op: continue
        tgt_exp = parse_exp(p.get('target_complexity', {}).get('time', ''))
        if tgt_exp is None or math.isinf(tgt_exp): continue
        # Loop over all models × shifted samples
        for m in MODELS:
            ent = RES.get(f'{m}__direct', {}).get(p['id'], {}).get('shifted', {})
            for sr in ent.get('sample_results', []):
                if not sr.get('correct'): continue
                e_exp = parse_exp(sr.get('est_t'))
                if e_exp is None or math.isinf(e_exp): continue
                diffs.append(e_exp - tgt_exp)
    fig_gapt_data[op] = {
        'n_samples': len(diffs),
        'mean_diff': sum(diffs)/len(diffs) if diffs else None,
        'std_diff':  statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        'pct_optimal': sum(1 for d in diffs if d <= 0.1)/len(diffs)*100 if diffs else None,
        'note': 'mean of (estimated_exponent - target_exponent) over correct samples across 6 models',
    }
with open(OUT / 'fig_gapt_data.json', 'w') as f:
    json.dump(fig_gapt_data, f, indent=2)

# =============================================================
# Summary printout
# =============================================================
print("\n" + "=" * 90)
print("ALGO TAXONOMY")
print("=" * 90)
print(f"  Coverage: {coverage*100:.1f}% ({total_classified}/{len(BENCH)}) ✅" if coverage >= 0.8 else
      f"  Coverage: {coverage*100:.1f}% ({total_classified}/{len(BENCH)}) ⚠️ below 80%")
print(f"  Families: {sorted(set(ALGO_TO_FAMILY.values()))}")
print(f"  Distance dist: {dict(dist_counts)}")
print()

print("=" * 90)
print("tab:gen_gap_dist (Δ p@1 by alg distance, per-model)")
print("=" * 90)
print(f"{'Model':<22} {'Same':<14} {'Near':<14} {'Far':<14}")
for m, row in table_gen_gap_dist.items():
    cells = []
    for d in ['Same','Near','Far']:
        r = row[d]
        cells.append(f"{r['delta_pp']:+.1f}pp(n={r['n']})" if r['delta_pp'] is not None else f"---(n={r['n']})")
    print(f"{m:<22} {cells[0]:<14} {cells[1]:<14} {cells[2]:<14}")

print()
print("=" * 90)
print("fig:transition (top cells by n)")
print("=" * 90)
print(f"{'src_family':<20} {'tgt_family':<20} {'dist':<6} {'n':<3} {'p@1':<8} {'opt_t':<8}")
for c in fig_transition_data[:15]:
    p1 = f'{c["mean_p1"]:.1f}' if c['mean_p1'] is not None else '---'
    ot = f'{c["mean_opt_t"]:.1f}' if c['mean_opt_t'] is not None else '---'
    print(f"{c['src_family']:<20} {c['tgt_family']:<20} {c['distance']:<6} {c['n_problems']:<3} {p1:<8} {ot:<8}")

print()
print("=" * 90)
print("fig:gapt (effective exponent excess by operator)")
print("=" * 90)
print(f"{'Op':<4} {'n_samples':<10} {'mean(Δexp)':<12} {'%optimal(Δ≤0.1)':<18}")
for op, d in fig_gapt_data.items():
    md = f'{d["mean_diff"]:+.2f}' if d['mean_diff'] is not None else '---'
    pc = f'{d["pct_optimal"]:.1f}%' if d['pct_optimal'] is not None else '---'
    print(f"{op:<4} {d['n_samples']:<10} {md:<12} {pc:<18}")

print()
print(f"→ All JSONs saved to {OUT}/")
