#!/usr/bin/env python3
"""
Stage 6: Quantify generalization gap between source and shifted problems.
Computes: text similarity, constraint magnitude ratio, complexity delta,
          algorithm class distance, structural novelty score.
"""
import json, re, math
from pathlib import Path
from collections import Counter

BASE_DIR     = Path(__file__).resolve().parent.parent
BENCH_FILE   = str(BASE_DIR / "data" / "final_benchmark.jsonl")
OUTPUT_FILE  = str(BASE_DIR / "results" / "generalization_gap.json")
SUMMARY_FILE = str(BASE_DIR / "results" / "generalization_summary.json")

# ── Algorithm family taxonomy ─────────────────────────────────────────────────
ALG_FAMILY = {
    "prefix_sum":                          "array_static",
    "brute_force_O(n2)":                   "brute_force",
    "brute_force":                         "brute_force",
    "bellman_ford":                        "graph_shortest",
    "bellman_ford_or_floyd":               "graph_shortest",
    "dijkstra_with_heap":                  "graph_shortest",
    "dsu":                                 "graph_connectivity",
    "offline_dynamic_connectivity_dsu_rollback": "graph_connectivity",
    "online_dsu_with_size":                "graph_connectivity",
    "bfs_per_query":                       "graph_traversal",
    "bfs_dfs":                             "graph_traversal",
    "kruskal":                             "graph_spanning",
    "kruskal_or_prim":                     "graph_spanning",
    "kruskal_plus_lca_on_kruskal_tree":    "graph_spanning",
    "kadane":                              "dp_1d",
    "dp_O(nm)":                            "dp_2d",
    "iterative_dp":                        "dp_1d",
    "LCS_as_LIS_via_inverse_permutation":  "dp_1d",
    "01_knapsack_dp":                      "dp_knapsack",
    "dp_with_switch_count":                "dp_constrained",
    "dp_with_type_cooldown_state":         "dp_constrained",
    "dp_sorted_intervals_with_switch_count": "dp_constrained",
    "fenwick_tree_BIT":                    "array_dynamic",
    "lazy_segment_tree":                   "array_dynamic",
    "greedy_earliest_finish":              "greedy",
    "greedy_by_profit_with_dsu":           "greedy",
    "greedy_by_value_per_weight":          "greedy",
    "greedy_by_value_per_weight_fractional": "greedy",
    "prefix_sum_with_merge_sort_or_BIT":   "array_dynamic",
    "prefix_sum_with_BIT":                 "array_dynamic",
    "hashmap_or_twopointer":               "hash_linear",
    "matrix_exponentiation":               "math_exp",
}

FAMILY_DISTANCE = {
    # (fam_a, fam_b) → distance class: "same", "near", "far"
    ("array_static",       "array_dynamic"):      "near",
    ("array_static",       "dp_1d"):              "near",
    ("brute_force",        "hash_linear"):        "near",
    ("brute_force",        "array_dynamic"):      "near",
    ("brute_force",        "dp_1d"):              "near",
    ("graph_shortest",     "graph_spanning"):     "near",
    ("graph_traversal",    "graph_connectivity"): "near",
    ("graph_connectivity", "graph_connectivity"): "same",
    ("graph_spanning",     "graph_spanning"):     "same",
    ("dp_1d",              "dp_2d"):              "near",
    ("dp_1d",              "dp_constrained"):     "near",
    ("dp_1d",              "dp_knapsack"):        "near",
    ("greedy",             "dp_constrained"):     "far",
    ("greedy",             "dp_1d"):              "far",
    ("greedy",             "dp_knapsack"):        "far",
    ("graph_spanning",     "graph_connectivity"): "near",
    ("dp_1d",              "math_exp"):           "far",
    ("array_dynamic",      "graph_connectivity"): "far",
    ("array_static",       "greedy"):             "far",
}

def alg_distance(src_alg, tgt_alg):
    if src_alg == tgt_alg: return "same"
    fam_s = ALG_FAMILY.get(src_alg, "other")
    fam_t = ALG_FAMILY.get(tgt_alg, "other")
    if fam_s == fam_t: return "same"
    key = (min(fam_s, fam_t), max(fam_s, fam_t))
    key2 = (fam_s, fam_t)
    return FAMILY_DISTANCE.get(key2, FAMILY_DISTANCE.get(key, "far"))

# ── Complexity exponent mapping ───────────────────────────────────────────────
def complexity_exponent(c):
    c = c.lower()
    if "n³" in c or "n^3" in c or "n3" in c: return 3.0
    if "n²" in c or "n^2" in c or "n2" in c or "o(n²)" in c: return 2.0
    if "n log n" in c or "n log²" in c: return 1.5
    if "n sqrt" in c: return 1.5
    if "o(n)" in c or "o(n+q)" in c or "o(n+m)" in c: return 1.0
    if "o(nw)" in c or "o(n*w)" in c: return 1.8   # knapsack ~ n*W
    if "o(log n)" in c: return 0.5
    if "o(1)" in c: return 0.0
    if "alpha" in c or "α" in c: return 1.0         # amortized ~linear
    if "polylog" in c: return 1.3
    return 1.0  # default

# ── Text similarity (word-overlap / Jaccard) ─────────────────────────────────
def word_jaccard(text_a, text_b):
    def tokens(t):
        return set(re.findall(r'\b[a-z0-9]+\b', t.lower()))
    a, b = tokens(text_a), tokens(text_b)
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def bleu1(reference, hypothesis):
    ref_tokens  = re.findall(r'\b[a-z0-9]+\b', reference.lower())
    hyp_tokens  = re.findall(r'\b[a-z0-9]+\b', hypothesis.lower())
    if not hyp_tokens: return 0.0
    ref_count = Counter(ref_tokens)
    clipped = sum(min(cnt, ref_count.get(tok, 0))
                  for tok, cnt in Counter(hyp_tokens).items())
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return bp * clipped / len(hyp_tokens)

# ── Constraint magnitude extraction ──────────────────────────────────────────
def extract_n(constraint_str):
    """Extract largest N value from constraint string."""
    patterns = [
        r'[nN]\s*[≤<=]+\s*(\d+)\s*[×x]\s*10\^(\d+)',  # N ≤ 2×10^5
        r'[nN]\s*[≤<=]+\s*(\d+)[eE](\d+)',              # N ≤ 2e5
        r'[nN]\s*[≤<=]+\s*(\d[\d,]*)',                  # N ≤ 200000
        r'(\d+)\s*[×x]\s*10\^(\d+)',                     # 2×10^5
        r'10\^(\d+)',                                     # 10^5
        r'(\d[\d,]+)',                                    # any large number
    ]
    best = 0
    for pat in patterns:
        for m in re.finditer(pat, constraint_str.replace(' ', '')):
            groups = m.groups()
            try:
                if len(groups) == 2:
                    val = int(groups[0]) * (10 ** int(groups[1]))
                else:
                    val = int(str(groups[0]).replace(',', ''))
                best = max(best, val)
            except: pass
    return best if best > 0 else None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    problems = []
    with open(BENCH_FILE) as f:
        for line in f:
            if line.strip(): problems.append(json.loads(line))

    print(f"Computing generalization gap for {len(problems)} problems...")

    results = []
    for prob in problems:
        src_stmt  = prob.get("source_statement", "")
        shf_stmt  = prob.get("shifted_statement", "")
        src_con   = prob.get("source_constraints", "")
        shf_con   = prob.get("shifted_constraints", "")
        src_alg   = prob.get("source_reference_algorithm", "")
        tgt_alg   = prob.get("target_algorithm", "")
        src_cpx   = prob.get("source_complexity", {}).get("time", "")
        tgt_cpx   = prob.get("target_complexity", {}).get("time", "")

        # a) Text similarity
        jaccard   = word_jaccard(src_stmt, shf_stmt)
        bleu      = bleu1(src_stmt, shf_stmt)

        # b) Length ratio
        len_ratio = len(shf_stmt) / max(len(src_stmt), 1)

        # c) Constraint magnitude ratio
        src_n = extract_n(src_con)
        shf_n = extract_n(shf_con)
        n_ratio = (shf_n / src_n) if src_n and shf_n else None

        # d) Algorithm class distance
        dist = alg_distance(src_alg, tgt_alg)

        # e) Complexity exponent delta
        src_exp = complexity_exponent(src_cpx)
        tgt_exp = complexity_exponent(tgt_cpx)
        exp_delta = tgt_exp - src_exp

        entry = {
            "id": prob["id"],
            "title": prob["title"],
            "operator": prob["operator"],
            "text_jaccard": round(jaccard, 4),
            "text_bleu1": round(bleu, 4),
            "len_ratio": round(len_ratio, 3),
            "src_n": src_n,
            "shifted_n": shf_n,
            "n_ratio": round(n_ratio, 1) if n_ratio else None,
            "src_alg": src_alg,
            "tgt_alg": tgt_alg,
            "alg_distance": dist,
            "src_complexity": src_cpx,
            "tgt_complexity": tgt_cpx,
            "src_exponent": src_exp,
            "tgt_exponent": tgt_exp,
            "complexity_delta": round(exp_delta, 2),
        }
        results.append(entry)
        print(f"  {prob['id']:8s} [{prob['operator']}] jaccard={jaccard:.2f} "
              f"n_ratio={n_ratio}× alg_dist={dist} Δexp={exp_delta:+.1f}")

    # Write per-problem results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate summary
    def avg(lst): return sum(lst)/len(lst) if lst else 0
    def std(lst):
        if len(lst) < 2: return 0
        m = avg(lst); return (sum((x-m)**2 for x in lst)/len(lst))**0.5

    jaccard_vals  = [r["text_jaccard"]    for r in results]
    bleu_vals     = [r["text_bleu1"]      for r in results]
    len_vals      = [r["len_ratio"]       for r in results]
    nratio_vals   = [r["n_ratio"]         for r in results if r["n_ratio"]]
    exp_vals      = [r["complexity_delta"] for r in results]
    dist_counts   = Counter(r["alg_distance"] for r in results)
    n             = len(results)

    summary = {
        "n_problems": n,
        "text_jaccard":    {"mean": round(avg(jaccard_vals),3), "std": round(std(jaccard_vals),3)},
        "text_bleu1":      {"mean": round(avg(bleu_vals),3),    "std": round(std(bleu_vals),3)},
        "len_ratio":       {"mean": round(avg(len_vals),3),     "std": round(std(len_vals),3)},
        "n_ratio":         {"mean": round(avg(nratio_vals),1),  "std": round(std(nratio_vals),1),
                            "n_available": len(nratio_vals)},
        "complexity_delta":{"mean": round(avg(exp_vals),2),     "std": round(std(exp_vals),2)},
        "alg_distance_dist": {"same": dist_counts.get("same",0),
                               "near": dist_counts.get("near",0),
                               "far":  dist_counts.get("far",0)},
        "alg_distance_far_pct": round(dist_counts.get("far",0)/n*100, 1),
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Generalization Gap Summary (n={n}) ===")
    print(f"  Text Jaccard similarity:    {summary['text_jaccard']['mean']:.3f} ± {summary['text_jaccard']['std']:.3f}")
    print(f"  Text BLEU-1:                {summary['text_bleu1']['mean']:.3f} ± {summary['text_bleu1']['std']:.3f}")
    print(f"  Statement length ratio:     {summary['len_ratio']['mean']:.2f}× ± {summary['len_ratio']['std']:.2f}")
    print(f"  Constraint magnitude ratio: {summary['n_ratio']['mean']:.0f}× ± {summary['n_ratio']['std']:.0f} (n={summary['n_ratio']['n_available']})")
    print(f"  Complexity exponent delta:  {summary['complexity_delta']['mean']:+.2f} ± {summary['complexity_delta']['std']:.2f}")
    print(f"  Algorithm distance: same={dist_counts.get('same',0)} near={dist_counts.get('near',0)} far={dist_counts.get('far',0)}  ({summary['alg_distance_far_pct']}% far)")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"       {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
