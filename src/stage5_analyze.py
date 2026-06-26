#!/usr/bin/env python3
"""
Stage 5: Compute summary statistics and prepare paper table numbers.
"""

import json, os, time
from collections import defaultdict

RESULTS_FILE  = "/path/to/algobench/results/main_results.json"
SUMMARY_FILE  = "/path/to/algobench/results/summary_stats.json"
LOG_FILE      = "/path/to/algobench/logs/stage5.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def main():
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    log(f"Stage 5: Analyzing {len(results)} problems...")

    # Aggregate by model × strategy
    agg = defaultdict(lambda: {"pass1": [], "pass5": [], "trap": [], "optt": []})
    by_operator = defaultdict(lambda: defaultdict(lambda: {"pass1": [], "trap": [], "optt": []}))

    for pid, data in results.items():
        op = data.get("operator", "?")
        for key, ev in data.get("evals", {}).items():
            if ev.get("pass_at_1") is None:
                continue
            agg[key]["pass1"].append(float(ev["pass_at_1"]))
            agg[key]["pass5"].append(float(ev.get("pass_at_5", ev["pass_at_1"])))
            agg[key]["trap"].append(ev.get("trap_rate", 0))
            agg[key]["optt"].append(ev.get("opt_t", 0))
            by_operator[op][key]["pass1"].append(float(ev["pass_at_1"]))
            by_operator[op][key]["trap"].append(ev.get("trap_rate", 0))
            by_operator[op][key]["optt"].append(ev.get("opt_t", 0))

    def avg(lst): return sum(lst)/len(lst) if lst else 0.0

    summary = {"aggregate": {}, "by_operator": {}, "n_problems": len(results)}

    log("\n=== Main Results Table ===")
    log(f"{'Model/Strategy':<30} {'pass@1':>7} {'pass@5':>7} {'TrapRate':>9} {'OptT':>7} {'N':>4}")
    log("-" * 65)

    for key in sorted(agg.keys()):
        d = agg[key]
        n = len(d["pass1"])
        p1 = avg(d["pass1"])
        p5 = avg(d["pass5"])
        tr = avg(d["trap"])
        ot = avg(d["optt"])
        summary["aggregate"][key] = {"pass_at_1": p1, "pass_at_5": p5, "trap_rate": tr, "opt_t": ot, "n": n}
        log(f"{key:<30} {p1:>7.1%} {p5:>7.1%} {tr:>9.1%} {ot:>7.1%} {n:>4}")

    log("\n=== By Operator ===")
    for op in sorted(by_operator.keys()):
        log(f"\nOperator: {op}")
        for key in sorted(by_operator[op].keys()):
            d = by_operator[op][key]
            n = len(d["pass1"])
            p1 = avg(d["pass1"])
            tr = avg(d["trap"])
            ot = avg(d["optt"])
            if op not in summary["by_operator"]:
                summary["by_operator"][op] = {}
            summary["by_operator"][op][key] = {"pass_at_1": p1, "trap_rate": tr, "opt_t": ot, "n": n}
            log(f"  {key:<30} pass@1={p1:.1%} TrapRate={tr:.1%} OptT={ot:.1%} (n={n})")

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nSummary saved to {SUMMARY_FILE}")

    # Generate paper-ready numbers
    paper_nums = []
    for key, vals in summary["aggregate"].items():
        paper_nums.append(f"{key}: pass@1={vals['pass_at_1']:.1%}, TrapRate={vals['trap_rate']:.1%}, OptT={vals['opt_t']:.1%}")

    paper_file = SUMMARY_FILE.replace(".json", "_paper_numbers.txt")
    with open(paper_file, "w") as f:
        f.write("=== Paper-ready numbers ===\n\n")
        f.write(f"N problems = {len(results)}\n\n")
        for line in paper_nums:
            f.write(line + "\n")
    log(f"Paper numbers saved to {paper_file}")

    # Signal completion
    os.system("ocplatform system event --text 'ConstraintShift Stage5 complete: summary_stats.json ready' --mode now 2>/dev/null || true")

if __name__ == "__main__":
    main()
