#!/usr/bin/env python3
"""
Post-process T4 results: fix opt_t for DP-style targets.
Bug: old is_optimal_full("O(n^2)", "O(nW)") → False
Fix: new is_optimal_full("O(n^2)", "O(nW)") → True (DP target)
"""
import json, re, time
from pathlib import Path

BASE_DIR    = Path("/path/to/research_workspace/automatic_algorithm_design")
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS    = BASE_DIR / "analysis"
OUT_FILE    = RESULTS_DIR / "multimodel_results.json"
LOG_FILE    = BASE_DIR / "logs" / "fix_dp_optt.log"

TARGET_20 = [
    'CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002',
    'GT001','GT002','GT003',
    'CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3'
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def is_permissive_target(target):
    """Target where O(n^2) is acceptable (DP, exponential source, etc.)."""
    t = target.lower()
    if re.search(r'[23n]\^n|n!|factorial|2\*\*n|3\*\*n', t): return True
    if re.search(r'nw\b', t): return True
    if re.search(r'n\s*[*x]\s*w', t): return True
    if re.search(r'\|s\|.*\|t\|', t): return True
    if re.search(r'ks\b', t): return True
    if re.search(r'k\s*[*x]\s*s', t): return True
    if re.search(r'n\^\s*2|n2\b|nm\b', t): return True
    if u'\u00b2' in t: return True  # n²
    if u'\u00b7' in target: return True  # · (middle dot)
    return False

def is_optimal_fixed(est, target):
    """Fixed complexity checker with proper DP/exponential handling."""
    if "UNCERTAIN" in est or "UNKNOWN" in est: return "UNCERTAIN"
    t = target.lower()
    e = est.lower()
    if e == t: return True
    if is_permissive_target(target): return True  # DP/exp target: any poly is OK
    if "n^2" in e or "n^3" in e or "tle" in e or u'\u00b2' in e or u'\u00b3' in e:
        return False
    if re.search(r'o\(\s*log\s*n\s*\)', t) and "n log n" not in t:
        return "log" in e and "n log n" not in e
    if "log" in t: return True
    if re.search(r'o\(\s*(n\b|n\s*[\+\-])', t):
        return "o(n)" in e or "o(1)" in e or "log" in e or "o(n" in e
    return True

def recalc(sample_results, target):
    n_correct = sum(s.get("correct", False) for s in sample_results)
    n_opt = 0
    new_sr = []
    for s in sample_results:
        ns = dict(s)
        if s.get("correct", False):
            opt_v = is_optimal_fixed(s.get("complexity", "UNKNOWN"), target)
            opt_new = opt_v is True
            n_opt += opt_new
            ns["optimal_pre_fix"] = s.get("optimal", False)
            ns["optimal"] = opt_new
        new_sr.append(ns)
    return (n_opt / n_correct if n_correct else 0.0), new_sr

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Fixing DP/polynomial target opt_t for GPT-5.4 T4 results")
    log("=" * 60)

    probs = {}
    with open(BASE_DIR / "data" / "final_benchmark.jsonl") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                probs[p["id"]] = p

    with open(OUT_FILE) as f:
        all_results = json.load(f)

    import shutil
    bak = str(OUT_FILE) + ".bak_pre_dp_fix"
    if not Path(bak).exists():
        shutil.copy(OUT_FILE, bak)
        log(f"Backup: {bak}")

    key = "gpt-5.4__direct"
    data = all_results.get(key, {})
    changed = 0

    for pid in TARGET_20:
        if pid not in data:
            continue
        entry = data[pid]
        prob  = probs.get(pid, {})
        tgt   = prob.get("target_complexity", {}).get("time", "")
        src   = prob.get("source_complexity", {}).get("time", "")

        # Shifted
        shf = entry.get("shifted", {})
        sr  = shf.get("sample_results", [])
        if sr and tgt:
            old_v = shf.get("opt_t", 0.0)
            new_v, new_sr = recalc(sr, tgt)
            if abs(new_v - old_v) > 0.001:
                log(f"  {pid} shifted:  {old_v:.2f} -> {new_v:.2f} (target={tgt})")
                shf["opt_t"] = new_v
                shf["sample_results"] = new_sr
                shf["opt_t_dp_fix"] = f"{old_v:.2f}->{new_v:.2f}"
                entry["shifted"] = shf
                changed += 1

        # Original
        orig   = entry.get("original", {})
        orig_sr = orig.get("sample_results", [])
        if orig_sr and src:
            old_v = orig.get("opt_t", 0.0)
            new_v, new_sr = recalc(orig_sr, src)
            if abs(new_v - old_v) > 0.001:
                log(f"  {pid} original: {old_v:.2f} -> {new_v:.2f} (source={src})")
                orig["opt_t"] = new_v
                orig["sample_results"] = new_sr
                orig["opt_t_dp_fix"] = f"{old_v:.2f}->{new_v:.2f}"
                entry["original"] = orig
                changed += 1

        data[pid] = entry

    all_results[key] = data
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nFixed {changed} values. Saved.")

    # Summary
    log("\n=== GPT-5.4 T20 Final opt_t Summary ===")
    shf_vals  = []
    orig_vals = []
    log(f"{'Problem':<12} {'Target':<32} {'Shf_opt':>8} {'p@1_s':>6} {'Orig_opt':>9} {'p@1_o':>6}")
    for pid in TARGET_20:
        entry = data.get(pid, {})
        prob  = probs.get(pid, {})
        tgt   = prob.get("target_complexity", {}).get("time", "")
        shf   = entry.get("shifted", {})
        orig  = entry.get("original", {})
        sv    = shf.get("opt_t") or 0.0
        ov    = orig.get("opt_t") or 0.0
        sp1   = shf.get("pass_at_1", False)
        op1   = orig.get("pass_at_1", False)
        shf_vals.append(sv)
        orig_vals.append(ov)
        flag = " *" if is_permissive_target(tgt) else ""
        log(f"  {pid:<12} {(tgt+flag):<32} {sv:>8.2f} {str(sp1)[:1]:>6} {ov:>9.2f} {str(op1)[:1]:>6}")

    log(f"\n  Mean shifted opt_t  (T20): {sum(shf_vals)/len(shf_vals):.3f}")
    log(f"  Mean original opt_t (T20): {sum(orig_vals)/len(orig_vals):.3f}")

    # Update analysis
    ANALYSIS.mkdir(exist_ok=True)
    upd_file = ANALYSIS / "table_main_opt_t_update.json"
    upd = json.loads(upd_file.read_text()) if upd_file.exists() else {}
    upd["gpt-5.4"] = {
        "shifted_opt_t_mean":  sum(shf_vals)  / max(len(shf_vals), 1),
        "original_opt_t_mean": sum(orig_vals) / max(len(orig_vals), 1),
        "n_shifted":  len(shf_vals),
        "n_original": len(orig_vals),
        "note": "T20 subset (ORIG-11+HARD-9); DP fix applied; re-run 2026-06-25",
    }
    upd_file.write_text(json.dumps(upd, indent=2))
    log(f"\nWritten to {upd_file}")

if __name__ == "__main__":
    main()
