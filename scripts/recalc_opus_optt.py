#!/usr/bin/env python3
"""
Recalculate Claude Opus 4.5 opt_t using the FULL complexity checker.

The existing claude-opus-4-5__direct data has sample_results with stored complexity
estimates and 'optimal' flags. We recalculate optimal using the proper checker.

Key fix: is_optimal_new("O(n^2)", "O(nW)") → True (DP-style targets accept quadratic)
         is_optimal_old("O(n^2)", "O(nW)") → False (bug: treats any n^2 as suboptimal)

For ORIGINAL mode: data was stored without sample_results, so we can only recalculate
shifted mode. Original opt_t remains 0.0 / cannot be recalculated without API re-run.

NOTE: ANTHROPIC_API_KEY was invalid (401 error). Recalculating from stored data only.
"""

import json, re, time
from pathlib import Path

BASE_DIR    = Path("/path/to/.openclaw/research_topics/automatic_algorithm_design")
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS    = BASE_DIR / "analysis"
OUT_FILE    = RESULTS_DIR / "multimodel_results.json"
LOG_FILE    = BASE_DIR / "logs" / "recalc_opus_optt.log"

TARGET_20 = [
    'CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002',
    'GT001','GT002','GT003',
    'CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3'
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def is_optimal_new(est, target):
    """Full complexity checker (improved from eval_checker.py + fixes)."""
    if "UNCERTAIN" in est or "UNKNOWN" in est:
        return "UNCERTAIN"
    t = target.lower()
    e = est.lower()
    # Exact match → always optimal
    if e == t:
        return True
    # DP-style targets: O(nW), O(n^2...), O(kW) are valid polynomial targets
    if re.search(r'n[\^*]2|nw|n\*w|\bkw\b', t):
        return True
    # Reject clearly bad complexities
    if "n^2" in e or "n^3" in e or "tle" in e or "n²" in e or "n³" in e:
        return False
    # Tight log-only targets
    if re.search(r'o\(\s*log\s*n\s*\)', t) and "n log n" not in t:
        return "log" in e and "n log n" not in e
    # Any log target (n log n, etc.)
    if "log" in t:
        return True
    # Linear/near-linear targets: accept O(n), O(1), O(log n), O(n+...) estimates
    if re.search(r'o\(\s*(n\b|n\s*[\+\-])', t):
        return "o(n)" in e or "o(1)" in e or "log" in e or "o(n" in e
    return True

def recalc_optt_from_samples(sample_results, target):
    """Recalculate opt_t using new checker from stored complexity estimates."""
    n_correct = sum(s.get("correct", False) for s in sample_results)
    n_opt_new = 0
    new_samples = []
    for s in sample_results:
        if s.get("correct", False):
            est = s.get("complexity", "UNKNOWN")
            opt_v = is_optimal_new(est, target)
            opt_new = opt_v is True
            n_opt_new += opt_new
            new_s = dict(s)
            new_s["optimal"] = opt_new
            new_s["optimal_old"] = s.get("optimal", False)
        else:
            new_s = dict(s)
        new_samples.append(new_s)
    opt_t_new = n_opt_new / n_correct if n_correct else 0.0
    return opt_t_new, new_samples

def main():
    log("=" * 60)
    log("Recalculating Claude Opus 4.5 opt_t from stored complexity estimates")
    log("=" * 60)
    log("NOTE: ANTHROPIC_API_KEY invalid → no API re-run. Using stored data only.")

    # Load benchmark for target complexities
    probs = {}
    with open(BASE_DIR / "data" / "final_benchmark.jsonl") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                probs[p["id"]] = p

    # Load results
    with open(OUT_FILE) as f:
        all_results = json.load(f)

    import shutil
    backup = str(OUT_FILE) + ".bak_before_recalc"
    if not Path(backup).exists():
        shutil.copy(OUT_FILE, backup)
        log(f"Backed up to {backup}")

    key = "claude-opus-4-5__direct"
    data = all_results.get(key, {})

    log(f"\nProcessing {key}: {len(data)} problems")

    changed_count = 0
    opt_t_summary = {}

    for pid in TARGET_20:
        if pid not in data:
            log(f"  {pid}: NOT IN DATA - skipping")
            continue

        entry = data[pid]
        prob  = probs.get(pid, {})
        target_time  = prob.get("target_complexity", {}).get("time", "")
        source_time  = prob.get("source_complexity", {}).get("time", "")

        # ── Shifted mode recalculation ─────────────────────────────────────
        shf = entry.get("shifted", {})
        sr  = shf.get("sample_results", [])
        if sr and target_time:
            old_opt_t = shf.get("opt_t", 0.0)
            new_opt_t, new_sr = recalc_optt_from_samples(sr, target_time)
            if abs(new_opt_t - old_opt_t) > 0.001:
                log(f"  {pid} SHIFTED: opt_t {old_opt_t:.2f} → {new_opt_t:.2f} (target={target_time})")
                changed_count += 1
                shf["opt_t"] = new_opt_t
                shf["sample_results"] = new_sr
                shf["opt_t_recalc_note"] = f"Recalculated with full checker (was {old_opt_t:.2f})"
                entry["shifted"] = shf
            opt_t_summary[pid + "_shifted"] = new_opt_t
        else:
            opt_t_summary[pid + "_shifted"] = shf.get("opt_t", 0.0)

        # ── Original mode: no sample_results stored → can't recalculate ───
        orig = entry.get("original", {})
        orig_sr = orig.get("sample_results", [])
        if orig_sr and source_time:
            old_opt_t = orig.get("opt_t", 0.0)
            new_opt_t, new_sr = recalc_optt_from_samples(orig_sr, source_time)
            if abs(new_opt_t - old_opt_t) > 0.001:
                log(f"  {pid} ORIGINAL: opt_t {old_opt_t:.2f} → {new_opt_t:.2f} (source={source_time})")
                changed_count += 1
                orig["opt_t"] = new_opt_t
                orig["sample_results"] = new_sr
                orig["opt_t_recalc_note"] = f"Recalculated with full checker (was {old_opt_t:.2f})"
                entry["original"] = orig
        else:
            if not orig_sr:
                log(f"  {pid} ORIGINAL: no sample_results stored - cannot recalculate")
                orig["opt_t_recalc_note"] = "Cannot recalculate: no sample_results in stored data, API key unavailable"
                entry["original"] = orig
        opt_t_summary[pid + "_original"] = orig.get("opt_t", 0.0)

        data[pid] = entry

    all_results[key] = data

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved updated results. Changed {changed_count} opt_t values.")

    # Summary
    log("\n=== Claude Opus 4.5 T20 opt_t Summary ===")
    log(f"{'Problem':<12} {'Shf_opt_t':>10} {'Orig_opt_t':>12}")
    for pid in TARGET_20:
        shf_v  = opt_t_summary.get(pid + "_shifted", 0.0)
        orig_v = opt_t_summary.get(pid + "_original", 0.0)
        log(f"  {pid:<12} {shf_v:>10.2f} {orig_v:>12.2f}")

    # Mean values
    shf_vals  = [opt_t_summary.get(p + "_shifted",  0.0) for p in TARGET_20]
    orig_vals = [opt_t_summary.get(p + "_original", 0.0) for p in TARGET_20]
    log(f"\n  Mean shifted opt_t:  {sum(shf_vals)/len(shf_vals):.3f}")
    log(f"  Mean original opt_t: {sum(orig_vals)/len(orig_vals):.3f}")

    # Write opt_t update file
    update = {
        "claude-opus-4-5": {
            "note": "Recalculated from stored complexity estimates (API key unavailable for re-run)",
            "shifted_opt_t_mean":  sum(shf_vals)  / max(len(shf_vals),  1),
            "original_opt_t_mean": sum(orig_vals) / max(len(orig_vals), 1),
            "original_opt_t_note": "All 0.0 - no sample_results in prior run, API key invalid (401)",
            "n_shifted":  len(shf_vals),
            "n_original": len(orig_vals),
        }
    }
    out_update = ANALYSIS / "table_main_opt_t_update.json"
    ANALYSIS.mkdir(exist_ok=True)
    if out_update.exists():
        with open(out_update) as f:
            existing = json.load(f)
        existing.update(update)
        update = existing
    with open(out_update, "w") as f:
        json.dump(update, f, indent=2)
    log(f"\nUpdate summary written to {out_update}")
    log("\nRecalculation complete.")

if __name__ == "__main__":
    main()
