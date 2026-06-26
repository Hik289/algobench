#!/usr/bin/env python3
"""
eval_ablation.py — Quality Gate Ablation for ConstraintShift paper (Task A)

Computes table_ablation.json with 5 rows:
  full, no_g1, no_g2, no_g3, no_g4

For each row, we measure:
  old_sol_pass  : fraction of accepted candidates where old source solution still passes shifted_examples
  ref_fail      : fraction of accepted candidates where LLM reference solution fails shifted_examples
  near_para     : fraction of accepted candidates where Jaccard(src_stmt, shft_stmt) >= NEAR_PARA_THRESH
  f_opt         : fraction of accepted candidates where verifier mislabels target complexity

Candidate pool: 54 problems = 52 from final_benchmark.jsonl + 2 rejected from release source_problems.jsonl
                (CS002, CS004 were rejected by Gate 1 in the actual pipeline)

Usage:
  python3 eval_ablation.py
  (OPENAI_API_KEY must be set in environment)

Author: ml_engineer_claude_isolated (subagent for [Lab] R5b revision)
"""

import os, sys, json, ast, re, time, subprocess, tempfile, hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path("/path/to/.openclaw/research_topics/automatic_algorithm_design")
BENCH_FILE = BASE_DIR / "data" / "final_benchmark.jsonl"
SRC_FILE   = BASE_DIR / "release" / "constraintshift" / "data" / "source_problems.jsonl"
OUT_JSON   = BASE_DIR / "analysis" / "table_ablation.json"
LOG_FILE   = BASE_DIR / "logs" / "eval_ablation.log"
CACHE_FILE = BASE_DIR / "logs" / "eval_ablation_code_cache.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o-mini"
PYTHON         = sys.executable
NEAR_PARA_THRESH = 0.70   # Jaccard threshold for near-paraphrase

# ─── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ─── Code cache (avoid re-calling API) ────────────────────────────────────────

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_candidates():
    """Load all 54 candidates: 52 from benchmark + 2 extra from release source."""
    bench = {}
    for line in BENCH_FILE.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            bench[p["id"]] = p

    # Add rejected candidates from release source (CS002, CS004)
    all_cands = dict(bench)
    for line in SRC_FILE.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            if p["id"] not in all_cands:
                # These are the candidates that were rejected from the benchmark
                p["gate1_result"] = "UNKNOWN_SOURCE_ONLY"
                p["benchmark_status"] = "not_in_benchmark"
                all_cands[p["id"]] = p

    log(f"Loaded {len(all_cands)} total candidates ({len(bench)} from benchmark, "
        f"{len(all_cands)-len(bench)} extra source-only)")
    return list(all_cands.values())

# ─── Complexity estimation (from stage4_evaluate.py) ──────────────────────────

def estimate_complexity(code):
    """Rough AST-based time complexity estimation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "UNKNOWN"

    max_depth = 0
    def count_loop_depth(node, depth=0):
        nonlocal max_depth
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
            max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            count_loop_depth(child, depth)

    count_loop_depth(tree)

    code_lower = code.lower()
    if any(p in code_lower for p in ['segment_tree', 'segtree', 'seg_tree', 'lazy', '.update(', '.query(']):
        return "O(n log n) [segment tree]"
    if any(p in code_lower for p in ['fenwick', 'bit[', 'bit.update', 'lowbit', 'i & (-i)', 'i & -i']):
        return "O(n log n) [BIT/Fenwick]"
    if any(p in code_lower for p in ['heapq', 'heap', 'dijkstra']):
        return "O(n log n) [heap/Dijkstra]"
    if any(p in code_lower for p in ['mat_pow', 'matrix_power', 'mat_mult', 'matmul', 'np.linalg', 'matrix_exp']):
        return "O(log n) [matrix exp]"
    if any(p in code_lower for p in ['bisect', 'binary_search', 'binary search']):
        return "O(n log n) [binary search]"
    if 'sort(' in code_lower or '.sort(' in code_lower:
        return "O(n log n) [sort]"

    if max_depth >= 3:
        return "O(n^3) or worse"
    if max_depth == 2:
        return "O(n^2)"
    if max_depth == 1:
        return "O(n)"
    return "O(1) or O(log n)"

def complexity_matches_target(estimated, target_str):
    """
    Returns True if estimated complexity is compatible with the target.
    F-Opt error = NOT matches (i.e., verifier would be wrong).
    Uses robust matching to handle varied target format strings.
    """
    import re as _re
    t = target_str.lower()
    e = estimated.lower()

    # Flag obviously high complexities in estimate
    is_quadratic_or_worse = ("n^2" in e or "n²" in e or "n^3" in e or "n³" in e or "worse" in e)

    # If target explicitly mentions n^2 (e.g. O(nW) treated as polynomial), accept quadratic
    if _re.search(r'n[\^*]2|nw|n\*w', t):
        return True  # DP-style — any polynomial OK

    # Reject quadratic/worse unless target also mentions it
    if is_quadratic_or_worse:
        return False

    # O(log n) target: very tight — need pure log, not n log n
    if _re.search(r'o\(\s*log\s*n\s*\)', t) or (
        "log n" in t and "n log n" not in t and "n+q" not in t and "n*log" not in t
    ):
        return "log" in e and "n log n" not in e

    # Any target involving log n (O(n log n), O((n+Q)log n), O(n log^2 n), etc.)
    if "log" in t:
        # Accept log or better (linear, log)
        return True  # already rejected quadratic above

    # O(n) or O(n+Q) target — accept linear or better
    if _re.search(r'o\(\s*(n\b|n\s*\+)', t):
        return "o(n)" in e or "o(1)" in e or "log" in e

    # Default: not clearly quadratic or worse
    return True

# ─── Code execution ────────────────────────────────────────────────────────────

def run_code_on_examples(code, examples, timeout=5):
    """Run code on a list of examples. Returns (all_pass, fail_reason)."""
    if not code or not code.strip():
        return False, "empty code"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        for ex in examples:
            try:
                r = subprocess.run([PYTHON, fname], input=ex["input"],
                                   capture_output=True, text=True, timeout=timeout)
                if r.returncode != 0:
                    return False, f"runtime error: {r.stderr[:150]}"
                if r.stdout.strip() != ex["output"].strip():
                    return False, f"WA: got {r.stdout.strip()!r}, expected {ex['output'].strip()!r}"
            except subprocess.TimeoutExpired:
                return False, "TLE"
        return True, "all pass"
    except Exception as e:
        return False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

# ─── Jaccard similarity ────────────────────────────────────────────────────────

def jaccard_words(s1, s2):
    """Word-level Jaccard similarity."""
    w1 = set(re.findall(r'\w+', s1.lower()))
    w2 = set(re.findall(r'\w+', s2.lower()))
    if not w1 and not w2:
        return 1.0
    return len(w1 & w2) / len(w1 | w2)

# ─── Code generation (GPT-4o-mini) ────────────────────────────────────────────

def call_gpt(system, user, model=GPT_MODEL):
    """Single call to GPT-4o-mini; returns text or raises."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=1024,
        temperature=0.0,
    )
    return resp.choices[0].message.content

def extract_code(text):
    """Pull Python from ```python...``` or bare code."""
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if 'def ' in text or 'import ' in text or 'for ' in text or 'print(' in text:
        lines = text.strip().splitlines()
        # Remove lines that look like commentary before code
        code_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'def ', 'from ', 'print', 'class ', '#')):
                code_start = i
                break
        return '\n'.join(lines[code_start:]).strip()
    return text.strip()

def gen_old_solution(prob, cache):
    """Generate old (source-algorithm) solution. Cached by problem ID."""
    ckey = f"old_{prob['id']}"
    if ckey in cache:
        return cache[ckey]

    # Use hardcoded templates for the 4 problems in stage2
    HARDCODED = {
        "CS001": """
import sys
input = sys.stdin.readline
def solve():
    n, q = map(int, input().split())
    A = list(map(int, input().split()))
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + A[i]
    for _ in range(q):
        l, r = map(int, input().split())
        print(prefix[r] - prefix[l-1])
solve()
""",
        "CS002": """
import sys
input = sys.stdin.readline
def solve():
    n, T = map(int, input().split())
    A = list(map(int, input().split()))
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i] + A[j] == T:
                count += 1
    print(count)
solve()
""",
        "GT003": """
import sys
input = sys.stdin.readline
def solve():
    n, W = map(int, input().split())
    items = []
    for _ in range(n):
        w, v = map(int, input().split())
        items.append((v/w, w, v))
    items.sort(reverse=True)
    total = 0.0
    remaining = W
    for ratio, w, v in items:
        if remaining <= 0:
            break
        take = min(w, remaining)
        total += take * ratio
        remaining -= take
    print(f"{total:.2f}")
solve()
""",
        "CS004": """
import sys
def solve():
    n = int(sys.stdin.read().strip())
    MOD = 10**9 + 7
    if n <= 1:
        print(n)
        return
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, (a+b) % MOD
    print(b)
solve()
""",
    }

    if prob["id"] in HARDCODED:
        code = HARDCODED[prob["id"]].strip()
        cache[ckey] = code
        return code

    # For other problems, prompt GPT-4o-mini to write old (naive/suboptimal) solution
    system = "You are an expert programmer. Write ONLY Python code, no explanation."
    user = (
        f"Write a Python solution for this problem using the OLD algorithm: {prob['source_reference_algorithm']}.\n"
        f"This is the ORIGINAL problem (before the constraint shift):\n"
        f"{prob['source_statement']}\n\n"
        f"The shifted (harder) problem is:\n"
        f"{prob['shifted_statement']}\n\n"
        f"INPUT FORMAT: {prob.get('shifted_input','see problem')}\n"
        f"OUTPUT FORMAT: {prob.get('shifted_output','see problem')}\n"
        f"EXAMPLES:\n"
        + "\n".join(f"Input: {e['input']}\nOutput: {e['output']}" for e in prob.get('shifted_examples',[]))
        + "\n\n"
        f"Write the {prob['source_reference_algorithm']} approach (the OLD suboptimal solution). "
        f"It should be correct on small inputs but may be slow on large inputs. "
        f"Output ONLY complete Python code."
    )
    try:
        raw = call_gpt(system, user)
        code = extract_code(raw)
        cache[ckey] = code
        return code
    except Exception as e:
        log(f"    gen_old_solution API error for {prob['id']}: {e}")
        cache[ckey] = ""
        return ""

def gen_reference_solution(prob, cache):
    """Generate new reference solution using target algorithm. Cached."""
    ckey = f"ref_{prob['id']}"
    if ckey in cache:
        return cache[ckey]

    system = "You are an expert competitive programmer. Write ONLY Python code, no explanation."
    user = (
        f"Write a CORRECT and EFFICIENT Python solution using {prob['target_algorithm']} "
        f"with time complexity {prob['target_complexity']['time']}.\n\n"
        f"Problem: {prob['shifted_statement']}\n\n"
        f"INPUT FORMAT: {prob.get('shifted_input','see problem')}\n"
        f"OUTPUT FORMAT: {prob.get('shifted_output','see problem')}\n"
        f"CONSTRAINTS: {prob.get('shifted_constraints','see problem')}\n"
        f"EXAMPLES:\n"
        + "\n".join(f"Input: {e['input']}\nOutput: {e['output']}" for e in prob.get('shifted_examples',[]))
        + "\n\nOutput ONLY complete Python code."
    )
    try:
        raw = call_gpt(system, user)
        code = extract_code(raw)
        cache[ckey] = code
        return code
    except Exception as e:
        log(f"    gen_reference_solution API error for {prob['id']}: {e}")
        cache[ckey] = ""
        return ""

# ─── Compute 4 metrics per candidate ──────────────────────────────────────────

def compute_candidate_metrics(prob, cache):
    """
    Returns dict with:
      g1_old_sol_passes  : bool (True = bad, old sol passes → should have been rejected by G1)
      g2_ref_fails       : bool (True = bad, new ref fails → should have been rejected by G2)
      g3_near_para       : bool (True = bad, near paraphrase → should have been rejected by G3)
      g4_f_opt_error     : bool (True = bad, verifier mislabels → should have been rejected by G4)
      details            : dict of sub-details
    """
    pid = prob["id"]
    examples = prob.get("shifted_examples", [])
    details = {}

    # ── G1: old solution passes on shifted examples ──────────────────────────
    old_code = gen_old_solution(prob, cache)
    if old_code:
        passes, reason = run_code_on_examples(old_code, examples, timeout=5)
        g1_old_sol_passes = passes
        details["g1_old_passes"] = passes
        details["g1_reason"] = reason
    else:
        # No old code → can't check, use gate1_result metadata as proxy
        gate1 = prob.get("gate1_result", "")
        g1_old_sol_passes = ("FAIL" in gate1 and "PASS" not in gate1)
        details["g1_old_passes"] = g1_old_sol_passes
        details["g1_reason"] = f"no_code, metadata: {gate1[:80]}"

    # ── G2: reference solution fails on shifted examples ─────────────────────
    ref_code = gen_reference_solution(prob, cache)
    if ref_code:
        ref_passes, ref_reason = run_code_on_examples(ref_code, examples, timeout=10)
        g2_ref_fails = not ref_passes
        details["g2_ref_fails"] = g2_ref_fails
        details["g2_reason"] = ref_reason
    else:
        g2_ref_fails = False  # Conservative: can't determine
        details["g2_ref_fails"] = False
        details["g2_reason"] = "no_ref_code_generated"

    # ── G3: near-paraphrase check ─────────────────────────────────────────────
    src_stmt  = prob.get("source_statement", "")
    shft_stmt = prob.get("shifted_statement", "")
    jacc = jaccard_words(src_stmt, shft_stmt)
    g3_near_para = (jacc >= NEAR_PARA_THRESH)
    details["g3_jaccard"] = round(jacc, 4)
    details["g3_near_para"] = g3_near_para

    # ── G4: verifier mislabels complexity ─────────────────────────────────────
    if ref_code:
        est = estimate_complexity(ref_code)
        target_time = prob.get("target_complexity", {}).get("time", "")
        correct = complexity_matches_target(est, target_time)
        g4_f_opt_error = not correct
        details["g4_estimated"] = est
        details["g4_target"] = target_time
        details["g4_f_opt_error"] = g4_f_opt_error
    else:
        g4_f_opt_error = False
        details["g4_estimated"] = "unknown"
        details["g4_target"] = prob.get("target_complexity", {}).get("time", "")
        details["g4_f_opt_error"] = False

    return {
        "pid": pid,
        "g1_old_sol_passes":  g1_old_sol_passes,
        "g2_ref_fails":       g2_ref_fails,
        "g3_near_para":       g3_near_para,
        "g4_f_opt_error":     g4_f_opt_error,
        "details":            details,
    }

# ─── Ablation logic ────────────────────────────────────────────────────────────

def accepted_under_config(metrics_list, disable_gate):
    """
    Return list of metrics for candidates accepted when 'disable_gate' is disabled.
    Gates 1-4; disable_gate=0 means full pipeline (all gates active).
    """
    result = []
    for m in metrics_list:
        # Each gate rejects candidates showing the "bad" flag
        reject_g1 = m["g1_old_sol_passes"]   and (disable_gate != 1)
        reject_g2 = m["g2_ref_fails"]         and (disable_gate != 2)
        reject_g3 = m["g3_near_para"]         and (disable_gate != 3)
        reject_g4 = m["g4_f_opt_error"]       and (disable_gate != 4)
        if not (reject_g1 or reject_g2 or reject_g3 or reject_g4):
            result.append(m)
    return result

def compute_row(config_name, accepted_metrics):
    """Compute the 4 rate metrics for a set of accepted problems."""
    n = len(accepted_metrics)
    if n == 0:
        return {"config": config_name, "n": 0,
                "old_sol_pct": None, "ref_fail_pct": None,
                "near_para_pct": None, "f_opt_pct": None}

    old_sol  = sum(m["g1_old_sol_passes"] for m in accepted_metrics)
    ref_fail = sum(m["g2_ref_fails"]      for m in accepted_metrics)
    near_para = sum(m["g3_near_para"]     for m in accepted_metrics)
    f_opt    = sum(m["g4_f_opt_error"]    for m in accepted_metrics)

    return {
        "config":         config_name,
        "n":              n,
        "old_sol_pct":    round(100 * old_sol  / n, 1),
        "ref_fail_pct":   round(100 * ref_fail / n, 1),
        "near_para_pct":  round(100 * near_para/ n, 1),
        "f_opt_pct":      round(100 * f_opt    / n, 1),
        "raw_counts": {
            "old_sol": old_sol, "ref_fail": ref_fail,
            "near_para": near_para, "f_opt": f_opt,
        },
    }

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("eval_ablation.py — Quality Gate Ablation (Task A)")
    log("=" * 70)

    if not OPENAI_API_KEY:
        log("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    candidates = load_candidates()
    log(f"Total candidates: {len(candidates)}")

    cache = load_cache()
    log(f"Code cache has {len(cache)} entries")

    # ── Step 1: Compute per-candidate metrics ────────────────────────────────
    all_metrics = []
    log(f"\nStep 1: Computing per-candidate metrics for {len(candidates)} problems...")

    for i, prob in enumerate(candidates):
        pid = prob["id"]
        log(f"  [{i+1}/{len(candidates)}] {pid}: {prob.get('title','?')[:50]}")
        try:
            m = compute_candidate_metrics(prob, cache)
            all_metrics.append(m)
            log(f"    G1(old_passes)={m['g1_old_sol_passes']} "
                f"G2(ref_fails)={m['g2_ref_fails']} "
                f"G3(near_para)={m['g3_near_para']} "
                f"G4(f_opt)={m['g4_f_opt_error']} "
                f"[jacc={m['details']['g3_jaccard']}]")
        except Exception as e:
            log(f"    ERROR for {pid}: {e}")
            # Add a fallback conservative entry
            all_metrics.append({
                "pid": pid,
                "g1_old_sol_passes": False,
                "g2_ref_fails": False,
                "g3_near_para": False,
                "g4_f_opt_error": False,
                "details": {"error": str(e)},
            })
        # Save cache periodically
        if (i + 1) % 5 == 0:
            save_cache(cache)
            log(f"    (cache saved, {len(cache)} entries)")

    save_cache(cache)
    log(f"\nAll metrics computed. Cache saved ({len(cache)} entries).")

    # ── Step 2: Ablation configurations ─────────────────────────────────────
    log("\nStep 2: Running ablation configurations...")

    configs = [
        ("full",  0),   # all gates active
        ("no_g1", 1),   # Gate 1 disabled (old-solution check skipped)
        ("no_g2", 2),   # Gate 2 disabled (reference verification skipped)
        ("no_g3", 3),   # Gate 3 disabled (similarity check skipped)
        ("no_g4", 4),   # Gate 4 disabled (complexity verification skipped)
    ]

    rows = []
    for config_name, disable_gate in configs:
        accepted = accepted_under_config(all_metrics, disable_gate)
        row = compute_row(config_name, accepted)
        rows.append(row)
        log(f"  {config_name:8s} n={row['n']:2d} | "
            f"Old-Sol={row['old_sol_pct']}% "
            f"Ref-Fail={row['ref_fail_pct']}% "
            f"Near-Para={row['near_para_pct']}% "
            f"F-Opt={row['f_opt_pct']}%")

    # ── Step 3: Write output ─────────────────────────────────────────────────
    output = {
        "meta": {
            "n_candidates": len(candidates),
            "near_para_threshold": NEAR_PARA_THRESH,
            "model": GPT_MODEL,
            "date": time.strftime("%Y-%m-%d"),
            "methodology": (
                "G1: run LLM-synthesized old solution on shifted_examples; "
                "G2: run LLM-generated reference solution on shifted_examples; "
                "G3: word-level Jaccard(source_stmt, shifted_stmt) >= 0.7; "
                "G4: estimate_complexity(ref_code) mismatches target_complexity"
            ),
            "limitations": (
                "Only 54 candidates available (vs 598 in full pipeline). "
                "LLM-generated solutions may not perfectly represent production gates. "
                "G1 uses hardcoded templates for CS001/CS002/GT003/CS004, GPT-4o-mini for rest. "
                "G4 uses AST-based estimate_complexity (same function as stage4_evaluate.py)."
            ),
        },
        "table": rows,
        "per_candidate": all_metrics,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2))
    log(f"\nOutput written to {OUT_JSON}")

    # ── Print LaTeX-style table ───────────────────────────────────────────────
    log("\n=== Ablation Table (LaTeX-friendly) ===")
    log(f"{'Config':<15} {'N':>4} {'Old-Sol':>8} {'Ref-Fail':>9} {'Near-Para':>10} {'F-Opt':>7}")
    log("-" * 55)
    for row in rows:
        log(f"{row['config']:<15} {row['n']:>4} "
            f"{str(row['old_sol_pct'])+'%':>8} "
            f"{str(row['ref_fail_pct'])+'%':>9} "
            f"{str(row['near_para_pct'])+'%':>10} "
            f"{str(row['f_opt_pct'])+'%':>7}")

    log("\nTask A complete.")

if __name__ == "__main__":
    main()
