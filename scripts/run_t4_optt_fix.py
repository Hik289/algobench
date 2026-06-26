#!/usr/bin/env python3
"""
T4: Re-run GPT-5.4 and Claude Opus 4.5 on ORIG-11 + HARD-9 (n=20) with proper OptT checker.

Fix: Use FULL complexity_optimal checker (static+tags layers) from eval_checker.py,
     not the buggy strict regex-only version in stage4_multimodel.py.

Both modes: shifted AND original.
N=5 samples per cell.
Direct prompting only.

Usage:
  python3 scripts/run_t4_optt_fix.py
"""

import ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
BENCH_FILE  = BASE_DIR / "data" / "final_benchmark.jsonl"
RESULTS_DIR = BASE_DIR / "results"
LOG_FILE    = BASE_DIR / "logs" / "run_t4_optt_fix.log"
OUT_FILE    = RESULTS_DIR / "multimodel_results.json"
ANALYSIS    = BASE_DIR / "analysis"

PYTHON = sys.executable
N_SAMPLES = 5
TIMEOUT_CODE = 5

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

# Target 20 problems: ORIG-11 + HARD-9
TARGET_20 = [
    'CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002',
    'GT001','GT002','GT003',                              # ORIG-11
    'CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2',
    'SD_H2','OP_H2','CS_H3'                               # HARD-9
]

# Model configs
MODELS_TO_RUN = [
    ("gpt-5.4", "openai"),
    # claude-opus-4-5: ANTHROPIC_API_KEY is invalid (401). 
    # Recalculate from stored data via recalc_opus_optt.py instead.
    # ("claude-opus-4-5", "anthropic"),
]

# ── Logging ─────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── FULL complexity checker (static + tags layers from eval_checker.py) ──────

def estimate_complexity_full(code):
    """Full static+tags complexity estimator (from eval_checker.py)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "UNKNOWN"
    md = [0]
    def cd(node, d=0):
        if isinstance(node, (ast.For, ast.While)):
            d += 1
            md[0] = max(md[0], d)
        for c in ast.iter_child_nodes(node):
            cd(c, d)
    cd(tree)
    cl = code.lower()
    # Keyword tags
    if any(p in cl for p in ['segment_tree','segtree','seg_tree','lazy','.update(','.query(']):
        return "O(n log n) [seg_tree]"
    if any(p in cl for p in ['fenwick','bit[','bit.update','lowbit','i & (-i)','i & -i']):
        return "O(n log n) [fenwick]"
    if any(p in cl for p in ['heapq','heappush','heappop','dijkstra','priority_queue']):
        return "O(n log n) [heap]"
    if any(p in cl for p in ['matrix_pow','mat_pow','matrix_power','matmul','matrix_exp','np.linalg']):
        return "O(log n) [matrix_exp]"
    if any(p in cl for p in ['union_find','dsu','parent[','find(']):
        return "O(n alpha) [dsu]"
    if any(p in cl for p in ['bisect','binary_search','mid = (','mid = l +']):
        return "O(n log n) [binary_search]"
    if 'sort(' in cl or '.sort(' in cl or 'sorted(' in cl:
        return "O(n log n) [sort]"
    return {"0": "O(1)", "1": "O(n)", "2": "O(n^2)", "3": "O(n^3)"}.get(
        str(min(md[0], 3)), "O(n^3)")

def is_optimal_full(est, target):
    """Full complexity checker (improved from eval_checker.py + fixes)."""
    if "UNCERTAIN" in est or "UNKNOWN" in est:
        return "UNCERTAIN"
    t = target.lower()
    e = est.lower()
    # Exact match → always optimal
    if e == t:
        return True
    # DP-style targets: O(nW), O(kW), O(n^2...) are valid polynomial targets
    if re.search(r'n[\^*]2|nw|n\*w|\bkw\b', t):
        return True
    # Reject clearly bad complexities
    if "n^2" in e or "n^3" in e or "tle" in e or "n²" in e or "n³" in e:
        return False
    # Tight log-only targets
    if re.search(r'o\(\s*log\s*n\)', t) and "n log n" not in t:
        return "log" in e and "n log n" not in e
    # Any log target (n log n, etc.)
    if "log" in t:
        return True
    # Linear/near-linear targets: accept O(n), O(1), O(log n), O(n+...) estimates
    if re.search(r'o\(\s*(n\b|n\s*[\+\-])', t):
        return "o(n)" in e or "o(1)" in e or "log" in e or "o(n" in e
    return True

# ── Prompt builder ───────────────────────────────────────────────────────────

def fmt_examples(examples):
    return "\n\n".join(
        "Input:\n" + e["input"] + "\nOutput:\n" + e["output"]
        for e in examples if "input" in e and "output" in e
    )

def make_direct_prompt(prob, use_shifted=True):
    prefix = "shifted" if use_shifted else "source"
    stmt = prob.get(f"{prefix}_statement", "")
    cons = prob.get(f"{prefix}_constraints", "")
    inp  = prob.get(f"{prefix}_input", "")
    out  = prob.get(f"{prefix}_output", "")
    exs  = fmt_examples(prob.get(f"{prefix}_examples", []))

    sys_msg = ("You are an expert competitive programmer. Solve the problem below by "
               "writing a complete, correct, and efficient Python solution. "
               "Output ONLY the code, no explanation.")
    user = (f"Problem: {stmt}\n\nInput format: {inp}\nOutput format: {out}\n"
            f"Constraints: {cons}\n\nExamples:\n{exs}\n\nWrite a complete Python solution.")
    return {"system": sys_msg, "user": user}

# ── Model callers ────────────────────────────────────────────────────────────

def call_openai(prompt, model, n=N_SAMPLES):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Newer GPT-5.x models require max_completion_tokens; older models use max_tokens
    NEWER_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5",
                    "o1", "o1-pro", "o3", "o3-mini"}
    use_new_param = any(model.startswith(m) for m in NEWER_MODELS)
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": prompt["system"]},
                      {"role": "user",   "content": prompt["user"]}],
            n=n, temperature=0.8,
        )
        if use_new_param:
            kwargs["max_completion_tokens"] = 2048
        else:
            kwargs["max_tokens"] = 2048
        resp = client.chat.completions.create(**kwargs)
        return [c.message.content for c in resp.choices], None
    except Exception as e:
        return [], str(e)

def call_anthropic(prompt, model, n=N_SAMPLES):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []
    for _ in range(n):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}])
            results.append(resp.content[0].text)
        except Exception as e:
            log(f"    Anthropic error: {e}")
            time.sleep(2)
    return results, None if results else "no responses"

def call_model(prompt, model, backend, n=N_SAMPLES):
    if backend == "openai":    return call_openai(prompt, model, n)
    if backend == "anthropic": return call_anthropic(prompt, model, n)
    return [], f"unknown backend {backend}"

# ── Code execution ────────────────────────────────────────────────────────────

def extract_code(text):
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    if any(k in text for k in ['def ', 'import ', 'for ', 'print(']):
        return text.strip()
    return text

def run_code(code, input_data, timeout=TIMEOUT_CODE):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run([PYTHON, fname], input=input_data,
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode == 0, r.stderr[:200]
    except subprocess.TimeoutExpired:
        return "", False, "TLE"
    except Exception as e:
        return "", False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def check_correct(code, examples):
    if not code or not code.strip():
        return False, "empty"
    for ex in examples:
        if "input" not in ex or "output" not in ex:
            continue
        out, ok, err = run_code(code, ex['input'])
        if not ok:
            return False, f"error:{err}"
        if out != ex['output'].strip():
            return False, f"WA:got'{out[:80]}'"
    return True, "ok"

def has_trap(code, patterns):
    if not patterns:
        return False, []
    cl = code.lower()
    hits = [p for p in patterns if p.lower() in cl]
    return bool(hits), hits

# ── Single evaluation ────────────────────────────────────────────────────────

def evaluate_one(prob, model, backend, use_shifted=True, n=N_SAMPLES):
    prompt   = make_direct_prompt(prob, use_shifted)
    prefix   = "shifted" if use_shifted else "source"
    examples = prob.get(f"{prefix}_examples", [])

    if use_shifted:
        target   = prob["target_complexity"]["time"]
        patterns = prob.get("trap_patterns", [])
    else:
        target   = prob["source_complexity"]["time"]
        patterns = []

    responses, err = call_model(prompt, model, backend, n=n)
    time.sleep(0.5)  # Rate limit courtesy

    if err or not responses:
        return {"error": err or "no_responses", "n_samples": 0}

    sample_results = []
    for resp in responses:
        code  = extract_code(resp)
        correct, reason = check_correct(code, examples)
        est   = estimate_complexity_full(code)
        opt_v = is_optimal_full(est, target)
        opt   = opt_v is True  # UNCERTAIN → False for opt_t calc
        trap, trap_hits = has_trap(code, patterns)
        sample_results.append({
            "correct":     correct,
            "reason":      reason,
            "complexity":  est,
            "optimal":     opt,
            "trap":        trap,
            "trap_hits":   trap_hits,
        })

    n_correct = sum(r["correct"] for r in sample_results)
    n_wrong   = len(sample_results) - n_correct
    n_opt     = sum(r["optimal"] for r in sample_results if r["correct"])
    n_trap_w  = sum(r["trap"]    for r in sample_results if not r["correct"])

    return {
        "pass_at_1":      sample_results[0]["correct"],
        "pass_at_5":      n_correct > 0,
        "opt_t":          n_opt / n_correct if n_correct else 0.0,
        "trap_rate":      n_trap_w / n_wrong if n_wrong else 0.0,
        "n_samples":      len(sample_results),
        "sample_results": sample_results,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("T4: Re-run GPT-5.4 + Claude Opus 4.5 with full OptT checker")
    log("=" * 70)
    log(f"Target problems: {len(TARGET_20)}")
    log(f"N_SAMPLES per cell: {N_SAMPLES}")

    # Check API keys
    if not OPENAI_API_KEY:
        log("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    if not ANTHROPIC_API_KEY:
        log("NOTE: ANTHROPIC_API_KEY not set (claude-opus-4-5 skipped; handled by recalc_opus_optt.py)")

    # Install anthropic if needed
    try:
        import anthropic
    except ImportError:
        log("Installing anthropic...")
        subprocess.run([PYTHON, "-m", "pip", "install", "-q", "anthropic"], check=True)
        import anthropic  # noqa

    # Load benchmark
    problems = {}
    with open(BENCH_FILE) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                problems[p["id"]] = p
    log(f"Loaded {len(problems)} benchmark problems")

    target_probs = [problems[pid] for pid in TARGET_20 if pid in problems]
    log(f"Target problems found in benchmark: {len(target_probs)}")
    missing = [pid for pid in TARGET_20 if pid not in problems]
    if missing:
        log(f"WARNING: Missing from benchmark: {missing}")

    # Load existing results (preserve other 22 keys)
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            all_results = json.load(f)
        log(f"Loaded existing results: {len(all_results)} keys")
    else:
        all_results = {}

    # Back up current data before overwriting
    import shutil
    backup = str(OUT_FILE) + ".bak_before_t4"
    if not Path(backup).exists():
        shutil.copy(OUT_FILE, backup)
        log(f"Backed up to {backup}")

    # ── Run models ────────────────────────────────────────────────────────────
    for model, backend in MODELS_TO_RUN:
        key = f"{model}__direct"
        log(f"\n{'='*60}")
        log(f"Model: {model}  Backend: {backend}  Key: {key}")
        log(f"{'='*60}")

        # Initialize key with a clean dict (we'll re-run all 20)
        # Preserve non-target-20 problem entries (other problems may have valid data)
        existing = all_results.get(key, {})
        new_entry = {}

        # Copy non-TARGET_20 entries if any
        for pid, val in existing.items():
            if pid not in TARGET_20:
                new_entry[pid] = val

        for i, prob in enumerate(target_probs):
            pid = prob["id"]
            log(f"\n  [{i+1}/{len(target_probs)}] {pid} — {prob.get('title','')[:50]}")

            # ── Shifted evaluation ─────────────────────────────────────────
            log(f"    shifted ({prob['target_complexity']['time']})...")
            ev_shifted = evaluate_one(prob, model, backend, use_shifted=True)
            if "error" not in ev_shifted:
                log(f"    shifted → p@1={ev_shifted['pass_at_1']} p@5={ev_shifted['pass_at_5']} "
                    f"opt_t={ev_shifted['opt_t']:.2f} n={ev_shifted['n_samples']}")
            else:
                log(f"    shifted → ERROR: {ev_shifted['error']}")

            # ── Original evaluation ────────────────────────────────────────
            log(f"    original ({prob['source_complexity']['time']})...")
            ev_original = evaluate_one(prob, model, backend, use_shifted=False)
            if "error" not in ev_original:
                log(f"    original → p@1={ev_original['pass_at_1']} "
                    f"opt_t={ev_original['opt_t']:.2f} n={ev_original['n_samples']}")
            else:
                log(f"    original → ERROR: {ev_original['error']}")

            new_entry[pid] = {
                "pid":      pid,
                "title":    prob.get("title", ""),
                "operator": prob.get("operator", ""),
                "model":    model,
                "strategy": "direct",
                "shifted":  ev_shifted,
                "original": ev_original,
            }

            # Save incrementally after each problem
            all_results[key] = new_entry
            with open(OUT_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            log(f"    Saved incrementally (key={key}, n={len(new_entry)} entries)")

        log(f"\n  {model} complete: {len(new_entry)} problems processed")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "="*70)
    log("T4 SUMMARY")
    log("="*70)
    for model, backend in MODELS_TO_RUN:
        key = f"{model}__direct"
        probs_data = all_results.get(key, {})
        t20_data   = {k: v for k, v in probs_data.items() if k in TARGET_20}
        n = len(t20_data)

        # Shifted stats
        shf_evs = [v.get("shifted", {}) for v in t20_data.values()]
        shf_evs = [e for e in shf_evs if "error" not in e and "pass_at_1" in e]
        shf_p1  = sum(e["pass_at_1"] for e in shf_evs) / max(len(shf_evs),1) * 100
        shf_opt = sum(e["opt_t"]     for e in shf_evs) / max(len(shf_evs),1) * 100

        # Original stats
        orig_evs = [v.get("original", {}) for v in t20_data.values()]
        orig_evs = [e for e in orig_evs if "error" not in e and "pass_at_1" in e]
        orig_p1  = sum(e["pass_at_1"] for e in orig_evs) / max(len(orig_evs),1) * 100
        orig_opt = sum(e["opt_t"]     for e in orig_evs) / max(len(orig_evs),1) * 100

        log(f"  {model}:")
        log(f"    Shifted  (n={len(shf_evs)}):  p@1={shf_p1:.1f}%  opt_t={shf_opt:.1f}%")
        log(f"    Original (n={len(orig_evs)}): p@1={orig_p1:.1f}%  opt_t={orig_opt:.1f}%")

    # Write opt_t update summary for analysis
    opt_t_update = {}
    for model, backend in MODELS_TO_RUN:
        key = f"{model}__direct"
        probs_data = all_results.get(key, {})
        t20_data   = {k: v for k, v in probs_data.items() if k in TARGET_20}

        shf_evs  = [v.get("shifted",  {}) for v in t20_data.values()]
        orig_evs = [v.get("original", {}) for v in t20_data.values()]
        shf_evs  = [e for e in shf_evs  if "error" not in e and "pass_at_1" in e]
        orig_evs = [e for e in orig_evs if "error" not in e and "pass_at_1" in e]

        opt_t_update[model] = {
            "shifted_opt_t_mean":  sum(e["opt_t"] for e in shf_evs)  / max(len(shf_evs),1),
            "original_opt_t_mean": sum(e["opt_t"] for e in orig_evs) / max(len(orig_evs),1),
            "n_shifted":  len(shf_evs),
            "n_original": len(orig_evs),
        }

    out_analysis = ANALYSIS / "table_main_opt_t_update.json"
    ANALYSIS.mkdir(exist_ok=True)
    with open(out_analysis, "w") as f:
        json.dump(opt_t_update, f, indent=2)
    log(f"\nOptT update written to {out_analysis}")
    log(f"Results saved to {OUT_FILE}")
    log("\nT4 COMPLETE. Now run scripts/recompute_tables.py to update table_main.json")

if __name__ == "__main__":
    main()
