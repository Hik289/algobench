#!/usr/bin/env python3
"""
Stage 4: LLM Evaluation on ConstraintShift benchmark.
Models: GPT-4o mini (API), Qwen2.5-Coder-7B (local via ollama if available)
Strategies: Direct, CoT, RAG-source
Metrics: pass@1, pass@5, OptT, TrapRate
"""

import json, os, sys, time, re, ast, subprocess, tempfile, random
from pathlib import Path
from collections import defaultdict

BENCH_FILE   = "/path/to/algobench/data/final_benchmark.jsonl"
RESULTS_FILE = "/path/to/algobench/results/main_results.json"
LOG_FILE     = "/path/to/algobench/logs/stage4.log"
PYTHON       = "/path/to/venv"

# GPT-4o mini API settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o-mini"
N_SAMPLES      = 5   # for pass@5
TIMEOUT_CODE   = 5   # seconds

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ─── Prompt templates ──────────────────────────────────────────────────────

def fmt_examples(examples):
    parts = []
    for ex in examples:
        parts.append("Input:\n" + ex["input"] + "\nOutput:\n" + ex["output"])
    return "\n\n".join(parts)

def make_direct_prompt(prob):
    exs = fmt_examples(prob["shifted_examples"])
    return {
        "system": "You are an expert competitive programmer. Solve the problem below by writing a complete, correct, and efficient Python solution. Output ONLY the code, no explanation.",
        "user": (
            f"Problem: {prob['shifted_statement']}\n\n"
            f"Input format: {prob['shifted_input']}\n"
            f"Output format: {prob['shifted_output']}\n"
            f"Constraints: {prob['shifted_constraints']}\n\n"
            f"Examples:\n{exs}\n\n"
            "Write a complete Python solution."
        )
    }

def make_cot_prompt(prob):
    exs = fmt_examples(prob["shifted_examples"])
    return {
        "system": "You are an expert competitive programmer who reasons carefully before coding.",
        "user": (
            f"Problem: {prob['shifted_statement']}\n\n"
            f"Input format: {prob['shifted_input']}\n"
            f"Output format: {prob['shifted_output']}\n"
            f"Constraints: {prob['shifted_constraints']}\n\n"
            f"Examples:\n{exs}\n\n"
            "Reason step by step:\n"
            "1. Identify the key constraint that determines the required time complexity.\n"
            "2. Determine what time/space complexity class is needed.\n"
            "3. Select an appropriate algorithm or data structure.\n"
            "4. Check edge cases.\n"
            "5. Implement in Python.\n\n"
            "After your reasoning, output the complete Python code between ```python and ``` markers."
        )
    }

def make_rag_prompt(prob, all_problems):
    """RAG-source: find most similar source problem and prepend it."""
    candidates = [p for p in all_problems if p['id'] != prob['id'] and p['operator'] == prob['operator']]
    if not candidates:
        candidates = [p for p in all_problems if p['id'] != prob['id']]
    ref = random.choice(candidates) if candidates else None

    ref_text = ""
    if ref:
        ref_text = (
            "Reference problem (similar but DIFFERENT constraints):\n"
            + ref['shifted_statement'] + "\n"
            + "Reference solution approach: uses " + ref['target_algorithm']
            + " with complexity " + ref['target_complexity']['time'] + ".\n\n"
            "IMPORTANT: The new problem below has DIFFERENT constraints. Do NOT copy the reference solution.\n---\n\n"
        )
    exs = fmt_examples(prob["shifted_examples"])
    return {
        "system": "You are an expert competitive programmer. A similar reference problem is shown; adapt carefully.",
        "user": (
            ref_text
            + f"New problem to solve: {prob['shifted_statement']}\n\n"
            f"Input format: {prob['shifted_input']}\n"
            f"Output format: {prob['shifted_output']}\n"
            f"Constraints: {prob['shifted_constraints']}\n\n"
            f"Examples:\n{exs}\n\n"
            "Write a complete Python solution."
        )
    }

# ─── Code extraction ────────────────────────────────────────────────────────

def extract_code(text):
    """Extract Python code from LLM response."""
    # Try ```python ... ``` first
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ```...```
    m = re.search(r'```(.*?)```', text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if 'def ' in code or 'import ' in code or 'for ' in code or 'print(' in code:
            return code
    # Use the whole response if it looks like code
    if 'def ' in text or 'import ' in text or 'for ' in text:
        lines = [l for l in text.split('\n') if not l.startswith('#') or l.strip()]
        return '\n'.join(lines)
    return text

# ─── Code execution ─────────────────────────────────────────────────────────

def run_code(code, input_data, timeout=TIMEOUT_CODE):
    """Run code, return (stdout, passed, error)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, fname],
            input=input_data, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode == 0, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return "", False, "TLE"
    except Exception as e:
        return "", False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def check_correctness(code, examples):
    """Check code against all examples. Returns True if all pass."""
    if not code or not code.strip():
        return False, "empty code"
    for ex in examples:
        out, ok, err = run_code(code, ex['input'])
        if not ok:
            return False, f"runtime error: {err}"
        expected = ex['output'].strip()
        if out != expected:
            return False, f"WA: got '{out}', expected '{expected}'"
    return True, "all examples passed"

# ─── Complexity & trap analysis ─────────────────────────────────────────────

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

    # Pattern detection
    code_lower = code.lower()
    if any(p in code_lower for p in ['segment_tree', 'segtree', 'seg_tree', 'lazy', 'build(', '.update(', '.query(']):
        return "O(n log n) [segment tree]"
    if any(p in code_lower for p in ['fenwick', 'bit[', 'bit.update', 'lowbit', 'i & (-i)', 'i & -i']):
        return "O(n log n) [BIT/Fenwick]"
    if any(p in code_lower for p in ['heapq', 'heap', 'priority_queue', 'dijkstra']):
        return "O(n log n) [heap/Dijkstra]"
    if any(p in code_lower for p in ['mat_pow', 'matrix_power', 'mat_mult', 'matmul']):
        return "O(log n) [matrix exp]"
    if any(p in code_lower for p in ['bisect', 'binary_search', 'binary search']):
        return "O(n log n) [binary search]"
    if any(p in code_lower for p in ['merge_sort', 'mergesort']):
        return "O(n log n) [merge sort]"
    if 'sort(' in code_lower or '.sort(' in code_lower:
        return "O(n log n) [sort]"

    if max_depth >= 3:
        return "O(n³) or worse"
    if max_depth == 2:
        return "O(n²)"
    if max_depth == 1:
        return "O(n)"
    return "O(1) or O(log n)"

def check_trap(code, prob):
    """Check if code uses the old algorithm (trap patterns)."""
    code_lower = code.lower()
    patterns = prob.get("trap_patterns", [])
    hits = [p for p in patterns if p.lower() in code_lower]
    return len(hits) > 0, hits

def complexity_is_optimal(estimated, target):
    """Check if estimated complexity meets the target."""
    target_lower = target.lower()
    est_lower = estimated.lower()

    # Classify target complexity
    if "log n" in target_lower and "n log n" not in target_lower:
        # Target is O(log n) — very fast
        return "log" in est_lower and "n log" not in est_lower

    if "n log n" in target_lower:
        # Target is O(n log n) — accept O(n log n) or better
        optimal = ["o(n log n)", "o(n)", "o(log n)", "o(1)", "o(n log n) [sort]",
                   "o(n log n) [heap", "o(n log n) [bit", "o(n log n) [segment",
                   "o(n log n) [binary", "o(n log n) [merge"]
        return any(o in est_lower for o in optimal) or ("n log n" in est_lower and "n²" not in est_lower)

    if "o(n)" in target_lower or "o(n+q)" in target_lower:
        # Target is linear
        return "o(n)" in est_lower or "o(1)" in est_lower

    if "o(nw)" in target_lower or "o(n*w)" in target_lower:
        # DP knapsack — accept any polynomial
        return "n²" not in est_lower and "n³" not in est_lower

    # Default: not TLE (not n² or n³)
    return "n²" not in est_lower and "n³" not in est_lower and "worse" not in est_lower

# ─── GPT-4o mini API call ────────────────────────────────────────────────────

def call_gpt4o_mini(prompt, n=1, temperature=0.8):
    """Call GPT-4o mini API."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]

    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=2048,
        )
        return [c.message.content for c in resp.choices], None
    except Exception as e:
        return [], str(e)

# ─── Main evaluation loop ────────────────────────────────────────────────────

def load_benchmark():
    problems = []
    with open(BENCH_FILE) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line.strip()))
    return problems

def evaluate_problem(prob, all_problems, model_name, strategy):
    """Evaluate one problem × model × strategy. Returns dict of metrics."""
    pid = prob["id"]
    log(f"  Evaluating {pid} | model={model_name} | strategy={strategy}")

    # Build prompt
    if strategy == "direct":
        prompt = make_direct_prompt(prob)
    elif strategy == "cot":
        prompt = make_cot_prompt(prob)
    elif strategy == "rag":
        prompt = make_rag_prompt(prob, all_problems)
    else:
        prompt = make_direct_prompt(prob)

    # Call model
    if "gpt" in model_name:
        if not OPENAI_API_KEY:
            log(f"    SKIP: no OPENAI_API_KEY set")
            return None
        responses, err = call_gpt4o_mini(prompt, n=N_SAMPLES)
    else:
        log(f"    SKIP: model {model_name} not yet configured for local inference")
        return None

    if err or not responses:
        log(f"    API error: {err}")
        return {"error": err, "pass_at_1": None, "pass_at_5": None}

    time.sleep(0.5)  # rate limit

    # Evaluate each sample
    results = []
    codes = []
    for resp in responses:
        code = extract_code(resp)
        codes.append(code)
        correct, reason = check_correctness(code, prob["shifted_examples"])
        is_trap, trap_hits = check_trap(code, prob)
        complexity = estimate_complexity(code)
        is_optimal = complexity_is_optimal(complexity, prob["target_complexity"]["time"])
        results.append({
            "correct": correct,
            "reason": reason,
            "is_trap": is_trap,
            "trap_hits": trap_hits,
            "estimated_complexity": complexity,
            "is_optimal": is_optimal,
        })

    # Compute metrics
    n_correct = sum(r["correct"] for r in results)
    n_trap_wrong = sum(r["is_trap"] for r in results if not r["correct"])
    n_wrong = sum(not r["correct"] for r in results)
    n_optimal = sum(r["is_optimal"] for r in results if r["correct"])
    n_correct_total = sum(r["correct"] for r in results)

    pass_at_1 = results[0]["correct"] if results else False
    pass_at_5 = n_correct > 0
    trap_rate  = n_trap_wrong / n_wrong if n_wrong > 0 else 0.0
    opt_t      = n_optimal / n_correct_total if n_correct_total > 0 else 0.0

    log(f"    pass@1={pass_at_1}, pass@5={pass_at_5}, TrapRate={trap_rate:.2f}, OptT={opt_t:.2f}")

    return {
        "pid": pid,
        "model": model_name,
        "strategy": strategy,
        "n_samples": len(results),
        "pass_at_1": pass_at_1,
        "pass_at_5": pass_at_5,
        "trap_rate": trap_rate,
        "opt_t": opt_t,
        "sample_results": results,
    }

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log("Stage 4: LLM Evaluation starting...")

    if not OPENAI_API_KEY:
        log("ERROR: OPENAI_API_KEY not set. Export it before running.")
        log("Usage: OPENAI_API_KEY=sk-... python3 stage4_evaluate.py")
        sys.exit(1)

    problems = load_benchmark()
    log(f"Loaded {len(problems)} benchmark problems")

    MODELS     = ["gpt-4o-mini"]
    STRATEGIES = ["direct", "cot", "rag"]

    all_results = {}

    for prob in problems:
        pid = prob["id"]
        all_results[pid] = {"problem_title": prob["title"], "operator": prob["operator"], "evals": {}}

        for model in MODELS:
            for strategy in STRATEGIES:
                key = f"{model}_{strategy}"
                result = evaluate_problem(prob, problems, model, strategy)
                if result:
                    all_results[pid]["evals"][key] = result

                # Save incrementally
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

    log(f"\nStage 4 complete. Results saved to {RESULTS_FILE}")

    # Print summary
    log("\n=== Summary ===")
    for model in MODELS:
        for strategy in STRATEGIES:
            key = f"{model}_{strategy}"
            pass1_vals = []
            trap_vals  = []
            optt_vals  = []
            for pid, data in all_results.items():
                ev = data["evals"].get(key)
                if ev and ev.get("pass_at_1") is not None:
                    pass1_vals.append(float(ev["pass_at_1"]))
                    trap_vals.append(ev.get("trap_rate", 0))
                    optt_vals.append(ev.get("opt_t", 0))
            if pass1_vals:
                n = len(pass1_vals)
                log(f"  {model} / {strategy}: pass@1={sum(pass1_vals)/n:.1%} "
                    f"TrapRate={sum(trap_vals)/n:.1%} OptT={sum(optt_vals)/n:.1%} (n={n})")

if __name__ == "__main__":
    main()
