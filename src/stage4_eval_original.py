#!/usr/bin/env python3
"""
Evaluate GPT-4o mini on ORIGINAL (un-shifted) problems.
Provides the "Original" column baseline for Table 1.
"""
import json, os, sys, time, re, ast, subprocess, tempfile
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent
BENCH_FILE   = str(BASE_DIR / "data" / "final_benchmark.jsonl")
RESULTS_FILE = str(BASE_DIR / "results" / "original_results.json")
LOG_FILE     = str(BASE_DIR / "logs" / "eval_original.log")
PYTHON       = sys.executable

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o-mini"
N_SAMPLES      = 5
TIMEOUT_CODE   = 5

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def fmt_examples(examples):
    return "\n\n".join("Input:\n" + e["input"] + "\nOutput:\n" + e["output"]
                       for e in examples)

def make_direct_prompt_original(prob):
    exs = fmt_examples(prob.get("source_examples", []))
    return {
        "system": "You are an expert competitive programmer. Solve the problem below by writing a complete, correct, and efficient Python solution. Output ONLY the code, no explanation.",
        "user": (
            f"Problem: {prob['source_statement']}\n\n"
            f"Input format: {prob.get('source_input','')}\n"
            f"Output format: {prob.get('source_output','')}\n"
            f"Constraints: {prob['source_constraints']}\n\n"
            f"Examples:\n{exs}\n\n"
            "Write a complete Python solution."
        )
    }

def extract_code(text):
    import re
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    if 'def ' in text or 'import ' in text or 'for ' in text:
        return text.strip()
    return text

def run_code(code, input_data, timeout=TIMEOUT_CODE):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code); fname = f.name
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

def check_correctness(code, examples):
    if not code or not code.strip():
        return False, "empty code"
    for ex in examples:
        out, ok, err = run_code(code, ex['input'])
        if not ok: return False, f"error: {err}"
        if out != ex['output'].strip():
            return False, f"WA: got '{out}', expected '{ex['output'].strip()}'"
    return True, "ok"

def estimate_complexity(code):
    try: tree = ast.parse(code)
    except: return "UNKNOWN"
    depth = [0]
    def walk(node, d=0):
        if isinstance(node, (ast.For, ast.While)): d += 1
        depth[0] = max(depth[0], d)
        for c in ast.iter_child_nodes(node): walk(c, d)
    walk(tree)
    c = code.lower()
    if any(p in c for p in ['heapq','heap','dijkstra']): return "O(n log n)"
    if any(p in c for p in ['segment','segtree','lazy']): return "O(n log n)"
    if any(p in c for p in ['fenwick','bit[','lowbit']): return "O(n log n)"
    if any(p in c for p in ['bisect','binary_search']): return "O(n log n)"
    if 'sort(' in c or '.sort(' in c: return "O(n log n)"
    if depth[0] >= 2: return "O(n²)"
    if depth[0] == 1: return "O(n)"
    return "O(1)"

def complexity_optimal(est, target):
    e, t = est.lower(), target.lower()
    if "n log n" in t: return "n²" not in e and "n³" not in e
    if "o(n)" in t: return "o(n)" in e or "o(1)" in e
    return "n²" not in e and "n³" not in e

def call_api(prompt, n=1):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role":"system","content":prompt["system"]},
                      {"role":"user","content":prompt["user"]}],
            n=n, temperature=0.8, max_tokens=2048)
        return [c.message.content for c in resp.choices], None
    except Exception as e:
        return [], str(e)

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    if not OPENAI_API_KEY:
        log("ERROR: OPENAI_API_KEY not set"); sys.exit(1)

    problems = []
    with open(BENCH_FILE) as f:
        for line in f:
            if line.strip(): problems.append(json.loads(line))
    log(f"Evaluating {len(problems)} problems on ORIGINAL statements...")

    all_results = {}
    for prob in problems:
        pid = prob["id"]
        log(f"  {pid}: {prob['title']}")
        prompt = make_direct_prompt_original(prob)
        responses, err = call_api(prompt, n=N_SAMPLES)
        time.sleep(0.5)
        if err or not responses:
            log(f"    API error: {err}"); continue

        sample_results = []
        for resp in responses:
            code = extract_code(resp)
            correct, reason = check_correctness(code, prob.get("source_examples", []))
            est = estimate_complexity(code)
            opt = complexity_optimal(est, prob["source_complexity"]["time"])
            sample_results.append({"correct": correct, "reason": reason,
                                    "complexity": est, "optimal": opt})

        n_correct = sum(r["correct"] for r in sample_results)
        n_opt     = sum(r["optimal"] for r in sample_results if r["correct"])
        pass1 = sample_results[0]["correct"]
        pass5 = n_correct > 0
        optt  = n_opt / n_correct if n_correct else 0.0

        all_results[pid] = {
            "title": prob["title"], "operator": prob["operator"],
            "pass_at_1": pass1, "pass_at_5": pass5, "opt_t": optt,
            "n_correct": n_correct, "n_samples": len(sample_results),
            "sample_results": sample_results,
        }
        log(f"    pass@1={pass1} pass@5={pass5} OptT={optt:.2f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    vals = [v for v in all_results.values()]
    n = len(vals)
    if n:
        avg_p1  = sum(float(v["pass_at_1"]) for v in vals) / n
        avg_p5  = sum(float(v["pass_at_5"]) for v in vals) / n
        avg_opt = sum(v["opt_t"] for v in vals) / n
        log(f"\nOriginal problems summary (n={n}):")
        log(f"  pass@1={avg_p1:.1%}  pass@5={avg_p5:.1%}  OptT={avg_opt:.1%}")

    log(f"Done. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
