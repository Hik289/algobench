#!/usr/bin/env python3
"""
Stage 4 (Multi-Model): Evaluate all available models on AlgoBench.

Models:
  API:   gpt-4o, gpt-4o-mini  (OPENAI_API_KEY)
         gemini-2.0-flash       (GEMINI_API_KEY)
  Local: qwen2.5-coder:7b      (Ollama)
         codellama:34b          (Ollama, optional — large)

Strategies: direct, cot, rag (3 strategies)
Metrics:    pass@1, pass@5, OptT, TrapRate

Usage:
  python3 stage4_multimodel.py [--models gpt-4o gpt-4o-mini gemini] [--strategies direct cot]
"""
import argparse, ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
from collections import defaultdict

BASE_DIR     = Path(__file__).resolve().parent.parent
BENCH_FILE   = str(BASE_DIR / "data"    / "final_benchmark.jsonl")
ORIG_FILE    = str(BASE_DIR / "results" / "original_results.json")
RESULTS_DIR  = str(BASE_DIR / "results")
LOG_FILE     = str(BASE_DIR / "logs"    / "stage4_multi.log")
PYTHON       = sys.executable

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
N_SAMPLES      = 5
TIMEOUT_CODE   = 5

ALL_MODELS = {
    "gpt-4o":                        "openai",
    "gpt-4o-mini":                   "openai",
    "gemini-2.0-flash":              "gemini",
    "Qwen/Qwen2.5-7B-Instruct":      "hf_local",
}

# ── Logging ────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Prompt builders ────────────────────────────────────────────────────────

def fmt_examples(examples):
    return "\n\n".join("Input:\n" + e["input"] + "\nOutput:\n" + e["output"]
                       for e in examples)

def make_prompt(prob, strategy, all_problems, use_shifted=True):
    prefix = "shifted" if use_shifted else "source"
    stmt  = prob[f"{prefix}_statement"]
    cons  = prob[f"{prefix}_constraints"]
    inp   = prob.get(f"{prefix}_input", "")
    out   = prob.get(f"{prefix}_output", "")
    exs   = fmt_examples(prob.get(f"{prefix}_examples", []))

    sys_msg = "You are an expert competitive programmer. Solve the problem below by writing a complete, correct, and efficient Python solution. Output ONLY the code, no explanation."

    if strategy == "direct":
        user = (f"Problem: {stmt}\n\nInput format: {inp}\nOutput format: {out}\n"
                f"Constraints: {cons}\n\nExamples:\n{exs}\n\nWrite a complete Python solution.")

    elif strategy == "cot":
        sys_msg = "You are an expert competitive programmer who reasons carefully before coding."
        user = (f"Problem: {stmt}\n\nInput format: {inp}\nOutput format: {out}\n"
                f"Constraints: {cons}\n\nExamples:\n{exs}\n\n"
                "Reason step by step:\n"
                "1. Identify the key constraint determining required time complexity.\n"
                "2. Determine what complexity class is needed.\n"
                "3. Select an appropriate algorithm or data structure.\n"
                "4. Check edge cases.\n"
                "5. Implement in Python.\n\n"
                "After reasoning, output the complete Python code between ```python and ``` markers.")

    elif strategy == "rag":
        candidates = [p for p in all_problems if p["id"] != prob["id"]
                      and p["operator"] == prob["operator"]]
        if not candidates:
            candidates = [p for p in all_problems if p["id"] != prob["id"]]
        import random; ref = random.choice(candidates) if candidates else None
        ref_text = ""
        if ref:
            ref_text = (f"Reference problem (similar but DIFFERENT constraints):\n"
                        f"{ref['shifted_statement']}\n"
                        f"Reference solution uses: {ref['target_algorithm']} "
                        f"({ref['target_complexity']['time']})\n\n"
                        "IMPORTANT: new problem has DIFFERENT constraints. Do NOT copy reference.\n---\n\n")
        sys_msg = "You are an expert competitive programmer. A reference problem is shown; adapt carefully."
        user = (ref_text + f"New problem: {stmt}\n\nInput format: {inp}\n"
                f"Output format: {out}\nConstraints: {cons}\n\nExamples:\n{exs}\n\n"
                "Write a complete Python solution.")

    return {"system": sys_msg, "user": user}

# ── Model callers ──────────────────────────────────────────────────────────

def call_openai(prompt, model, n=N_SAMPLES):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt["system"]},
                      {"role": "user",   "content": prompt["user"]}],
            n=n, temperature=0.8, max_tokens=2048)
        return [c.message.content for c in resp.choices], None
    except Exception as e:
        return [], str(e)

def call_gemini(prompt, model, n=N_SAMPLES):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gm = genai.GenerativeModel(
        model,
        system_instruction=prompt["system"])
    results = []
    for _ in range(n):
        try:
            r = gm.generate_content(prompt["user"],
                generation_config={"temperature": 0.8, "max_output_tokens": 2048})
            results.append(r.text)
        except Exception as e:
            log(f"    Gemini error: {e}")
        time.sleep(0.3)
    return results, None if results else "no responses"

def call_hf_local(prompt, model, n=N_SAMPLES):
    """Run local HuggingFace model with 4-bit quantization (bitsandbytes)."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        return [], f"import error: {e}"

    log(f"    Loading {model} in 4-bit NF4...")
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
        mdl.eval()
    except Exception as e:
        return [], f"load error: {e}"

    messages = [{"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(mdl.device)

    results = []
    for _ in range(min(n, 3)):
        try:
            with torch.no_grad():
                out = mdl.generate(
                    **inputs, max_new_tokens=1024,
                    do_sample=True, temperature=0.8,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            results.append(gen)
        except Exception as e:
            log(f"    inference error: {e}"); break
    del mdl, tok
    torch.cuda.empty_cache()
    return results, (None if results else "hf_local failed")

def call_model(prompt, model, backend):
    if backend == "openai":   return call_openai(prompt, model)
    if backend == "gemini":   return call_gemini(prompt, model)
    if backend == "hf_local": return call_hf_local(prompt, model)
    return [], f"unknown backend {backend}"

# ── Code execution + metrics ───────────────────────────────────────────────

def extract_code(text):
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    if any(k in text for k in ['def ', 'import ', 'for ', 'print(']):
        return text.strip()
    return text

def run_code(code, input_data, timeout=TIMEOUT_CODE):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code); fname = f.name
    try:
        r = subprocess.run([PYTHON, fname], input=input_data,
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode == 0, r.stderr[:200]
    except subprocess.TimeoutExpired: return "", False, "TLE"
    except Exception as e:            return "", False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def check_correct(code, examples):
    if not code or not code.strip(): return False, "empty"
    for ex in examples:
        out, ok, err = run_code(code, ex['input'])
        if not ok: return False, f"error:{err}"
        if out != ex['output'].strip(): return False, f"WA:got'{out}'"
    return True, "ok"

def complexity_exponent(c):
    c = c.lower()
    if "n²" in c or "n^2" in c: return 2.0
    if "n log n" in c:          return 1.5
    if "o(n)" in c:             return 1.0
    if "o(log n)" in c:         return 0.5
    if "o(nw)" in c:            return 1.8
    return 1.0

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
    for pat, label in [(['heapq','heap','dijkstra'], "O(n log n)"),
                       (['segment','segtree','lazy'], "O(n log n)"),
                       (['fenwick','bit[','lowbit'],  "O(n log n)"),
                       (['bisect'],                   "O(n log n)"),
                       (['mat_pow','matrix_power'],   "O(log n)"),]:
        if any(p in c for p in pat): return label
    if 'sort(' in c or '.sort(' in c: return "O(n log n)"
    return {0:"O(1)", 1:"O(n)", 2:"O(n²)"}.get(depth[0], "O(n³)")

def is_optimal(est, target):
    e, t = est.lower(), target.lower()
    if "n log n" in t: return "n²" not in e and "n³" not in e
    if "o(n)" in t:    return "o(n)" in e or "o(1)" in e
    return "n²" not in e and "n³" not in e

def has_trap(code, patterns):
    cl = code.lower()
    return any(p.lower() in cl for p in patterns), \
           [p for p in patterns if p.lower() in cl]

# ── Single evaluation ──────────────────────────────────────────────────────

def evaluate_one(prob, all_probs, model, backend, strategy,
                 use_shifted=True):
    prompt   = make_prompt(prob, strategy, all_probs, use_shifted)
    examples = prob.get("shifted_examples" if use_shifted else "source_examples", [])
    target   = prob["target_complexity"]["time"] if use_shifted else \
               prob["source_complexity"]["time"]
    patterns = prob.get("trap_patterns", []) if use_shifted else []

    responses, err = call_model(prompt, model, backend)
    time.sleep(0.4)
    if err or not responses:
        return {"error": err}

    sample_results = []
    for resp in responses:
        code    = extract_code(resp)
        correct, reason = check_correct(code, examples)
        est     = estimate_complexity(code)
        opt     = is_optimal(est, target)
        trap, trap_hits = has_trap(code, patterns)
        sample_results.append({"correct": correct, "reason": reason,
                                "complexity": est, "optimal": opt,
                                "trap": trap, "trap_hits": trap_hits})

    n_correct = sum(r["correct"] for r in sample_results)
    n_wrong   = len(sample_results) - n_correct
    n_opt     = sum(r["optimal"]  for r in sample_results if r["correct"])
    n_trap_w  = sum(r["trap"]     for r in sample_results if not r["correct"])

    return {
        "pass_at_1":  sample_results[0]["correct"],
        "pass_at_5":  n_correct > 0,
        "opt_t":      n_opt / n_correct if n_correct else 0.0,
        "trap_rate":  n_trap_w / n_wrong if n_wrong else 0.0,
        "n_samples":  len(sample_results),
        "sample_results": sample_results,
    }

# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "Qwen/Qwen2.5-7B-Instruct"],
                   help="Models to evaluate")
    p.add_argument("--strategies", nargs="+",
                   default=["direct", "cot", "rag"])
    p.add_argument("--also-original", action="store_true", default=True,
                   help="Also evaluate on original (un-shifted) problems")
    p.add_argument("--output", default=None)
    return p.parse_args()

def main():
    args  = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Install google-generativeai if needed
    if any(ALL_MODELS.get(m) == "gemini" for m in args.models):
        subprocess.run([PYTHON, "-m", "pip", "install", "-q", "google-generativeai"],
                       capture_output=True)

    problems = []
    with open(BENCH_FILE) as f:
        for line in f:
            if line.strip(): problems.append(json.loads(line))
    log(f"Loaded {len(problems)} benchmark problems")
    log(f"Models: {args.models}")
    log(f"Strategies: {args.strategies}")

    out_file = args.output or str(Path(RESULTS_DIR) / "multimodel_results.json")
    if Path(out_file).exists():
        with open(out_file) as _f: all_results = json.load(_f)
        log(f"Resuming: {len(all_results)} existing keys loaded")
    else:
        all_results = {}

    for model in args.models:
        backend = ALL_MODELS.get(model, "openai")

        # Check availability
        if backend == "openai" and not OPENAI_API_KEY:
            log(f"SKIP {model}: no OPENAI_API_KEY"); continue
        if backend == "gemini" and (not GEMINI_API_KEY or GEMINI_API_KEY == "SKIP"):
            log(f"SKIP {model}: gemini key invalid/missing"); continue
        if backend == "hf_local":
            pass  # model weights loaded on demand via transformers

        for strategy in args.strategies:
            key = f"{model}__{strategy}"
            if len(all_results.get(key, {})) >= 11:
                log(f"SKIP {key}: already complete"); continue
            log(f"\n{'='*60}")
            log(f"Model={model}  Strategy={strategy}")

            for prob in problems:
                pid = prob["id"]
                log(f"  {pid} [{prob['operator']}] {prob['title'][:45]}")

                # Shifted evaluation
                ev = evaluate_one(prob, problems, model, backend, strategy,
                                  use_shifted=True)
                if "error" not in ev:
                    p1 = ev["pass_at_1"]; p5 = ev["pass_at_5"]
                    tr = ev["trap_rate"]; ot = ev["opt_t"]
                    log(f"    shifted → p@1={p1} p@5={p5} trap={tr:.0%} opt={ot:.0%}")

                result_entry = {
                    "pid": pid, "title": prob["title"],
                    "operator": prob["operator"],
                    "model": model, "strategy": strategy,
                    "shifted": ev,
                }

                # Original evaluation (Direct strategy only to save API cost)
                if args.also_original and strategy == "direct":
                    ev_orig = evaluate_one(prob, problems, model, backend,
                                          strategy, use_shifted=False)
                    if "error" not in ev_orig:
                        log(f"    original → p@1={ev_orig['pass_at_1']} opt={ev_orig['opt_t']:.0%}")
                    result_entry["original"] = ev_orig

                all_results.setdefault(key, {})[pid] = result_entry

                # Save incrementally
                with open(out_file, "w") as f:
                    json.dump(all_results, f, indent=2)

    # Final summary
    log("\n" + "="*60)
    log("FINAL SUMMARY")
    log("="*60)
    log(f"{'Model/Strategy':<35} {'p@1':>6} {'p@5':>6} {'TrapRate':>9} {'OptT':>6} N")
    for key, probs in all_results.items():
        evs = [v["shifted"] for v in probs.values() if "error" not in v.get("shifted",{})]
        n   = len(evs)
        if not n: continue
        p1  = sum(float(e["pass_at_1"]) for e in evs)/n
        p5  = sum(float(e["pass_at_5"]) for e in evs)/n
        tr  = sum(e["trap_rate"]         for e in evs)/n
        ot  = sum(e["opt_t"]             for e in evs)/n
        log(f"{key:<35} {p1:>6.1%} {p5:>6.1%} {tr:>9.1%} {ot:>6.1%} {n}")

    log(f"\nAll results saved to {out_file}")

    # Optional local completion hook, for example: export ALGOBENCH_NOTIFY_CMD="say done"
    notify_cmd = os.environ.get("ALGOBENCH_NOTIFY_CMD")
    if notify_cmd:
        os.system(notify_cmd)

if __name__ == "__main__":
    main()
