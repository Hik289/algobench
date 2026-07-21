#!/usr/bin/env python3
"""
eval_checker.py v2 — Complexity Verifier Validation (Task B)
Handles both stdin-style (input/output) and func-style (args/expected) problems.
"""
import os, sys, json, ast, re, time, subprocess, tempfile, math, random
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path("/path/to/research_workspace/automatic_algorithm_design")
BENCH_FILE = BASE_DIR / "data" / "final_benchmark.jsonl"
OUT_JSON   = BASE_DIR / "analysis" / "table_checker.json"
LOG_FILE   = BASE_DIR / "logs" / "eval_checker.log"
CACHE_FILE = BASE_DIR / "logs" / "eval_checker_code_cache.json"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-4o-mini"
PYTHON = sys.executable
N_OPTIMAL = 80; N_SUBOPT = 80; N_BRUTEFORCE = 60; N_LLM_NOHINT = 60
RUNTIME_N_SMALL = 500; RUNTIME_N_LARGE = 3000; RUNTIME_TIMEOUT = 2.0
RUNTIME_RATIO_THRESHOLD = 15.0
MAX_EXAMPLE_CHARS = 800  # truncate examples in prompts
random.seed(42)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

def load_cache():
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def load_benchmark():
    probs = [json.loads(l) for l in BENCH_FILE.read_text().splitlines() if l.strip()]
    log(f"Loaded {len(probs)} benchmark problems"); return probs

def is_func_style(prob):
    ex = prob.get("shifted_examples", [])
    return bool(ex) and "args" in ex[0]

def examples_str(prob):
    """Return formatted examples for LLM prompt (max MAX_EXAMPLE_CHARS total)."""
    ex = prob.get("shifted_examples", [])
    if not ex: return "(no examples)"
    parts = []
    total = 0
    for i, e in enumerate(ex):
        if "input" in e and "output" in e:
            inp = str(e["input"])[:200]
            out = str(e["output"])[:200]
            s = f"Example {i+1}:\nInput: {inp}\nOutput: {out}"
        elif "args" in e and "expected" in e:
            inp = str(e["args"])[:200]
            out = str(e["expected"])[:200]
            s = f"Example {i+1}:\nArgs: {inp}\nExpected: {out}"
        else:
            s = f"Example {i+1}: {str(e)[:200]}"
        total += len(s)
        if total > MAX_EXAMPLE_CHARS:
            parts.append(f"(remaining examples truncated for brevity)")
            break
        parts.append(s)
    return "\n\n".join(parts)

def problem_io_format(prob):
    """Get I/O format description for LLM prompt."""
    if is_func_style(prob):
        ex = prob.get("shifted_examples", [{}])[0]
        return (f"Function signature: {prob.get('shifted_input','solve(...)→...')}\n"
                f"Return: {prob.get('shifted_output','see examples')}")
    return (f"Input format: {prob.get('shifted_input','')}\n"
            f"Output format: {prob.get('shifted_output','')}")

def call_gpt(system, user, temperature=0.0):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    r = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=1024, temperature=temperature)
    return r.choices[0].message.content

def extract_code(text):
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    if any(kw in text for kw in ('def ','import ','for ','print(')):
        lines = text.strip().splitlines()
        for i, l in enumerate(lines):
            if l.strip().startswith(('import ','def ','from ','class ','sys')):
                return '\n'.join(lines[i:]).strip()
        return text.strip()
    return text.strip()

def gen_code(ckey, cache, system, user, temperature=0.0):
    if ckey in cache: return cache[ckey]
    try:
        raw = call_gpt(system, user, temperature=temperature)
        cache[ckey] = extract_code(raw)
    except Exception as e:
        log(f"    API error {ckey}: {e}"); cache[ckey] = ""
    return cache[ckey]

def gen_optimal(prob, cache, idx=0):
    io_fmt = problem_io_format(prob)
    ex_str = examples_str(prob)
    style = "function" if is_func_style(prob) else "stdin/stdout"
    return gen_code(f"opt_{prob['id']}_{idx}", cache,
        f"Expert competitive programmer. Write ONLY complete Python code ({style} style).",
        f"Solve using {prob['target_algorithm']} with {prob['target_complexity']['time']} time.\n"
        f"Problem: {prob['shifted_statement'][:800]}\n"
        f"{io_fmt}\nConstraints: {prob.get('shifted_constraints','')[:200]}\n"
        f"Examples:\n{ex_str}\n"
        f"Output ONLY complete Python code.", temperature=0.0 if idx==0 else 0.5)

def gen_suboptimal(prob, cache, idx=0):
    src_alg = prob.get("source_reference_algorithm", "naive/brute-force approach (simpler, slower)")
    io_fmt = problem_io_format(prob)
    ex_str = examples_str(prob)
    style = "function" if is_func_style(prob) else "stdin/stdout"
    return gen_code(f"sub_{prob['id']}_{idx}", cache,
        f"Expert programmer. Write ONLY complete Python code ({style} style).",
        f"Solve using OLD/NAIVE approach: {src_alg}. "
        f"Do NOT use {prob['target_algorithm']}.\n"
        f"Problem: {prob['shifted_statement'][:800]}\n"
        f"{io_fmt}\nExamples:\n{ex_str}\n"
        f"Output ONLY Python code using old naive approach.", temperature=0.0 if idx==0 else 0.5)

def gen_bruteforce(prob, cache):
    io_fmt = problem_io_format(prob)
    ex_str = examples_str(prob)
    style = "function" if is_func_style(prob) else "stdin/stdout"
    return gen_code(f"bf_{prob['id']}", cache,
        f"Expert programmer. Write ONLY complete Python code ({style} style).",
        f"Solve with SLOWEST brute-force. REQUIRED: at least 2 nested for-loops.\n"
        f"Problem: {prob['shifted_statement'][:800]}\n"
        f"{io_fmt}\nExamples:\n{ex_str}\n"
        f"Output ONLY brute-force Python code with nested loops.")

def gen_nohint(prob, cache):
    io_fmt = problem_io_format(prob)
    ex_str = examples_str(prob)
    style = "function" if is_func_style(prob) else "stdin/stdout"
    return gen_code(f"nohint_{prob['id']}", cache,
        f"Expert competitive programmer. Write ONLY complete Python code ({style} style).",
        f"Problem: {prob['shifted_statement'][:800]}\n"
        f"{io_fmt}\nConstraints: {prob.get('shifted_constraints','')[:200]}\n"
        f"Examples:\n{ex_str}\nWrite a complete Python solution.", temperature=0.7)

def run_code_stdin(code, inp, timeout=5):
    if not code: return False, "", "empty"
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',delete=False) as f:
        f.write(code); fname=f.name
    try:
        r = subprocess.run([PYTHON,fname],input=inp,capture_output=True,text=True,timeout=timeout)
        return r.returncode==0, r.stdout.strip(), r.stderr[:200]
    except subprocess.TimeoutExpired: return False, "", "TLE"
    except Exception as e: return False, "", str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def run_code_func(code, args, expected, timeout=5):
    """Run function-style code. Wraps code in a test harness."""
    if not code: return False, None, "empty"
    # Build test harness
    harness = code + f"\n\nresult = solve(*{repr(args)})\nprint(result)\n"
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',delete=False) as f:
        f.write(harness); fname=f.name
    try:
        r = subprocess.run([PYTHON,fname],capture_output=True,text=True,timeout=timeout)
        if r.returncode != 0: return False, None, r.stderr[:200]
        # Parse output
        out = r.stdout.strip()
        try:
            import ast as ast2
            got = ast2.literal_eval(out)
            return got == expected, got, ""
        except:
            return str(got) == str(expected), out, ""
    except subprocess.TimeoutExpired: return False, None, "TLE"
    except Exception as e: return False, None, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def check_correctness(code, prob):
    ex = prob.get("shifted_examples", [])
    if not ex or not code: return False
    for e in ex:
        if "input" in e and "output" in e:
            ok, out, err = run_code_stdin(code, e["input"], timeout=5)
            if not ok: return False
            if out != e["output"].strip(): return False
        elif "args" in e and "expected" in e:
            ok, got, err = run_code_func(code, e["args"], e["expected"], timeout=5)
            if not ok: return False
    return True

def run_timed_stdin(code, inp, timeout=RUNTIME_TIMEOUT):
    if not code: return float('inf'), False, "empty"
    with tempfile.NamedTemporaryFile(mode='w',suffix='.py',delete=False) as f:
        f.write(code); fname=f.name
    try:
        t0=time.time()
        r=subprocess.run([PYTHON,fname],input=inp,capture_output=True,text=True,timeout=timeout)
        return time.time()-t0, r.returncode==0, r.stderr[:200]
    except subprocess.TimeoutExpired: return float('inf'), False, "TLE"
    except Exception as e: return float('inf'), False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def gen_large_input(prob, n):
    """Generate scaled input for runtime testing (stdin-style only)."""
    if is_func_style(prob): return None
    ex = prob.get("shifted_examples",[])
    if not ex: return None
    e0 = ex[0]
    if "input" not in e0: return None
    lines = e0["input"].strip().splitlines()
    rng = random.Random(42)
    try:
        first = lines[0].split()
        if len(first)>=2 and first[0].isdigit() and first[1].isdigit():
            q = min(int(first[1]), n)
            arr = " ".join(str(rng.randint(1,1000)) for _ in range(n))
            qlines = []
            if len(lines)>=3:
                sq = lines[2].split()
                for _ in range(q):
                    l=rng.randint(1,n); r=rng.randint(l,n)
                    if len(sq)==3 and sq[0].isdigit():
                        qlines.append(f"{rng.randint(1,2)} {l} {r}")
                    else: qlines.append(f"{l} {r}")
            return f"{n} {q}\n{arr}\n" + "\n".join(qlines) if qlines else f"{n} {q}\n{arr}"
        elif len(first)==1 and first[0].isdigit():
            arr = " ".join(str(rng.randint(1,1000)) for _ in range(n))
            return f"{n}\n{arr}"
        elif len(first)==2:
            W = int(first[1]) if first[1].isdigit() else n*10
            rows = "\n".join(f"{rng.randint(1,max(1,W//n))} {rng.randint(1,100)}" for _ in range(n))
            return f"{n} {W}\n{rows}"
    except: pass
    return None

def est_static_only(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "UNKNOWN"
    md=[0]
    def cd(node,d=0):
        if isinstance(node,(ast.For,ast.While)): d+=1; md[0]=max(md[0],d)
        for c in ast.iter_child_nodes(node): cd(c,d)
    cd(tree)
    return {"0":"O(1)","1":"O(n)","2":"O(n^2)","3":"O(n^3)"}.get(str(min(md[0],3)),"O(n^3)")

def est_static_tags(code):
    try: tree=ast.parse(code)
    except SyntaxError: return "UNKNOWN"
    md=[0]
    def cd(node,d=0):
        if isinstance(node,(ast.For,ast.While)): d+=1; md[0]=max(md[0],d)
        for c in ast.iter_child_nodes(node): cd(c,d)
    cd(tree)
    cl=code.lower()
    if any(p in cl for p in ['segment_tree','segtree','seg_tree','lazy','.update(','.query(']): return "O(n log n) [seg_tree]"
    if any(p in cl for p in ['fenwick','bit[','bit.update','lowbit','i & (-i)','i & -i']): return "O(n log n) [fenwick]"
    if any(p in cl for p in ['heapq','heappush','heappop','dijkstra','priority_queue']): return "O(n log n) [heap]"
    if any(p in cl for p in ['matrix_pow','mat_pow','matrix_power','matmul','matrix_exp','np.linalg']): return "O(log n) [matrix_exp]"
    if any(p in cl for p in ['union_find','dsu','parent[','find(']): return "O(n alpha) [dsu]"
    if any(p in cl for p in ['bisect','binary_search','mid = (','mid = l +']): return "O(n log n) [binary_search]"
    if 'sort(' in cl or '.sort(' in cl or 'sorted(' in cl: return "O(n log n) [sort]"
    return {"0":"O(1)","1":"O(n)","2":"O(n^2)","3":"O(n^3)"}.get(str(min(md[0],3)),"O(n^3)")

def est_runtime(code, prob):
    si=gen_large_input(prob,RUNTIME_N_SMALL); li=gen_large_input(prob,RUNTIME_N_LARGE)
    if si is None or li is None: return "UNCERTAIN"
    ts,ok_s,_=run_timed_stdin(code,si)
    if not ok_s or ts==float('inf'): return "O(n^2) [TLE on small]"
    tl,ok_l,_=run_timed_stdin(code,li)
    if not ok_l or tl==float('inf'): return "O(n^2) [TLE on large]"
    if ts<1e-4: return "UNCERTAIN"
    ratio=tl/max(ts,1e-6)
    return f"O(n^2) [ratio={ratio:.1f}]" if ratio>RUNTIME_RATIO_THRESHOLD else f"O(n log n or better) [ratio={ratio:.1f}]"

def is_optimal(est, target):
    if "UNCERTAIN" in est or "UNKNOWN" in est: return "UNCERTAIN"
    t=target.lower(); e=est.lower()
    if "n^2" in e or "n^3" in e or "tle" in e: return False
    if re.search(r'o\(\s*log\s*n\)',t) and "n log n" not in t:
        return "log" in e and "n log n" not in e
    if "log" in t: return True
    if re.search(r'o\(\s*(n\b|n\s*\+)',t): return "o(n)" in e or "o(1)" in e or "log" in e
    return True

def verdict(est, target):
    r=is_optimal(est,target)
    return "OPTIMAL" if r is True else ("SUBOPTIMAL" if r is False else "UNCERTAIN")

def verify_static_only(code, prob):
    e=est_static_only(code); return {"estimate":e,"verdict":verdict(e,prob["target_complexity"]["time"])}
def verify_runtime_only(code, prob):
    e=est_runtime(code,prob)
    if "UNCERTAIN" in e: return {"estimate":e,"verdict":"UNCERTAIN"}
    bad="n^2" in e.lower() or "tle" in e.lower()
    return {"estimate":e,"verdict":"SUBOPTIMAL" if bad else "OPTIMAL"}
def verify_static_tags(code, prob):
    e=est_static_tags(code); return {"estimate":e,"verdict":verdict(e,prob["target_complexity"]["time"])}
def verify_full(code, prob):
    vs=verify_static_tags(code,prob); vr=verify_runtime_only(code,prob)
    a,b=vs["verdict"],vr["verdict"]
    if a=="UNCERTAIN" and b=="UNCERTAIN": f="UNCERTAIN"
    elif a=="UNCERTAIN": f=b
    elif b=="UNCERTAIN": f=a
    elif a==b: f=a
    elif b=="SUBOPTIMAL": f="SUBOPTIMAL"
    else: f="UNCERTAIN"
    return {"estimate_static":vs["estimate"],"estimate_runtime":vr["estimate"],"verdict":f}

def compute_metrics(solutions, vfn, config_name, probs_by_id):
    n=len(solutions)
    gt_opt=sum(s["ground_truth"] for s in solutions)
    gt_sub=n-gt_opt
    uncertain=fo_n=fo_d=fs_n=fs_d=agreed=decided=0
    for sol in solutions:
        if not sol["code"]: v="UNCERTAIN"
        else: v=vfn(sol["code"],probs_by_id[sol["pid"]])["verdict"]
        gt=sol["ground_truth"]
        if v=="UNCERTAIN": uncertain+=1; continue
        decided+=1; pred=(v=="OPTIMAL")
        if gt==pred: agreed+=1
        if not gt: fo_d+=1; fo_n+=(1 if pred else 0)
        else: fs_d+=1; fs_n+=(1 if not pred else 0)
    return {
        "config":config_name,"n":n,"n_optimal_gt":gt_opt,"n_subopt_gt":gt_sub,
        "f_opt_pct":round(100*fo_n/max(fo_d,1),1),
        "f_sub_pct":round(100*fs_n/max(fs_d,1),1),
        "uncertain_pct":round(100*uncertain/max(n,1),1),
        "agreement_pct":round(100*agreed/max(decided,1),1),
        "raw_counts":{"f_opt_errors":fo_n,"f_sub_errors":fs_n,"uncertain":uncertain,
                      "agreed":agreed,"decided":decided},
    }

def main():
    LOG_FILE.parent.mkdir(parents=True,exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True)
    log("="*70)
    log("eval_checker.py v2 — Complexity Verifier Validation (Task B)")
    log("="*70)
    if not OPENAI_API_KEY: log("ERROR: no OPENAI_API_KEY"); sys.exit(1)
    probs=load_benchmark()
    probs_by_id={p["id"]:p for p in probs}
    cache=load_cache()
    log(f"Cache: {len(cache)} entries (pre-loaded)")
    solutions=[]
    sc = {c: 0 for c in ['opt','sub','bf','nh']}

    log("\nStep 1: Generating solutions...")

    log("  [1a] Optimal solutions (target_algorithm hint)...")
    for prob in probs:
        if sc['opt'] >= N_OPTIMAL: break
        code=gen_optimal(prob,cache,0)
        solutions.append({"pid":prob["id"],"category":"optimal","ground_truth":True,"code":code})
        sc['opt']+=1
        log(f"    [{sc['opt']}] {prob['id']} len={len(code)} func={is_func_style(prob)}")
        if sc['opt']%5==0: save_cache(cache)
    for prob in probs:
        if sc['opt'] >= N_OPTIMAL: break
        code=gen_optimal(prob,cache,1)
        solutions.append({"pid":prob["id"],"category":"optimal","ground_truth":True,"code":code})
        sc['opt']+=1
    save_cache(cache)
    log(f"  Total optimal: {sc['opt']}")

    log("  [1b] Suboptimal solutions (source_reference_algorithm hint)...")
    for prob in probs:
        if sc['sub'] >= N_SUBOPT: break
        code=gen_suboptimal(prob,cache,0)
        solutions.append({"pid":prob["id"],"category":"suboptimal","ground_truth":False,"code":code})
        sc['sub']+=1
        log(f"    [{sc['sub']}] {prob['id']}")
        if sc['sub']%5==0: save_cache(cache)
    for prob in probs:
        if sc['sub'] >= N_SUBOPT: break
        code=gen_suboptimal(prob,cache,1)
        solutions.append({"pid":prob["id"],"category":"suboptimal","ground_truth":False,"code":code})
        sc['sub']+=1
    save_cache(cache)
    log(f"  Total suboptimal: {sc['sub']}")

    log("  [1c] Brute-force solutions...")
    for prob in probs[:N_BRUTEFORCE]:
        code=gen_bruteforce(prob,cache)
        solutions.append({"pid":prob["id"],"category":"bruteforce","ground_truth":False,"code":code})
        sc['bf']+=1
        log(f"    [{sc['bf']}] {prob['id']}")
        if sc['bf']%5==0: save_cache(cache)
    save_cache(cache)
    log(f"  Total brute-force: {sc['bf']}")

    log("  [1d] LLM no-hint solutions...")
    for prob in probs[:N_LLM_NOHINT]:
        code=gen_nohint(prob,cache)
        if code:
            e=est_static_tags(code); gt_v=is_optimal(e,prob["target_complexity"]["time"]); gt=(gt_v is True)
        else: gt=False
        solutions.append({"pid":prob["id"],"category":"llm_nohint","ground_truth":gt,"code":code})
        sc['nh']+=1
        log(f"    [{sc['nh']}] {prob['id']} gt={gt}")
        if sc['nh']%5==0: save_cache(cache)
    save_cache(cache)
    log(f"  Total no-hint: {sc['nh']}")

    total=len(solutions)
    cats={c:sum(1 for s in solutions if s["category"]==c)
          for c in ["optimal","suboptimal","bruteforce","llm_nohint"]}
    log(f"\nTotal solutions: {total}, by category: {cats}")

    log("\nStep 2: Correctness spot-check...")
    cc=defaultdict(lambda:{"ok":0,"tot":0})
    for sol in solutions:
        cat=sol["category"]; cc[cat]["tot"]+=1
        if sol["code"]:
            try: ok=check_correctness(sol["code"], probs_by_id[sol["pid"]])
            except: ok=False
            cc[cat]["ok"]+=ok
    for cat,c in cc.items():
        log(f"  {cat}: {c['ok']}/{c['tot']} correct = {100*c['ok']/max(c['tot'],1):.1f}%")

    log("\nStep 3: Running 4 verifier configs...")
    VERIFIERS=[("static_only",verify_static_only),("runtime_only",verify_runtime_only),
               ("static_tags",verify_static_tags),("full",verify_full)]
    rows=[]
    for name,vfn in VERIFIERS:
        log(f"  Running {name}...")
        row=compute_metrics(solutions,vfn,name,probs_by_id)
        rows.append(row)
        log(f"    F-Opt={row['f_opt_pct']}% F-Sub={row['f_sub_pct']}% "
            f"Uncert={row['uncertain_pct']}% Agree={row['agreement_pct']}% n={row['n']}")

    output={
        "meta":{
            "n_solutions":total,"categories":cats,"model":GPT_MODEL,
            "date":time.strftime("%Y-%m-%d"),
            "methodology":(
                "optimal=generated with target_algorithm hint; "
                "suboptimal=generated with source_reference_algorithm hint; "
                "bruteforce=O(n^2) nested loops instruction; "
                "llm_nohint=no algorithm hint, ground truth by static_tags verifier. "
                "17/52 probs stdin-style, 35/52 func-style (args/expected format). "
                "runtime_only: timing at N=500,3000 for stdin-style only (func-style → UNCERTAIN). "
                "static_only: pure AST loop depth. static_tags: loop+keyword patterns. "
                "full: static_tags+runtime, disagree → UNCERTAIN."),
            "limitations":(
                "52 problems available. "
                "35/52 problems are func-style (args/expected); runtime_only returns UNCERTAIN for these. "
                "Ground truth for 'optimal' category relies on LLM following algorithm hints. "
                "Some optimal solutions may be incorrect despite the hint.")},
        "table":rows,
        "solutions_summary":[{"pid":s["pid"],"category":s["category"],
                               "ground_truth":s["ground_truth"],"has_code":bool(s["code"])}
                              for s in solutions]}
    OUT_JSON.write_text(json.dumps(output,indent=2))
    log(f"\nOutput written to {OUT_JSON}")

    log("\n=== Checker Validation Table ===")
    log(f"{'Config':<15}{'N':>5}{'F-Opt':>8}{'F-Sub':>8}{'Uncert':>8}{'Agree':>8}")
    log("-"*52)
    for row in rows:
        log(f"{row['config']:<15}{row['n']:>5}"
            f"{str(row['f_opt_pct'])+'%':>8}{str(row['f_sub_pct'])+'%':>8}"
            f"{str(row['uncertain_pct'])+'%':>8}{str(row['agreement_pct'])+'%':>8}")
    log("\nTask B complete.")

if __name__=="__main__": main()
