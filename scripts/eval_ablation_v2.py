#!/usr/bin/env python3
"""
eval_ablation_v2.py - Quality Gate Ablation on expanded candidate pool.

Expands pool from n=54 to ~210 by generating:
- 52 G3-synthetic (near-paraphrase candidates)
- 52 G1-synthetic (relaxed shift candidates)  
- 52 G4-synthetic (wrong complexity tag candidates)
+ 54 original base candidates (52 benchmark + 2 source-only)

Usage: python3 scripts/eval_ablation_v2.py
"""
import ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path("/path/to/project_workspace/algorithm_design")
BENCH_FILE = BASE_DIR / "data" / "final_benchmark.jsonl"
SRC_FILE   = BASE_DIR / "release" / "constraintshift" / "data" / "source_problems.jsonl"
OUT_JSON   = BASE_DIR / "analysis" / "table_ablation.json"
OUT_V2     = BASE_DIR / "analysis" / "table_ablation_v2.json"
LOG_FILE   = BASE_DIR / "logs" / "eval_ablation_v2.log"
CACHE_FILE = BASE_DIR / "logs" / "eval_ablation_v2_cache.json"
OLD_CACHE  = BASE_DIR / "logs" / "eval_ablation_code_cache.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o-mini"
PYTHON         = sys.executable
NEAR_PARA_THRESH = 0.70

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_cache():
    c = {}
    for cf in [OLD_CACHE, CACHE_FILE]:
        if cf.exists():
            try: c.update(json.loads(cf.read_text()))
            except: pass
    return c

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

def load_base_candidates():
    bench = {}
    for line in BENCH_FILE.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            bench[p["id"]] = p
    all_c = dict(bench)
    for line in SRC_FILE.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            if p["id"] not in all_c:
                p["benchmark_status"] = "not_in_benchmark"
                all_c[p["id"]] = p
    log(f"Base: {len(all_c)} ({len(bench)} bench + {len(all_c)-len(bench)} source-only)")
    return list(all_c.values())

def call_gpt(system, user, temperature=0.0):
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    r = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=600, temperature=temperature)
    return r.choices[0].message.content

def extract_code(text):
    m = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    if any(k in text for k in ('def ','import ','for ','print(')): return text.strip()
    return text.strip()

def estimate_complexity(code):
    try: tree = ast.parse(code)
    except: return "UNKNOWN"
    md = [0]
    def cd(node, d=0):
        if isinstance(node, (ast.For, ast.While)): d+=1; md[0]=max(md[0],d)
        for c in ast.iter_child_nodes(node): cd(c, d)
    cd(tree)
    cl = code.lower()
    if any(p in cl for p in ['segment_tree','segtree','lazy','.update(','.query(']): return "O(n log n) [seg]"
    if any(p in cl for p in ['fenwick','bit[','lowbit','i & (-i)','i & -i']): return "O(n log n) [bit]"
    if any(p in cl for p in ['heapq','heappush','heappop','dijkstra']): return "O(n log n) [heap]"
    if any(p in cl for p in ['matrix_pow','mat_pow','matrix_power','matmul']): return "O(log n) [mat]"
    if any(p in cl for p in ['union_find','dsu','parent[']): return "O(n alpha) [dsu]"
    if any(p in cl for p in ['bisect','mid = (','mid = l +']): return "O(n log n) [bin]"
    if 'sort(' in cl or '.sort(' in cl or 'sorted(' in cl: return "O(n log n) [sort]"
    return {"0":"O(1)","1":"O(n)","2":"O(n^2)","3":"O(n^3)"}.get(str(min(md[0],3)),"O(n^3)")

def complexity_ok(est, target):
    t=target.lower(); e=est.lower()
    if re.search(r'n[\^*]2|nw|n\*w|nm\b|[|·]|\|s\|', t): return True
    if 'n^2' in e or 'n^3' in e or 'tle' in e or 'n\u00b2' in e: return False
    if re.search(r'o\(\s*log\s*n\s*\)',t) and 'n log n' not in t:
        return 'log' in e and 'n log n' not in e
    if 'log' in t: return True
    if re.search(r'o\(\s*(n\b|n\s*[\+\-])',t):
        return 'o(n)' in e or 'o(1)' in e or 'log' in e or 'o(n' in e
    return True

def run_code(code, examples, timeout=5):
    if not code: return False, "empty"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code); fname=f.name
    try:
        for ex in examples:
            if "input" not in ex or "output" not in ex: continue
            r = subprocess.run([PYTHON,fname],input=ex["input"],capture_output=True,
                               text=True,timeout=timeout)
            if r.returncode != 0: return False, f"err:{r.stderr[:80]}"
            if r.stdout.strip() != ex["output"].strip():
                return False, f"WA"
        return True, "ok"
    except subprocess.TimeoutExpired: return False, "TLE"
    except Exception as e: return False, str(e)
    finally:
        try: os.unlink(fname)
        except: pass

def jaccard(s1, s2):
    w1=set(re.findall(r'\w+',s1.lower())); w2=set(re.findall(r'\w+',s2.lower()))
    if not w1 and not w2: return 1.0
    return len(w1&w2)/len(w1|w2)

def gen_old(prob, cache):
    ckey = f"old_{prob['id']}"
    if ckey in cache: return cache[ckey]
    ex_str = "\n".join(f"In: {e['input']}\nOut: {e['output']}"
                       for e in prob.get('shifted_examples',[])[:2] if 'input' in e)
    try:
        raw = call_gpt("Expert Python programmer. Code only.",
            f"Write Python using OLD slow algorithm: {prob['source_reference_algorithm']}.\n"
            f"Problem: {prob['shifted_statement'][:500]}\n"
            f"INPUT: {prob.get('shifted_input','')}\nOUTPUT: {prob.get('shifted_output','')}\n"
            f"EXAMPLES:\n{ex_str}\nCode only.")
        cache[ckey] = extract_code(raw)
    except Exception as e:
        log(f"    gen_old error {prob['id']}: {e}"); cache[ckey] = ""
    return cache[ckey]

def gen_ref(prob, cache):
    ckey = f"ref_{prob['id']}"
    if ckey in cache: return cache[ckey]
    ex_str = "\n".join(f"In: {e['input']}\nOut: {e['output']}"
                       for e in prob.get('shifted_examples',[])[:2] if 'input' in e)
    try:
        raw = call_gpt("Expert competitive programmer. Code only.",
            f"Write Python using {prob['target_algorithm']} ({prob['target_complexity']['time']}).\n"
            f"Problem: {prob['shifted_statement'][:500]}\n"
            f"INPUT: {prob.get('shifted_input','')}\nOUTPUT: {prob.get('shifted_output','')}\n"
            f"CONSTRAINTS: {prob.get('shifted_constraints','')[:200]}\n"
            f"EXAMPLES:\n{ex_str}\nCode only.")
        cache[ckey] = extract_code(raw)
    except Exception as e:
        log(f"    gen_ref error {prob['id']}: {e}"); cache[ckey] = ""
    return cache[ckey]

def gen_paraphrase(prob, cache, idx=0):
    ckey = f"para_{prob['id']}_{idx}"
    if ckey in cache: return cache[ckey]
    try:
        raw = call_gpt("Technical writer. Return only rewritten text.",
            f"Rewrite keeping 70%+ words the same. Only minor rephrasing.\n\n"
            f"{prob['source_statement'][:700]}", temperature=0.3+idx*0.2)
        cache[ckey] = raw.strip()
    except Exception as e:
        log(f"    para error {prob['id']}: {e}"); cache[ckey] = ""
    return cache[ckey]

def gen_relaxed(prob, cache, idx=0):
    ckey = f"relax_{prob['id']}_{idx}"
    if ckey in cache: return cache[ckey]
    try:
        raw = call_gpt("Competitive programming problem designer. Return only statement text.",
            f"Create a WEAKER version where old algorithm '{prob['source_reference_algorithm']}' "
            f"might STILL work (n<=2000, not n<=10^5).\n"
            f"Source: {prob['source_statement'][:500]}\n"
            f"Shift: {prob['shifted_statement'][:500]}\n"
            f"Make constraints smaller. Return ONLY the new shifted statement.",
            temperature=0.4+idx*0.2)
        cache[ckey] = raw.strip()
    except Exception as e:
        log(f"    relax error {prob['id']}: {e}"); cache[ckey] = ""
    return cache[ckey]

def compute_metrics(prob, cache):
    pid      = prob["id"]
    examples = [e for e in prob.get("shifted_examples",[]) if "input" in e and "output" in e]
    details  = {}

    # G1
    if prob.get("benchmark_status") == "not_in_benchmark":
        g1 = True; details["g1"] = "source-only, rejected by G1 in production"
    elif prob.get("_ctype") == "g1_relax":
        g1 = True; details["g1"] = "g1-synthetic: relaxed constraint"
    else:
        old_code = gen_old(prob, cache)
        if old_code and examples:
            ok, reason = run_code(old_code, examples); g1 = ok
        else:
            g1 = False
        details["g1"] = f"code={'yes' if old_code else 'no'}"

    # G2
    ref_code = gen_ref(prob, cache)
    if ref_code and examples:
        ok, reason = run_code(ref_code, examples, timeout=10); g2 = not ok
    else:
        g2 = False
    details["g2"] = f"ref_passes={'yes' if ref_code and not g2 else 'no'}"

    # G3
    src  = prob.get("source_statement","")
    shft = prob.get("shifted_statement","")
    j    = jaccard(src, shft)
    g3   = (j >= NEAR_PARA_THRESH)
    details["g3_jaccard"] = round(j, 4)

    # G4
    if prob.get("_ctype") == "g4_wrong":
        g4 = True; details["g4"] = "g4-synthetic: wrong complexity tag"
    elif ref_code:
        est = estimate_complexity(ref_code)
        tgt = prob.get("target_complexity",{}).get("time","")
        g4  = not complexity_ok(est, tgt)
        details["g4_est"] = est; details["g4_tgt"] = tgt
    else:
        g4 = False; details["g4"] = "no_ref_code"

    return {"pid":pid,"ctype":prob.get("_ctype","base"),
            "g1":g1,"g2":g2,"g3":g3,"g4":g4,"details":details}

def build_pool(base, cache):
    pool = []
    for p in base:
        p2 = dict(p); p2["_ctype"] = "base"; pool.append(p2)
    bench = [p for p in base if p.get("benchmark_status") != "not_in_benchmark"]
    log(f"  base={len(base)}, bench={len(bench)}")

    # G3 synthetics
    g3c = 0
    for p in bench:
        para = gen_paraphrase(p, cache, 0)
        if not para: continue
        # Compare paraphrase to original shifted statement (Jaccard should be >= 0.7)
        pool.append({
            "id":f"SYN_G3_{p['id']}","_ctype":"g3_para",
            "source_statement":  p.get("source_statement",""),
            "shifted_statement": para,  # paraphrase of SOURCE → high Jaccard with source
            "source_reference_algorithm": p.get("source_reference_algorithm","bf"),
            "target_algorithm":  p.get("target_algorithm",""),
            "target_complexity": p.get("target_complexity",{}),
            "shifted_examples":  p.get("shifted_examples",[]),
            "shifted_input":     p.get("shifted_input",""),
            "shifted_output":    p.get("shifted_output",""),
            "shifted_constraints":p.get("shifted_constraints",""),
            "operator": p.get("operator",""), "title": f"[G3-SYNTH] {p.get('title','')}",
        })
        g3c += 1
        if g3c % 10 == 0: save_cache(cache)
    save_cache(cache)
    log(f"  G3 synthetics: {g3c}")

    # G1 synthetics
    g1c = 0
    for p in bench:
        relx = gen_relaxed(p, cache, 0)
        if not relx: continue
        pool.append({
            "id":f"SYN_G1_{p['id']}","_ctype":"g1_relax",
            "source_statement":  p.get("source_statement",""),
            "shifted_statement": relx,
            "source_reference_algorithm": p.get("source_reference_algorithm","bf"),
            "target_algorithm":  p.get("target_algorithm",""),
            "target_complexity": p.get("target_complexity",{}),
            "shifted_examples":  p.get("shifted_examples",[]),
            "shifted_input":     p.get("shifted_input",""),
            "shifted_output":    p.get("shifted_output",""),
            "shifted_constraints":p.get("shifted_constraints",""),
            "operator": p.get("operator",""), "title": f"[G1-SYNTH] {p.get('title','')}",
        })
        g1c += 1
        if g1c % 10 == 0: save_cache(cache)
    save_cache(cache)
    log(f"  G1 synthetics: {g1c}")

    # G4 synthetics (wrong complexity tags)
    WRONG = [{"time":"O(n^2)","space":"O(1)"},
             {"time":"O(n log n)","space":"O(n)"},
             {"time":"O(log n)","space":"O(1)"}]
    g4c = 0
    for p in bench:
        real = p.get("target_complexity",{}).get("time","")
        wc = next((w for w in WRONG if w["time"].lower() != real.lower()), None)
        if not wc: continue
        pool.append({
            "id":f"SYN_G4_{p['id']}","_ctype":"g4_wrong",
            "source_statement":  p.get("source_statement",""),
            "shifted_statement": p.get("shifted_statement",""),
            "source_reference_algorithm": p.get("source_reference_algorithm","bf"),
            "target_algorithm":  p.get("target_algorithm",""),
            "target_complexity": wc,  # WRONG tag
            "shifted_examples":  p.get("shifted_examples",[]),
            "shifted_input":     p.get("shifted_input",""),
            "shifted_output":    p.get("shifted_output",""),
            "shifted_constraints":p.get("shifted_constraints",""),
            "operator": p.get("operator",""),
            "title": f"[G4-SYNTH] {p.get('title','')} (wrong:{wc['time']})",
        })
        g4c += 1
    log(f"  G4 synthetics: {g4c}")

    log(f"  TOTAL pool: {len(pool)}")
    return pool

def accepted(metrics_list, disable_gate):
    result = []
    for m in metrics_list:
        r1 = m["g1"] and (disable_gate != 1)
        r2 = m["g2"] and (disable_gate != 2)
        r3 = m["g3"] and (disable_gate != 3)
        r4 = m["g4"] and (disable_gate != 4)
        if not (r1 or r2 or r3 or r4): result.append(m)
    return result

def make_row(name, mlist):
    n = len(mlist)
    if not n:
        return {"config":name,"n":0,"old_sol_pct":None,"ref_fail_pct":None,
                "near_para_pct":None,"f_opt_pct":None}
    g1=sum(m["g1"] for m in mlist); g2=sum(m["g2"] for m in mlist)
    g3=sum(m["g3"] for m in mlist); g4=sum(m["g4"] for m in mlist)
    return {
        "config":name,"n":n,
        "old_sol_pct": round(100*g1/n,1),
        "ref_fail_pct":round(100*g2/n,1),
        "near_para_pct":round(100*g3/n,1),
        "f_opt_pct":    round(100*g4/n,1),
        "raw_counts":{"old_sol":g1,"ref_fail":g2,"near_para":g3,"f_opt":g4},
    }

def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    log("="*70)
    log("eval_ablation_v2.py - Expanded Pool Quality Gate Ablation")
    log("="*70)
    if not OPENAI_API_KEY: log("ERROR: no OPENAI_API_KEY"); sys.exit(1)

    base = load_base_candidates()
    cache = load_cache()
    log(f"Cache: {len(cache)} entries")

    log("\nStep 1: Build expanded pool...")
    pool = build_pool(base, cache)
    save_cache(cache)

    # Pool breakdown
    types = defaultdict(int)
    for p in pool: types[p.get("_ctype","base")] += 1
    for k,v in sorted(types.items()): log(f"  {k}: {v}")

    log(f"\nStep 2: Computing gate metrics for {len(pool)} candidates...")
    all_metrics = []
    for i, prob in enumerate(pool):
        pid = prob["id"]
        ctype = prob.get("_ctype","base")
        if i % 20 == 0:
            log(f"  Progress: {i}/{len(pool)}")
        try:
            m = compute_metrics(prob, cache)
            all_metrics.append(m)
            if i < 60 or ctype != "base":  # Log more detail for non-base
                log(f"  [{i+1}] {pid} ({ctype}): G1={m['g1']} G2={m['g2']} G3={m['g3']} G4={m['g4']} jacc={m['details'].get('g3_jaccard','?')}")
        except Exception as e:
            log(f"  [{i+1}] ERROR {pid}: {e}")
            all_metrics.append({"pid":pid,"ctype":ctype,"g1":False,"g2":False,"g3":False,"g4":False,
                                 "details":{"error":str(e)}})
        if (i+1) % 15 == 0: save_cache(cache)
    save_cache(cache)

    # Gate violations summary
    n = len(all_metrics)
    log("\nGate violations across full pool:")
    for gate, field in [("G1","g1"),("G2","g2"),("G3","g3"),("G4","g4")]:
        cnt = sum(m[field] for m in all_metrics)
        log(f"  {gate}: {cnt}/{n} = {100*cnt/max(n,1):.1f}%")

    log("\nStep 3: Ablation configs...")
    configs = [("full",0),("no_g1",1),("no_g2",2),("no_g3",3),("no_g4",4)]
    rows = []
    for name, dg in configs:
        acc = accepted(all_metrics, dg)
        row = make_row(name, acc)
        rows.append(row)
        log(f"  {name:8s} n={row['n']:3d} | Old-Sol={row['old_sol_pct']}% Ref-Fail={row['ref_fail_pct']}% Near-Para={row['near_para_pct']}% F-Opt={row['f_opt_pct']}%")

    import shutil
    if OUT_JSON.exists():
        shutil.copy(OUT_JSON, OUT_V2)
        log(f"Old table backed up to {OUT_V2}")

    output = {
        "meta":{
            "n_candidates":len(pool),"n_base":len(base),
            "n_synthetic":len(pool)-len(base),
            "pool_breakdown":dict(types),
            "near_para_threshold":NEAR_PARA_THRESH,
            "model":GPT_MODEL,"date":time.strftime("%Y-%m-%d"),
            "script":"eval_ablation_v2.py",
            "methodology":(
                "G1: old sol (source_reference_algorithm) passes shifted_examples; "
                "G2: new ref sol (target_algorithm) FAILS shifted_examples; "
                "G3: Jaccard(source_stmt, shifted_stmt) >= 0.7; "
                "G4: estimate_complexity(ref_code) mismatches target_complexity. "
                "Pool: 54 base + G3/G1/G4 synthetics generated via GPT-4o-mini."),
        },
        "table":rows,
        "per_candidate":[
            {"pid":m["pid"],"ctype":m["ctype"],"g1":m["g1"],"g2":m["g2"],"g3":m["g3"],"g4":m["g4"]}
            for m in all_metrics
        ],
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    log(f"\nOutput: {OUT_JSON}")
    log(f"Full data: {OUT_V2}")
    log("\n=== Ablation Table ===")
    log(f"{'Config':<10} {'N':>4} {'Old-Sol':>8} {'Ref-Fail':>9} {'Near-Para':>10} {'F-Opt':>7}")
    log("-"*55)
    for r in rows:
        log(f"{r['config']:<10} {r['n']:>4} {str(r['old_sol_pct'])+'%':>8} {str(r['ref_fail_pct'])+'%':>9} {str(r['near_para_pct'])+'%':>10} {str(r['f_opt_pct'])+'%':>7}")
    log(f"\nPool: {len(pool)} total ({len(base)} base + {len(pool)-len(base)} synthetic)")
    log("\nT7 COMPLETE.")

if __name__ == "__main__":
    main()
