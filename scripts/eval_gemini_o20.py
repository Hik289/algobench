#!/usr/bin/env python3
"""
补跑 Gemini-2.5-flash direct prompting on 20 题 (ORIG11 ∪ HARD9)。
Schema 与 gemini-2.5-flash__direct['CS006'] 完全一致 (含 sample_results)。
N=5 samples × 2 mode (shifted + original) = 200 API calls。
"""

import os, json, re, ast, subprocess, time
from pathlib import Path
from google import genai as gai
from google.genai import types as gtypes

ROOT  = Path('/path/to/.openclaw/research_topics/automatic_algorithm_design')
RFILE = ROOT / 'results/multimodel_results.json'
BENCH = ROOT / 'data/final_benchmark.jsonl'

ORIG11 = ['CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002','GT001','GT002','GT003']
HARD9  = ['CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3']
O20    = ORIG11 + HARD9

N = 5
MODEL = 'gemini-2.5-flash'
SYS = ("You are an expert competitive programmer. "
       "Write a clean, correct Python function `solve(*args)` that returns the answer. "
       "Output ONLY a Python code block.")

GEMINI_KEY = os.environ.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
gc = gai.Client(api_key=GEMINI_KEY)

probs = {p['id']: p for p in (json.loads(l) for l in open(BENCH))}
mr    = json.load(open(RFILE))
KEY   = 'gemini-2.5-flash__direct'
if KEY not in mr: mr[KEY] = {}

# ── Gemini call ─────────────────────────────────────────────────────────
def call_gemini(prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            r = gc.models.generate_content(
                model=MODEL, contents=prompt,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=3000, temperature=0.8,
                    thinking_config=gtypes.ThinkingConfig(thinking_budget=0)))
            return r.text or ''
        except Exception as e:
            print(f"    Gemini error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return ''

# ── Prompt ──────────────────────────────────────────────────────────────
def make_prompt(p, use_source=False):
    if use_source:
        stmt = p.get('source_statement', '')
        cons = p.get('source_constraints', '')
        cplx = p.get('source_complexity', {})
        exs  = p.get('source_examples', [])[:2]
        algo = p.get('source_reference_algorithm', '')
    else:
        stmt = p.get('shifted_statement', '')
        cons = p.get('shifted_constraints', '')
        cplx = p.get('target_complexity', {})
        exs  = p.get('shifted_examples', [])[:2]
        algo = p.get('target_algorithm', '')
    tcs = cplx.get('time', '') if isinstance(cplx, dict) else str(cplx)
    ex_txt = ''
    for ex in exs:
        args = ex.get('args', []); exp = ex.get('expected', ex.get('output', ''))
        if len(json.dumps({'args': args})) > 300: continue
        ex_txt += f"  solve({', '.join(repr(a) for a in args)}) → {exp!r}\n"
    return (f"{SYS}\n\n"
            f"Problem: {stmt}\n"
            f"Constraints: {cons}\n"
            f"Target complexity: {tcs}  Algorithm hint: {algo}\n"
            f"Examples:\n{ex_txt}\n"
            f"Write the Python function `solve(*args):`")

# ── Code extraction ────────────────────────────────────────────────────
def extract_code(txt):
    if not txt: return ''
    m = re.search(r'```python\s*(.*?)```', txt, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'```\s*(.*?)```', txt, re.DOTALL)
    if m: return m.group(1).strip()
    lines = txt.split('\n')
    start = next((i for i,l in enumerate(lines) if l.strip().startswith('def solve')), None)
    if start is not None: return '\n'.join(lines[start:])
    return txt.strip()

# ── Correctness ────────────────────────────────────────────────────────
def run_code(code, p, use_source=False):
    exs = (p.get('source_examples', []) if use_source else p.get('shifted_examples', []))[:3]
    if not exs: return False, 'no_examples'
    for ex in exs:
        args = ex.get('args', []); exp = ex.get('expected', ex.get('output', ''))
        if len(json.dumps({'args': args})) > 2000: continue
        prog = (f"import sys, math, collections, heapq, itertools, functools, bisect\n"
                f"{code}\n"
                f"_r = solve(*{args!r})\n"
                f"if isinstance(_r, (list, tuple)): print(str(list(_r)).replace(' ',''))\n"
                f"else: print(_r)")
        try:
            r = subprocess.run(['python3', '-c', prog],
                               capture_output=True, text=True, timeout=10)
            got = r.stdout.strip()
            exp_s = str(list(exp)).replace(' ', '') if isinstance(exp, list) else str(exp).replace(' ', '')
            got_s = got.replace(' ', '')
            if got_s == exp_s: continue
            try:
                if abs(float(got_s) - float(exp_s)) < 1e-6: continue
            except: pass
            return False, f'WA: got {got!r} expected {exp_s!r}'
        except subprocess.TimeoutExpired:
            return False, 'TLE'
        except Exception as e:
            return False, f'error: {e}'
    return True, 'ok'

# ── Complexity / Space estimation (复用 eval_full_metrics.py) ─────────
def estimate_complexity(code):
    try: tree = ast.parse(code)
    except SyntaxError: return 'UNKNOWN'
    max_depth = 0
    def count_depth(node, depth=0):
        nonlocal max_depth
        if isinstance(node, (ast.For, ast.While)):
            depth += 1; max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node): count_depth(child, depth)
    count_depth(tree)
    cl = code.lower()
    if any(p in cl for p in ['mat_pow','matrix_power','mat_mult','matmul']): return 'O(log n) [matrix exp]'
    if any(p in cl for p in ['fenwick','bit[','lowbit','i & (-i)','i & -i']):  return 'O(n log n) [BIT]'
    if any(p in cl for p in ['segment_tree','segtree','seg_tree']):             return 'O(n log n) [seg tree]'
    if any(p in cl for p in ['heapq','dijkstra']):                              return 'O(n log n) [heap]'
    if any(p in cl for p in ['bisect','binary_search']):                        return 'O(n log n) [bisect]'
    if 'sort(' in cl or '.sort(' in cl:                                         return 'O(n log n) [sort]'
    if any(p in cl for p in ['tarjan','scc','kosaraju']):                       return 'O(V+E) [SCC]'
    if any(p in cl for p in ['kmp','lps','fail','failure_function']):           return 'O(n+m) [KMP]'
    if any(p in cl for p in ['manacher','palindrome']):                         return 'O(n) [Manacher]'
    if any(p in cl for p in ['union','find','dsu','disjoint']):                 return 'O(α(n)) [DSU]'
    if max_depth >= 3: return 'O(n³) or worse'
    if max_depth == 2: return 'O(n²)'
    if max_depth == 1: return 'O(n)'
    return 'O(1) or O(log n)'

def complexity_is_optimal(estimated, target_str):
    t = target_str.lower(); e = estimated.lower()
    if 'log n' in t and 'n log n' not in t:
        return 'log' in e and 'n log' not in e
    if 'n log n' in t or 'nlogn' in t:
        return ('log n' in e or 'log' in e) and 'n²' not in e and 'n³' not in e
    if 'v+e' in t or 'v + e' in t:
        return 'v+e' in e or 'scc' in e or 'kmp' in e or 'o(n)' in e
    if 'o(n)' in t or 'linear' in t:
        return 'o(n)' in e or 'o(1)' in e or 'manacher' in e or 'kmp' in e
    if 'o(1)' in t:
        return 'o(1)' in e or 'log n' in e
    return 'n²' not in e and 'n³' not in e and 'worse' not in e

def estimate_space(code):
    cl = code.lower()
    if any(p in cl for p in ['mat_pow','matrix_power']): return 'O(k²)'
    if any(p in cl for p in ['segment_tree','segtree']): return 'O(n)'
    if any(p in cl for p in ['fenwick','bit[']):          return 'O(n)'
    if any(p in cl for p in ['heapq','dijkstra']):        return 'O(n)'
    if any(p in cl for p in ['bisect']):                  return 'O(1) extra'
    if 'sort(' in cl or '.sort(' in cl:                   return 'O(n)'
    if any(p in cl for p in ['tarjan','scc','kosaraju']): return 'O(V+E)'
    if any(p in cl for p in ['kmp','lps','fail']):        return 'O(n)'
    if any(p in cl for p in ['manacher']):                return 'O(n)'
    if 'dp[' in cl or "dp = [" in cl:                    return 'O(n)'
    return 'O(n)'

def space_is_optimal(estimated_space, target_space_str):
    t = target_space_str.lower(); e = estimated_space.lower()
    if 'o(1)' in t:         return 'o(1)' in e or 'extra' in e
    if 'o(log n)' in t:     return 'log' in e
    if 'o(k' in t:          return 'k' in e
    if 'o(n log n)' in t:   return 'log' in e
    return True

def has_trap(code, patterns):
    if not patterns: return False
    cl = code.lower()
    return any(p.lower() in cl for p in patterns)

# ── Single (problem, mode) evaluation ──────────────────────────────────
def eval_one(p, use_source=False):
    pid = p['id']
    prompt = make_prompt(p, use_source)
    tc  = p.get('target_complexity', {}) if not use_source else p.get('source_complexity', {})
    tcs = tc.get('time', 'O(n log n)') if isinstance(tc, dict) else str(tc)
    tss = tc.get('space', 'O(n)') if isinstance(tc, dict) else 'O(n)'
    # trap_patterns 仅对 shifted 适用 (source 没有 trap)
    traps = p.get('trap_patterns', []) if not use_source else []

    sample_results = []
    for i in range(N):
        txt  = call_gemini(prompt)
        code = extract_code(txt) if txt else ''
        if code:
            ok, reason = run_code(code, p, use_source)
        else:
            ok, reason = False, 'no_code'
        est_t = estimate_complexity(code) if code else 'UNKNOWN'
        est_s = estimate_space(code) if code else 'UNKNOWN'
        is_opt_t = complexity_is_optimal(est_t, tcs)
        is_opt_s = space_is_optimal(est_s, tss) if ok else False
        trap_hit = has_trap(code, traps)
        sample_results.append({
            'correct': ok, 'reason': reason,
            'est_t': est_t, 'est_s': est_s,
            'is_opt_t': is_opt_t, 'is_opt_s': is_opt_s,
            'trap': trap_hit,
        })
        time.sleep(0.3)

    n_correct = sum(r['correct'] for r in sample_results)
    n_wrong   = N - n_correct
    n_opt_t   = sum(r['is_opt_t'] for r in sample_results if r['correct'])
    n_opt_s   = sum(r['is_opt_s'] for r in sample_results if r['correct'])
    n_trap_w  = sum(r['trap']     for r in sample_results if not r['correct'])

    return {
        'pass_at_1': sample_results[0]['correct'],
        'pass_at_5': n_correct > 0,
        'opt_t': n_opt_t / n_correct if n_correct else 0.0,
        'opt_s': n_opt_s / n_correct if n_correct else 0.0,
        'trap_rate': n_trap_w / n_wrong if n_wrong else 0.0,
        'n_pass': n_correct,
        'n_total': N,
        'sample_results': sample_results,
    }

def save():
    with open(RFILE, 'w') as f:
        json.dump(mr, f, indent=2)

# ── Main ───────────────────────────────────────────────────────────────
def main():
    print(f"=== Gemini-2.5-flash direct on O20 ({len(O20)} problems, N={N}) ===")
    print(f"开始时刻 JST: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    total_calls = 0
    for pid in O20:
        if pid not in probs:
            print(f"  SKIP {pid}: not in benchmark"); continue
        p = probs[pid]

        # Resumable: 若已有完整 schema 则跳过
        existing = mr[KEY].get(pid, {})
        has_shf = isinstance(existing.get('shifted'), dict) and 'sample_results' in existing.get('shifted', {})
        has_src = isinstance(existing.get('original'), dict) and 'sample_results' in existing.get('original', {})

        if pid not in mr[KEY]: mr[KEY][pid] = {}

        if not has_shf:
            print(f"  shf {pid}...", end=' ', flush=True)
            r = eval_one(p, use_source=False)
            mr[KEY][pid]['shifted'] = r; save(); total_calls += N
            print(f"p@1={'✓' if r['pass_at_1'] else '✗'} p@5={'✓' if r['pass_at_5'] else '✗'} "
                  f"OptT={r['opt_t']:.0%} OptS={r['opt_s']:.0%} Trap={r['trap_rate']:.0%} "
                  f"({r['n_pass']}/{r['n_total']})")
        else:
            print(f"  SKIP shf {pid} (already done)")

        if not has_src:
            print(f"  src {pid}...", end=' ', flush=True)
            r = eval_one(p, use_source=True)
            mr[KEY][pid]['original'] = r; save(); total_calls += N
            print(f"p@1={'✓' if r['pass_at_1'] else '✗'} p@5={'✓' if r['pass_at_5'] else '✗'} "
                  f"OptT={r['opt_t']:.0%} OptS={r['opt_s']:.0%} "
                  f"({r['n_pass']}/{r['n_total']})")
        else:
            print(f"  SKIP src {pid} (already done)")
        time.sleep(0.5)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n=== 完成 ({total_calls} API calls) ===")
    import statistics as stats
    d = mr[KEY]
    o20_in = [p for p in O20 if p in d]
    src_p1 = [int(d[p]['original']['pass_at_1']) for p in o20_in if 'pass_at_1' in d[p].get('original', {})]
    shf_p1 = [int(d[p]['shifted']['pass_at_1']) for p in o20_in if 'pass_at_1' in d[p].get('shifted', {})]
    src_p5 = [int(d[p]['original']['pass_at_5']) for p in o20_in if 'pass_at_5' in d[p].get('original', {})]
    shf_p5 = [int(d[p]['shifted']['pass_at_5']) for p in o20_in if 'pass_at_5' in d[p].get('shifted', {})]
    def fmt(v): return f"{sum(v)/len(v)*100:.1f}% ({sum(v)}/{len(v)})" if v else '---'
    print(f"O20 src p@1={fmt(src_p1)} p@5={fmt(src_p5)}")
    print(f"O20 shf p@1={fmt(shf_p1)} p@5={fmt(shf_p5)}")
    print(f"详细 summary 通过 scripts/gen_gemini_o20_summary.py 生成")

if __name__ == '__main__':
    main()
