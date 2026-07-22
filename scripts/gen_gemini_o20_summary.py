#!/usr/bin/env python3
"""生成 analysis/gemini_o20_summary.txt — 修复了 eval 脚本里 f 变量遮蔽的 bug。"""
import json, time
from pathlib import Path

ROOT  = Path('/path/to/project_workspace/algorithm_design')
RFILE = ROOT / 'results/multimodel_results.json'

ORIG11 = ['CS001','CS003','CS005','SD001','SD002','SD003','OP001','OP002','GT001','GT002','GT003']
HARD9  = ['CS_H1','CS_H2','SD_H1','OP_H1','GT_H1','GT_H2','SD_H2','OP_H2','CS_H3']
O20    = ORIG11 + HARD9

mr = json.load(open(RFILE))
d  = mr['gemini-2.5-flash__direct']

def fmt(v):
    return f"{sum(v)/len(v)*100:.1f}% ({sum(v)}/{len(v)})" if v else '---'

src_p1 = [int(d[p]['original']['pass_at_1']) for p in O20 if 'pass_at_1' in d.get(p,{}).get('original',{})]
shf_p1 = [int(d[p]['shifted']['pass_at_1'])  for p in O20 if 'pass_at_1' in d.get(p,{}).get('shifted',{})]
src_p5 = [int(d[p]['original']['pass_at_5']) for p in O20 if 'pass_at_5' in d.get(p,{}).get('original',{})]
shf_p5 = [int(d[p]['shifted']['pass_at_5'])  for p in O20 if 'pass_at_5' in d.get(p,{}).get('shifted',{})]
shf_ot = [d[p]['shifted']['opt_t']           for p in O20 if 'opt_t' in d.get(p,{}).get('shifted',{})]
shf_os = [d[p]['shifted']['opt_s']           for p in O20 if 'opt_s' in d.get(p,{}).get('shifted',{})]
shf_tr = [d[p]['shifted']['trap_rate']       for p in O20 if 'trap_rate' in d.get(p,{}).get('shifted',{})]
src_ot = [d[p]['original']['opt_t']          for p in O20 if 'opt_t' in d.get(p,{}).get('original',{})]

def mean(v): return f"{sum(v)/len(v)*100:.1f}%" if v else "---"

out_path = ROOT / 'analysis/gemini_o20_summary.txt'
with open(out_path, 'w') as fh:
    fh.write(f"# Gemini-2.5-flash direct on O20 — N=5 samples\n")
    fh.write(f"运行时刻 JST: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
    fh.write(f"\n## 总体 (20 题)\n")
    fh.write(f"  shifted : p@1={fmt(shf_p1)}  p@5={fmt(shf_p5)}  OptT={mean(shf_ot)}  OptS={mean(shf_os)}  TrapRate={mean(shf_tr)}\n")
    fh.write(f"  original: p@1={fmt(src_p1)}  p@5={fmt(src_p5)}  OptT={mean(src_ot)}\n")
    fh.write(f"\n## ORIG11 (11 题)\n")
    fh.write(f"{'PID':<8} | {'shf p@1':<8} {'p@5':<5} {'OptT':<5} {'OptS':<5} {'Trap':<5} | {'src p@1':<8} {'p@5':<5} {'OptT':<5}\n")
    fh.write("-"*88 + "\n")
    for pid in ORIG11:
        if pid not in d: continue
        shf = d[pid].get('shifted', {})
        src = d[pid].get('original', {})
        fh.write(f"{pid:<8} | {'✓' if shf.get('pass_at_1') else '✗':<8} "
                 f"{'✓' if shf.get('pass_at_5') else '✗':<5} "
                 f"{shf.get('opt_t',0)*100:>4.0f}% {shf.get('opt_s',0)*100:>4.0f}% "
                 f"{shf.get('trap_rate',0)*100:>4.0f}% | "
                 f"{'✓' if src.get('pass_at_1') else '✗':<8} "
                 f"{'✓' if src.get('pass_at_5') else '✗':<5} "
                 f"{src.get('opt_t',0)*100:>4.0f}%\n")
    fh.write(f"\n## HARD9 (9 题)\n")
    fh.write(f"{'PID':<8} | {'shf p@1':<8} {'p@5':<5} {'OptT':<5} {'OptS':<5} {'Trap':<5} | {'src p@1':<8} {'p@5':<5} {'OptT':<5}\n")
    fh.write("-"*88 + "\n")
    for pid in HARD9:
        if pid not in d: continue
        shf = d[pid].get('shifted', {})
        src = d[pid].get('original', {})
        fh.write(f"{pid:<8} | {'✓' if shf.get('pass_at_1') else '✗':<8} "
                 f"{'✓' if shf.get('pass_at_5') else '✗':<5} "
                 f"{shf.get('opt_t',0)*100:>4.0f}% {shf.get('opt_s',0)*100:>4.0f}% "
                 f"{shf.get('trap_rate',0)*100:>4.0f}% | "
                 f"{'✓' if src.get('pass_at_1') else '✗':<8} "
                 f"{'✓' if src.get('pass_at_5') else '✗':<5} "
                 f"{src.get('opt_t',0)*100:>4.0f}%\n")

    fh.write(f"\n## Schema 检查\n")
    fh.write(f"  Gemini direct 总题数: {len(d)} (27 → {len(d)})\n")
    complete = sum(1 for pid in O20 if pid in d and 'sample_results' in d[pid].get('shifted',{}) and 'sample_results' in d[pid].get('original',{}))
    fh.write(f"  O20 完整 schema (含 sample_results): {complete}/20\n")

print(open(out_path).read())
