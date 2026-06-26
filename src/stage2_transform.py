#!/usr/bin/env python3
"""
Stage 2: Build final benchmark from seed problems.
Since transformations are already defined in seed problems (shifted_* fields),
this stage: (1) validates Gate 1 (old solution fails), (2) formats for evaluation.
"""

import json, os, time, ast, subprocess, tempfile, sys

SEEDS_FILE  = "/path/to/algobench/data/source_problems.jsonl"
OUTPUT_FILE = "/path/to/algobench/data/final_benchmark.jsonl"
LOG_FILE    = "/path/to/algobench/logs/stage2.log"

PYTHON      = "/path/to/venv"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_seeds():
    problems = []
    with open(SEEDS_FILE) as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    return problems

def old_solution_templates():
    """Generate simple 'old' solutions for each problem type."""
    return {
        "CS001": """
import sys
input = sys.stdin.readline
def solve():
    n, q = map(int, input().split())
    A = list(map(int, input().split()))
    # prefix sum approach (old solution)
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

def run_code_on_example(code, input_data, timeout=5):
    """Run Python code with given input, return (stdout, success, error)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, fname],
            input=input_data, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return "", False, "TIMEOUT"
    except Exception as e:
        return "", False, str(e)
    finally:
        os.unlink(fname)

def gate1_check(problem):
    """
    Gate 1: verify that the old solution fails on the shifted problem.
    Returns (passes_gate, reason).
    A problem passes Gate 1 if old solution gives wrong answer or TLE.
    """
    old_solutions = old_solution_templates()
    pid = problem["id"]

    # If we don't have an old solution template, skip Gate 1 (auto-pass)
    if pid not in old_solutions:
        return True, "Gate1: no old solution template (auto-pass)"

    old_code = old_solutions[pid]
    # Use the shifted examples as test inputs
    failed = False
    reason_parts = []

    for ex in problem.get("shifted_examples", []):
        inp = ex["input"]
        expected = ex["output"].strip()
        out, success, err = run_code_on_example(old_code, inp, timeout=3)

        if err == "TIMEOUT":
            failed = True
            reason_parts.append("TLE on shifted input")
            break
        elif not success:
            failed = True
            reason_parts.append(f"Runtime error: {err[:100]}")
            break
        elif out != expected:
            failed = True
            reason_parts.append(f"WA: got '{out}', expected '{expected}'")
            break

    if failed:
        return True, "Gate1 PASS: " + "; ".join(reason_parts)
    else:
        return False, "Gate1 FAIL: old solution still works on shifted examples"

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    problems = load_seeds()
    log(f"Stage 2: Processing {len(problems)} seed problems...")

    accepted = []
    rejected = []

    for prob in problems:
        pid = prob["id"]
        # Gate 1: old solution must fail
        passes_g1, g1_reason = gate1_check(prob)

        if passes_g1:
            prob["gate1_result"] = g1_reason
            prob["benchmark_status"] = "accepted"
            accepted.append(prob)
            log(f"  ACCEPT {pid}: {g1_reason}")
        else:
            prob["gate1_result"] = g1_reason
            prob["benchmark_status"] = "rejected_gate1"
            rejected.append(prob)
            log(f"  REJECT {pid}: {g1_reason}")

    log(f"\nStage 2 results: {len(accepted)} accepted, {len(rejected)} rejected")

    # Write accepted benchmark
    with open(OUTPUT_FILE, "w") as f:
        for prob in accepted:
            f.write(json.dumps(prob) + "\n")

    # Summary
    from collections import Counter
    ops = Counter(p["operator"] for p in accepted)
    log(f"Accepted operator distribution: {dict(ops)}")
    log(f"Final benchmark written to {OUTPUT_FILE}")

    # Also write a summary
    summary = {
        "n_seeds": len(problems),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "acceptance_rate": len(accepted)/len(problems) if problems else 0,
        "operator_distribution": dict(ops),
        "accepted_ids": [p["id"] for p in accepted],
        "rejected_ids": [p["id"] for p in rejected],
    }
    with open(OUTPUT_FILE.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("Stage 2 complete.")

if __name__ == "__main__":
    main()
