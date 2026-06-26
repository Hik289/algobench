#!/usr/bin/env python3
"""
Stage 1: Collect seed problems for ConstraintShift benchmark.

Supports multiple data sources via a plugin architecture:
  - HardcodedSource    : 13 curated seed problems (always included)
  - CodeContestsSource : HuggingFace deepmind/code_contests dataset
  - CodeforcesAPISource: Codeforces public API (problems + ratings)
  - LocalFileSource    : load from an existing .jsonl file

Usage:
  python3 stage1_collect.py                          # hardcoded only
  python3 stage1_collect.py --source codecontests    # + CodeContests
  python3 stage1_collect.py --source codeforces      # + Codeforces API
  python3 stage1_collect.py --source all             # all sources
  python3 stage1_collect.py --source file --file-path /path/to/problems.jsonl
  python3 stage1_collect.py --source codecontests --max-problems 200

Output schema (every source must produce this format):
  {
    "id": str,                   # unique identifier
    "title": str,
    "operator": str,             # CS | SD | OP | GT | CC | EC | OR | TBD
    "source_statement": str,
    "source_constraints": str,
    "source_input": str,
    "source_output": str,
    "source_examples": [{"input": str, "output": str}],
    "source_reference_algorithm": str,
    "source_complexity": {"time": str, "space": str},
    "shifted_statement": str,    # "" if shift_generated=False
    "shifted_constraints": str,
    "shifted_input": str,
    "shifted_output": str,
    "shifted_examples": [...],
    "target_algorithm": str,
    "target_complexity": {"time": str, "space": str},
    "old_solution_failure": str,
    "trap_patterns": [str],
    "shift_generated": bool,     # False = stage2 must generate the shift
    "data_source": str,          # which source produced this
  }
"""

import argparse
import json
import os
import sys
import time
import re
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

OUTPUT_FILE = "/path/to/algobench/data/source_problems.jsonl"
LOG_FILE    = "/path/to/algobench/logs/stage1.log"

# ── Logging ──────────────────────────────────────────────────────────────────

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ── Base class ────────────────────────────────────────────────────────────────

class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, **kwargs) -> List[Dict]:
        """Return list of problems in standard schema."""

    @staticmethod
    def make_id(prefix: str, title: str) -> str:
        h = hashlib.md5(title.encode()).hexdigest()[:6].upper()
        return f"{prefix}_{h}"

    @staticmethod
    def empty_shift() -> Dict:
        """Placeholder shifted fields for source-only problems."""
        return {
            "shifted_statement": "",
            "shifted_constraints": "",
            "shifted_input": "",
            "shifted_output": "",
            "shifted_examples": [],
            "target_algorithm": "",
            "target_complexity": {"time": "", "space": ""},
            "old_solution_failure": "",
            "trap_patterns": [],
            "shift_generated": False,
        }

# ── Source 1: Hardcoded curated problems ─────────────────────────────────────

class HardcodedSource(DataSource):
    name = "hardcoded"

    def fetch(self, **kwargs) -> List[Dict]:
        problems = []
        for p in HARDCODED_PROBLEMS:
            p = dict(p)
            p.setdefault("shift_generated", True)
            p.setdefault("data_source", self.name)
            problems.append(p)
        log(f"  [HardcodedSource] {len(problems)} problems loaded")
        return problems

# ── Source 2: HuggingFace CodeContests ───────────────────────────────────────

class CodeContestsSource(DataSource):
    name = "codecontests"

    def fetch(self, max_problems: int = 100, min_difficulty: int = 3,
              split: str = "train", **kwargs) -> List[Dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            log("  [CodeContestsSource] ERROR: `datasets` not installed. "
                "Run: pip install datasets")
            return []

        log(f"  [CodeContestsSource] Loading deepmind/code_contests (split={split})...")
        try:
            ds = load_dataset("deepmind/code_contests", split=split,
                              trust_remote_code=True)
        except Exception as e:
            log(f"  [CodeContestsSource] Failed to load dataset: {e}")
            return []

        problems = []
        seen_titles = set()

        for item in ds:
            if len(problems) >= max_problems:
                break

            # Filter by difficulty (1=A, 2=B, 3=C, 4=D, 5=E)
            diff = item.get("difficulty", 0)
            if diff < min_difficulty:
                continue

            title = item.get("name", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Extract examples
            examples = []
            inputs  = item.get("public_tests", {}).get("input", [])
            outputs = item.get("public_tests", {}).get("output", [])
            for inp, out in zip(inputs[:2], outputs[:2]):
                examples.append({"input": inp.strip(), "output": out.strip()})

            # Extract reference solution (prefer Python)
            ref_algo = "unknown"
            py_solutions = item.get("solutions", {}).get("language", [])
            # CodeContests solution languages: 0=C++, 3=Python3
            sol_langs = item.get("solutions", {}).get("language", [])
            sol_codes = item.get("solutions", {}).get("solution", [])
            py_sol = ""
            for lang, code in zip(sol_langs, sol_codes):
                if lang == 3:  # Python3
                    py_sol = code
                    break

            prob = {
                "id": self.make_id("CC", title),
                "title": title,
                "operator": "TBD",    # stage2 will assign
                "source_statement": item.get("description", "").strip()[:2000],
                "source_constraints": self._extract_constraints(
                    item.get("description", "")),
                "source_input": "",
                "source_output": "",
                "source_examples": examples,
                "source_reference_algorithm": ref_algo,
                "source_complexity": {"time": "unknown", "space": "unknown"},
                "source_python_solution": py_sol[:3000] if py_sol else "",
                "difficulty": diff,
                "data_source": self.name,
                **self.empty_shift(),
            }
            problems.append(prob)

        log(f"  [CodeContestsSource] {len(problems)} problems fetched "
            f"(difficulty >= {min_difficulty})")
        return problems

    @staticmethod
    def _extract_constraints(description: str) -> str:
        """Try to extract constraint block from problem description."""
        lines = description.split("\n")
        constraint_lines = []
        in_block = False
        for line in lines:
            low = line.lower()
            if any(k in low for k in ["constraint", "limit", "bound", "≤", "<=",
                                       "1 ≤", "1 <=", "n ≤", "n <="]):
                in_block = True
            if in_block:
                constraint_lines.append(line.strip())
            if in_block and len(constraint_lines) > 8:
                break
        return " ".join(constraint_lines[:6]) or "see problem statement"

# ── Source 3: Codeforces Public API ──────────────────────────────────────────

class CodeforcesAPISource(DataSource):
    name = "codeforces"
    BASE_URL = "https://codeforces.com/api"

    def fetch(self, min_rating: int = 1800, max_rating: int = 2500,
              tags: Optional[List[str]] = None, limit: int = 100,
              **kwargs) -> List[Dict]:
        import requests as req

        log(f"  [CodeforcesAPISource] Fetching problems "
            f"(rating {min_rating}-{max_rating})...")

        tag_str = ";".join(tags) if tags else ""
        url = f"{self.BASE_URL}/problemset.problems"
        params = {}
        if tag_str:
            params["tags"] = tag_str

        try:
            resp = req.get(url, params=params, timeout=15)
            data = resp.json()
        except Exception as e:
            log(f"  [CodeforcesAPISource] API call failed: {e}")
            return []

        if data.get("status") != "OK":
            log(f"  [CodeforcesAPISource] API error: {data.get('comment')}")
            return []

        raw_problems = data["result"]["problems"]
        stats_map = {
            (s["contestId"], s["index"]): s.get("solvedCount", 0)
            for s in data["result"]["problemStatistics"]
        }

        problems = []
        seen = set()
        for p in raw_problems:
            if len(problems) >= limit:
                break

            rating = p.get("rating", 0)
            if not (min_rating <= rating <= max_rating):
                continue

            cid   = p.get("contestId", 0)
            index = p.get("index", "")
            name  = p.get("name", "")
            key   = (cid, index)
            if key in seen or not name:
                continue
            seen.add(key)

            solved = stats_map.get(key, 0)
            prob_tags = p.get("tags", [])

            # Map CF tags to reference algorithm
            ref_algo = self._map_tags_to_algorithm(prob_tags)

            prob = {
                "id": f"CF_{cid}{index}",
                "title": name,
                "operator": "TBD",
                "source_statement": (
                    f"[Codeforces {cid}{index}] {name}\n"
                    f"Rating: {rating} | Tags: {', '.join(prob_tags)}\n"
                    f"Solved by: {solved} users\n"
                    f"Full statement: "
                    f"https://codeforces.com/problemset/problem/{cid}/{index}"
                ),
                "source_constraints": f"rating={rating}; tags={prob_tags}",
                "source_input": "",
                "source_output": "",
                "source_examples": [],
                "source_reference_algorithm": ref_algo,
                "source_complexity": {
                    "time": self._rating_to_complexity(rating),
                    "space": "O(n)",
                },
                "cf_contest_id": cid,
                "cf_index": index,
                "cf_rating": rating,
                "cf_tags": prob_tags,
                "solved_count": solved,
                "data_source": self.name,
                **self.empty_shift(),
            }
            problems.append(prob)

        log(f"  [CodeforcesAPISource] {len(problems)} problems fetched")
        return problems

    @staticmethod
    def _map_tags_to_algorithm(tags: List[str]) -> str:
        tag_map = {
            "segment tree":        "segment_tree",
            "fenwick tree":        "fenwick_tree",
            "dsu":                 "dsu",
            "shortest paths":      "dijkstra",
            "flows":               "max_flow",
            "dp":                  "dynamic_programming",
            "greedy":              "greedy",
            "binary search":       "binary_search",
            "divide and conquer":  "divide_and_conquer",
            "graphs":              "bfs_dfs",
            "trees":               "tree_dp",
            "strings":             "string_algo",
            "math":                "math",
        }
        for tag in tags:
            for key, algo in tag_map.items():
                if key in tag.lower():
                    return algo
        return "unknown"

    @staticmethod
    def _rating_to_complexity(rating: int) -> str:
        if rating < 1600: return "O(n log n)"
        if rating < 2000: return "O(n log n) or O(n log^2 n)"
        if rating < 2400: return "O(n log n) or O(n sqrt n)"
        return "O(n log n) or O(n polylog n)"

# ── Source 4: Local JSONL file ────────────────────────────────────────────────

class LocalFileSource(DataSource):
    name = "local_file"

    def fetch(self, file_path: str = "", **kwargs) -> List[Dict]:
        if not file_path or not os.path.exists(file_path):
            log(f"  [LocalFileSource] File not found: {file_path}")
            return []

        problems = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    prob = json.loads(line)
                except json.JSONDecodeError as e:
                    log(f"  [LocalFileSource] Line {i+1} parse error: {e}")
                    continue

                # Validate required fields
                required = ["id", "title", "source_statement"]
                if not all(k in prob for k in required):
                    log(f"  [LocalFileSource] Line {i+1} missing required fields, skipping")
                    continue

                prob.setdefault("operator", "TBD")
                prob.setdefault("data_source", self.name)
                prob.setdefault("shift_generated",
                                bool(prob.get("shifted_statement", "").strip()))
                if not prob["shift_generated"]:
                    prob.update(self.empty_shift())

                problems.append(prob)

        log(f"  [LocalFileSource] {len(problems)} problems loaded from {file_path}")
        return problems

# ── Hardcoded seed problems ───────────────────────────────────────────────────
# (13 curated problems with pre-defined shifted variants and trap patterns)

HARDCODED_PROBLEMS = [
{
  "id": "CS001", "title": "Range Sum Query (Static→Dynamic with Lazy SegTree)",
  "operator": "CS",
  "source_statement": (
    "Given a static array A of n ≤ 2000 integers, answer Q ≤ 2000 queries. "
    "Each query (l, r) asks for the sum of A[l..r]."
  ),
  "source_constraints": "1 ≤ n, Q ≤ 2000; −10^9 ≤ A[i] ≤ 10^9",
  "source_input": "First line: n Q. Second line: n integers. Next Q lines: l r.",
  "source_output": "Q lines, each the sum A[l..r].",
  "source_examples": [{"input": "5 3\n1 2 3 4 5\n1 3\n2 5\n1 5", "output": "6\n14\n15"}],
  "source_reference_algorithm": "prefix_sum",
  "source_complexity": {"time": "O(n+Q)", "space": "O(n)"},
  "shifted_statement": (
    "Given an array A of n ≤ 2×10^5 integers and Q ≤ 2×10^5 operations. "
    "Operation type 1: add x to all elements in A[l..r]. "
    "Operation type 2: report the maximum element in A[l..r]."
  ),
  "shifted_constraints": "1 ≤ n, Q ≤ 2×10^5",
  "shifted_input": "First line: n Q. Second line: n integers. Next Q lines: type l r [x].",
  "shifted_output": "For each type-2 query, print the maximum.",
  "shifted_examples": [{"input": "5 3\n1 2 3 4 5\n1 1 3 2\n2 1 5\n2 2 4", "output": "7\n7"}],
  "target_algorithm": "lazy_segment_tree",
  "target_complexity": {"time": "O((n+Q)log n)", "space": "O(n)"},
  "old_solution_failure": "TLE — prefix sum cannot handle range-add updates",
  "trap_patterns": ["prefix", "cumsum", "presum", "for i in range(l, r+1)"],
},
{
  "id": "CS002", "title": "Counting Pairs with Sum",
  "operator": "CS",
  "source_statement": "Given array A of n ≤ 3000 integers and target T, count pairs (i,j) i<j with A[i]+A[j]=T.",
  "source_constraints": "1 ≤ n ≤ 3000",
  "source_input": "Line 1: n T. Line 2: n integers.",
  "source_output": "Number of pairs.",
  "source_examples": [{"input": "4 5\n1 4 3 2", "output": "2"}],
  "source_reference_algorithm": "brute_force_O(n2)",
  "source_complexity": {"time": "O(n²)", "space": "O(1)"},
  "shifted_statement": "Given array A of n ≤ 10^6 integers and target T, count pairs (i,j) i<j with A[i]+A[j]=T.",
  "shifted_constraints": "1 ≤ n ≤ 10^6",
  "shifted_input": "Line 1: n T. Line 2: n integers.",
  "shifted_output": "Number of pairs.",
  "shifted_examples": [{"input": "4 5\n1 4 3 2", "output": "2"}],
  "target_algorithm": "hashmap_or_twopointer",
  "target_complexity": {"time": "O(n)", "space": "O(n)"},
  "old_solution_failure": "TLE (O(n²) with n=10^6)",
  "trap_patterns": ["for i in range(n)", "for j in range(i+1, n)", "A[i] + A[j]"],
},
{
  "id": "CS003", "title": "Shortest Path (Bellman→Dijkstra)",
  "operator": "CS",
  "source_statement": "Given weighted directed graph n ≤ 500, m ≤ 5000, find shortest path 1→n.",
  "source_constraints": "1 ≤ n ≤ 500; 1 ≤ m ≤ 5000",
  "source_input": "Line 1: n m. Next m lines: u v w.",
  "source_output": "Shortest distance or -1.",
  "source_examples": [{"input": "4 5\n1 2 1\n2 3 2\n3 4 3\n1 3 10\n2 4 15", "output": "6"}],
  "source_reference_algorithm": "bellman_ford",
  "source_complexity": {"time": "O(nm)", "space": "O(n²)"},
  "shifted_statement": "Given weighted directed graph n ≤ 2×10^5, m ≤ 5×10^5, find shortest path 1→n.",
  "shifted_constraints": "1 ≤ n ≤ 2×10^5; 1 ≤ m ≤ 5×10^5",
  "shifted_input": "Line 1: n m. Next m lines: u v w.",
  "shifted_output": "Shortest distance or -1.",
  "shifted_examples": [{"input": "4 5\n1 2 1\n2 3 2\n3 4 3\n1 3 10\n2 4 15", "output": "6"}],
  "target_algorithm": "dijkstra_with_heap",
  "target_complexity": {"time": "O((n+m)log n)", "space": "O(n+m)"},
  "old_solution_failure": "TLE (Bellman-Ford O(nm) ≈ 10^10 ops)",
  "trap_patterns": ["for _ in range(n-1)", "relax all edges", "bellman", "floyd"],
},
{
  "id": "SD001", "title": "Static Connectivity → Dynamic (with Deletions)",
  "operator": "SD",
  "source_statement": "Given n nodes, m edges (upfront), answer Q connectivity queries.",
  "source_constraints": "1 ≤ n ≤ 10^5; 1 ≤ m, Q ≤ 2×10^5",
  "source_input": "Line 1: n m. Next m lines: u v. Line: Q. Next Q: u v.",
  "source_output": "YES or NO for each query.",
  "source_examples": [{"input": "4 3\n1 2\n2 3\n3 4\n2\n1 4\n1 5", "output": "YES\nNO"}],
  "source_reference_algorithm": "dsu",
  "source_complexity": {"time": "O((m+Q)α(n))", "space": "O(n)"},
  "shifted_statement": "Given n nodes, Q ops: add edge, delete edge (guaranteed present), or query connectivity.",
  "shifted_constraints": "1 ≤ n ≤ 2×10^5; 1 ≤ Q ≤ 2×10^5",
  "shifted_input": "Line 1: n Q. Next Q lines: op u v.",
  "shifted_output": "YES or NO for each type-3 query.",
  "shifted_examples": [{"input": "4 5\n1 1 2\n1 2 3\n3 1 3\n2 2 3\n3 1 3", "output": "YES\nNO"}],
  "target_algorithm": "offline_dynamic_connectivity_dsu_rollback",
  "target_complexity": {"time": "O(Q log Q α(n))", "space": "O(n+Q)"},
  "old_solution_failure": "WA — DSU cannot handle deletions",
  "trap_patterns": ["parent[find(u)] = find(v)", "union(", "find(", "DisjointSet"],
},
{
  "id": "SD002", "title": "Prefix Sum → BIT (Point Update)",
  "operator": "SD",
  "source_statement": "Given array A of n ≤ 10^5 integers, answer Q range-sum queries.",
  "source_constraints": "1 ≤ n, Q ≤ 10^5",
  "source_input": "Line 1: n Q. Line 2: n integers. Next Q lines: l r.",
  "source_output": "Q sums.",
  "source_examples": [{"input": "5 2\n1 2 3 4 5\n1 3\n2 4", "output": "6\n9"}],
  "source_reference_algorithm": "prefix_sum",
  "source_complexity": {"time": "O(n+Q)", "space": "O(n)"},
  "shifted_statement": "Given array A of n ≤ 5×10^5, Q ops: point update A[i]+=x or range-sum query.",
  "shifted_constraints": "1 ≤ n, Q ≤ 5×10^5",
  "shifted_input": "Line 1: n Q. Line 2: n integers. Next Q: 1 i x or 2 l r.",
  "shifted_output": "For each type-2 query, print sum.",
  "shifted_examples": [{"input": "5 3\n1 2 3 4 5\n1 3 10\n2 1 4\n2 2 5", "output": "20\n24"}],
  "target_algorithm": "fenwick_tree_BIT",
  "target_complexity": {"time": "O((n+Q)log n)", "space": "O(n)"},
  "old_solution_failure": "TLE+WA — prefix sum cannot handle point updates",
  "trap_patterns": ["prefix", "cumsum", "presum[r] - presum[l-1]"],
},
{
  "id": "SD003", "title": "BFS Reachability → Online DSU",
  "operator": "SD",
  "source_statement": "Given graph n ≤ 10^5, m edges (upfront), answer Q BFS queries.",
  "source_constraints": "1 ≤ n ≤ 10^5; 1 ≤ m, Q ≤ 2×10^5",
  "source_input": "Line 1: n m. Next m: u v. Line: Q. Next Q: u K.",
  "source_output": "Q answers.",
  "source_examples": [{"input": "4 3\n1 2\n2 3\n3 4\n2\n1 2\n2 1", "output": "3\n2"}],
  "source_reference_algorithm": "bfs_per_query",
  "source_complexity": {"time": "O(Q(n+m))", "space": "O(n+m)"},
  "shifted_statement": "Empty graph n nodes, Q ops: add edge or query component size.",
  "shifted_constraints": "1 ≤ n ≤ 10^5; 1 ≤ Q ≤ 2×10^5",
  "shifted_input": "Line 1: n Q. Next Q: 1 u v or 2 u.",
  "shifted_output": "For each type-2 query, component size.",
  "shifted_examples": [{"input": "4 4\n1 1 2\n1 2 3\n2 1\n1 3 4", "output": "3"}],
  "target_algorithm": "online_dsu_with_size",
  "target_complexity": {"time": "O(Q α(n))", "space": "O(n)"},
  "old_solution_failure": "TLE — BFS per query too slow for online insertions",
  "trap_patterns": ["bfs", "queue", "visited", "deque"],
},
{
  "id": "OP001", "title": "MST → Minimax Path Query",
  "operator": "OP",
  "source_statement": "Given weighted undirected graph n ≤ 2×10^5, m ≤ 5×10^5, find MST weight.",
  "source_constraints": "1 ≤ n ≤ 2×10^5; 1 ≤ m ≤ 5×10^5",
  "source_input": "Line 1: n m. Next m: u v w.",
  "source_output": "MST total weight.",
  "source_examples": [{"input": "4 5\n1 2 1\n2 3 2\n3 4 3\n1 3 4\n2 4 5", "output": "6"}],
  "source_reference_algorithm": "kruskal",
  "source_complexity": {"time": "O(m log m)", "space": "O(n+m)"},
  "shifted_statement": "Same graph + Q queries. Each query (u,v): min possible max edge on any u-v path.",
  "shifted_constraints": "1 ≤ n ≤ 2×10^5; 1 ≤ m ≤ 5×10^5; 1 ≤ Q ≤ 2×10^5",
  "shifted_input": "Line 1: n m Q. Next m: u v w. Next Q: u v.",
  "shifted_output": "Q minimax path values.",
  "shifted_examples": [{"input": "4 5 2\n1 2 1\n2 3 2\n3 4 3\n1 3 4\n2 4 5\n1 4\n2 3", "output": "3\n2"}],
  "target_algorithm": "kruskal_plus_lca_on_kruskal_tree",
  "target_complexity": {"time": "O((m+Q)log n)", "space": "O(n)"},
  "old_solution_failure": "WA — MST total weight ≠ minimax path queries",
  "trap_patterns": ["minimum_spanning_tree", "kruskal", "total_weight", "mst_weight"],
},
{
  "id": "OP002", "title": "Max Subarray → Count Positive Subarrays",
  "operator": "OP",
  "source_statement": "Given array A of n ≤ 10^5 integers, find maximum subarray sum.",
  "source_constraints": "1 ≤ n ≤ 10^5",
  "source_input": "Line 1: n. Line 2: n integers.",
  "source_output": "Maximum subarray sum.",
  "source_examples": [{"input": "8\n-2 1 -3 4 -1 2 1 -5", "output": "6"}],
  "source_reference_algorithm": "kadane",
  "source_complexity": {"time": "O(n)", "space": "O(1)"},
  "shifted_statement": "Given array A of n ≤ 2×10^5 integers, count contiguous subarrays with sum > 0.",
  "shifted_constraints": "1 ≤ n ≤ 2×10^5",
  "shifted_input": "Line 1: n. Line 2: n integers.",
  "shifted_output": "Number of subarrays with sum > 0.",
  "shifted_examples": [{"input": "4\n1 -2 3 -1", "output": "5"}],
  "target_algorithm": "prefix_sum_with_BIT",
  "target_complexity": {"time": "O(n log n)", "space": "O(n)"},
  "old_solution_failure": "WA — Kadane finds max subarray, not count of positives",
  "trap_patterns": ["kadane", "max_sum", "current_sum = max(0", "dp[i] = max(A[i]"],
},
{
  "id": "GT001", "title": "Interval Scheduling with Color-Switch Budget",
  "operator": "GT",
  "source_statement": "Given n ≤ 10^5 intervals [s,e], select max non-overlapping.",
  "source_constraints": "1 ≤ n ≤ 10^5",
  "source_input": "Line 1: n. Next n: s e.",
  "source_output": "Maximum non-overlapping intervals.",
  "source_examples": [{"input": "4\n1 3\n2 4\n3 5\n4 6", "output": "2"}],
  "source_reference_algorithm": "greedy_earliest_finish",
  "source_complexity": {"time": "O(n log n)", "space": "O(1)"},
  "shifted_statement": "Same intervals with color c_i. Select max non-overlapping with ≤ K consecutive color switches.",
  "shifted_constraints": "1 ≤ n ≤ 10^5; 1 ≤ c_i ≤ k ≤ 10; 0 ≤ K ≤ n",
  "shifted_input": "Line 1: n K. Next n: s e c.",
  "shifted_output": "Maximum intervals satisfying switch constraint.",
  "shifted_examples": [{"input": "4 1\n1 3 1\n2 4 2\n3 5 1\n4 6 2", "output": "2"}],
  "target_algorithm": "dp_with_switch_count",
  "target_complexity": {"time": "O(n² K)", "space": "O(nK)"},
  "old_solution_failure": "WA — greedy ignores global switch budget",
  "trap_patterns": ["sort.*finish", "earliest_finish", "if start >= last_end", "greedy"],
},
{
  "id": "GT002", "title": "Job Scheduling with Cooling Constraint",
  "operator": "GT",
  "source_statement": "n ≤ 10^5 jobs with deadlines d_i and profits p_i. Maximize profit (1 job/slot, before deadline).",
  "source_constraints": "1 ≤ n ≤ 10^5; 1 ≤ d_i ≤ n",
  "source_input": "Line 1: n. Next n: d p.",
  "source_output": "Maximum total profit.",
  "source_examples": [{"input": "4\n2 100\n1 50\n2 80\n1 20", "output": "180"}],
  "source_reference_algorithm": "greedy_by_profit_with_dsu",
  "source_complexity": {"time": "O(n log n)", "space": "O(n)"},
  "shifted_statement": "Same but each job has type t_i. After scheduling type t, wait ≥ C slots before same type again.",
  "shifted_constraints": "1 ≤ n ≤ 10^5; 1 ≤ t_i ≤ m ≤ 10; 1 ≤ C ≤ n",
  "shifted_input": "Line 1: n m C. Next n: d p t.",
  "shifted_output": "Maximum total profit.",
  "shifted_examples": [{"input": "4 2 2\n3 100 1\n3 80 1\n3 60 2\n3 50 2", "output": "160"}],
  "target_algorithm": "dp_with_type_cooldown",
  "target_complexity": {"time": "O(n² m)", "space": "O(nm)"},
  "old_solution_failure": "WA — greedy ignores cooldown; schedules same-type consecutively",
  "trap_patterns": ["sort.*profit", "descending.*profit", "dsu", "find(d_i)"],
},
{
  "id": "GT003", "title": "Fractional Knapsack → 0/1 Knapsack",
  "operator": "GT",
  "source_statement": "n ≤ 10^4 items w_i,v_i, capacity W ≤ 10^6. Fractional amounts allowed. Maximize value.",
  "source_constraints": "1 ≤ n ≤ 10^4; 1 ≤ W ≤ 10^6",
  "source_input": "Line 1: n W. Next n: w v.",
  "source_output": "Max value (2 decimal places).",
  "source_examples": [{"input": "3 50\n10 60\n20 100\n30 120", "output": "240.00"}],
  "source_reference_algorithm": "greedy_by_value_per_weight",
  "source_complexity": {"time": "O(n log n)", "space": "O(1)"},
  "shifted_statement": "Same but items must be taken whole or not at all (0/1 knapsack). n ≤ 500, W ≤ 10^4.",
  "shifted_constraints": "1 ≤ n ≤ 500; 1 ≤ W ≤ 10^4",
  "shifted_input": "Line 1: n W. Next n: w v.",
  "shifted_output": "Maximum integer value.",
  "shifted_examples": [{"input": "3 50\n10 60\n20 100\n30 120", "output": "220"}],
  "target_algorithm": "01_knapsack_dp",
  "target_complexity": {"time": "O(nW)", "space": "O(W)"},
  "old_solution_failure": "WA — greedy by v/w optimal for fractional, not 0/1",
  "trap_patterns": ["v_i/w_i", "value_per_weight", "sort.*ratio", "take.*fraction"],
},
{
  "id": "CS004", "title": "Fibonacci Large N (DP→Matrix Exp)",
  "operator": "CS",
  "source_statement": "Compute F(n) mod 10^9+7, n ≤ 10^6.",
  "source_constraints": "0 ≤ n ≤ 10^6",
  "source_input": "Single integer n.",
  "source_output": "F(n) mod 10^9+7.",
  "source_examples": [{"input": "10", "output": "55"}],
  "source_reference_algorithm": "iterative_dp",
  "source_complexity": {"time": "O(n)", "space": "O(1)"},
  "shifted_statement": "Compute F(n) mod 10^9+7, n ≤ 10^18.",
  "shifted_constraints": "0 ≤ n ≤ 10^18",
  "shifted_input": "Single integer n.",
  "shifted_output": "F(n) mod 10^9+7.",
  "shifted_examples": [{"input": "10", "output": "55"}],
  "target_algorithm": "matrix_exponentiation",
  "target_complexity": {"time": "O(log n)", "space": "O(1)"},
  "old_solution_failure": "TLE — O(n) loop with n=10^18",
  "trap_patterns": ["for i in range(n)", "dp[i] = dp[i-1] + dp[i-2]", "a, b = b, a+b"],
},
{
  "id": "CS005", "title": "LCS of Permutations (O(n²)→O(n log n))",
  "operator": "CS",
  "source_statement": "Given strings S, T with |S|,|T| ≤ 3000, find LCS length.",
  "source_constraints": "1 ≤ |S|, |T| ≤ 3000",
  "source_input": "Two lines: S and T.",
  "source_output": "LCS length.",
  "source_examples": [{"input": "ABCBDAB\nBDCAB", "output": "4"}],
  "source_reference_algorithm": "dp_O(nm)",
  "source_complexity": {"time": "O(|S||T|)", "space": "O(|S||T|)"},
  "shifted_statement": "Given permutations S, T of 1..n with n ≤ 2×10^5, find LCS length.",
  "shifted_constraints": "1 ≤ n ≤ 2×10^5; S and T are permutations of 1..n",
  "shifted_input": "Line 1: n. Line 2: S. Line 3: T.",
  "shifted_output": "LCS length.",
  "shifted_examples": [{"input": "5\n2 1 4 3 5\n1 2 3 4 5", "output": "3"}],
  "target_algorithm": "LCS_as_LIS_via_inverse_permutation",
  "target_complexity": {"time": "O(n log n)", "space": "O(n)"},
  "old_solution_failure": "TLE — O(n²) DP with n=2×10^5",
  "trap_patterns": ["dp[i][j]", "dp[i-1][j-1]+1", "2D dp", "LCS classic"],
},
]

# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(problems: List[Dict]) -> List[Dict]:
    seen_ids   = set()
    seen_titles = set()
    result = []
    for p in problems:
        pid   = p.get("id", "")
        title = p.get("title", "").lower().strip()
        if pid in seen_ids or title in seen_titles:
            continue
        seen_ids.add(pid)
        seen_titles.add(title)
        result.append(p)
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ConstraintShift Stage 1: Collect seed problems")
    parser.add_argument("--source", default="hardcoded",
        choices=["hardcoded", "codecontests", "codeforces", "file", "all"],
        help="Data source(s) to use (default: hardcoded)")
    parser.add_argument("--file-path", default="",
        help="Path to local JSONL file (for --source file)")
    parser.add_argument("--max-problems", type=int, default=100,
        help="Max problems to fetch from external sources (default: 100)")
    parser.add_argument("--min-difficulty", type=int, default=3,
        help="Min difficulty for CodeContests (1-5, default: 3)")
    parser.add_argument("--cf-min-rating", type=int, default=1800,
        help="Min Codeforces rating (default: 1800)")
    parser.add_argument("--cf-max-rating", type=int, default=2500,
        help="Max Codeforces rating (default: 2500)")
    parser.add_argument("--cf-tags", nargs="+", default=None,
        help="Codeforces tags to filter by (e.g. --cf-tags dp greedy)")
    parser.add_argument("--output", default=OUTPUT_FILE,
        help="Output JSONL path")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log(f"Stage 1: Collecting seed problems (source={args.source})")

    all_problems: List[Dict] = []

    # Always include hardcoded problems
    hc = HardcodedSource()
    all_problems.extend(hc.fetch())

    if args.source in ("codecontests", "all"):
        cc = CodeContestsSource()
        all_problems.extend(cc.fetch(
            max_problems=args.max_problems,
            min_difficulty=args.min_difficulty,
        ))

    if args.source in ("codeforces", "all"):
        cf = CodeforcesAPISource()
        all_problems.extend(cf.fetch(
            min_rating=args.cf_min_rating,
            max_rating=args.cf_max_rating,
            tags=args.cf_tags,
            limit=args.max_problems,
        ))

    if args.source == "file":
        lf = LocalFileSource()
        all_problems.extend(lf.fetch(file_path=args.file_path))

    # Deduplicate
    before = len(all_problems)
    all_problems = deduplicate(all_problems)
    after = len(all_problems)
    log(f"Deduplication: {before} → {after} problems")

    # Add metadata
    for p in all_problems:
        p["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write output
    with open(args.output, "w") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    ops   = Counter(p.get("operator", "?") for p in all_problems)
    srcs  = Counter(p.get("data_source", "?") for p in all_problems)
    shift = sum(1 for p in all_problems if p.get("shift_generated", False))

    log(f"\nStage 1 complete: {after} problems → {args.output}")
    log(f"  Operators:    {dict(ops)}")
    log(f"  Sources:      {dict(srcs)}")
    log(f"  Shift ready:  {shift}/{after} (rest need stage2 to generate shift)")

if __name__ == "__main__":
    main()
