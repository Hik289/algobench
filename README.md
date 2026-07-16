<h1 align="center">AlgoBench</h1>

<p align="center">
  <strong>Benchmarking Algorithmic Adaptation in Code Generation</strong>
</p>

<p align="center">
  Xinyuan Song &nbsp;&middot;&nbsp; Zekun Cai &nbsp;&middot;&nbsp; Liang Zhao
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.00062"><img src="https://img.shields.io/badge/arXiv-2607.00062-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT license"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg" alt="Python 3.9+"></a>
</p>

<p align="center">
  <img src="assets/algobench_intuition.png" alt="AlgoBench intuition: generate shifted algorithmic problems to test adaptation beyond memorized benchmark tasks." width="96%">
</p>

AlgoBench is the official code release for **[AlgoBench: Benchmarking Algorithmic Adaptation in Code Generation](https://arxiv.org/abs/2607.00062)**. It evaluates whether code-generation models can adapt algorithms to newly generated programming problems, instead of recalling solutions to fixed public benchmarks.

Many established programming benchmarks eventually become part of the public training ecosystem through released statements, editorials, tests, and generated solutions. AlgoBench addresses this by automatically constructing **constraint-shifted algorithmic variants** from known competitive-programming problems. Each accepted variant is traceable to a source problem, but the original reference algorithm must fail on the shifted problem.

## At a Glance

- **Research question.** Can code-generation models adapt algorithms to shifted problem constraints rather than memorize public benchmark solutions?
- **Core idea.** AlgoBench constructs traceable constraint-shifted variants whose original reference algorithms fail under the new setting.
- **What is included.** Benchmark snapshots, transformation operators, model-output evaluation, analysis scripts, and reproducibility protocols.

## Key Contributions

- **Generative benchmark construction.** Build new algorithmic tasks through structured transformations of known source problems.
- **Contamination-aware evaluation.** Fresh or private shifted variants reduce dependence on exact-item memorization.
- **Algorithmic adaptation focus.** Models must identify what changed, explain why the source algorithm fails, and produce a new efficient solution.
- **Complexity-aware metrics.** AlgoBench reports OPTT, OPTS, TRAPRATE, GAPT, CONSENS, and pass@k rather than only functional correctness.
- **Released benchmark snapshot.** This repository includes a reproducible 52-problem instance, model outputs, and analysis tables.

## Construction Pipeline

<p align="center">
  <img src="assets/algobench_pipeline.png" alt="AlgoBench pipeline from source problems to shifted tasks, quality gates, model evaluation, and analysis." width="96%">
</p>

AlgoBench treats benchmark creation as part of the evaluation protocol:

1. collect seed algorithmic problems with known reference algorithms;
2. transform the source problem through a structured constraint shift;
3. reject invalid, superficial, or weak variants through quality gates;
4. evaluate models on the accepted shifted tasks;
5. score both functional behavior and algorithmic suitability.

The released benchmark snapshot is stored in [`data/final_benchmark.jsonl`](data/final_benchmark.jsonl).

## Transformation Operators

The current release contains 52 accepted shifted problems:

| Operator | Count | What changes? | Expected adaptation |
| --- | ---: | --- | --- |
| `CS` Constraint Scale | 16 | Input scale or resource limits invalidate the source complexity. | Replace the source method with a lower-complexity algorithm or data structure. |
| `SD` Static to Dynamic | 13 | A one-shot task becomes online, streaming, or update-query based. | Use dynamic data structures or incremental maintenance. |
| `OP` Objective Perturbation | 10 | The optimization target changes while part of the original structure remains. | Re-derive the objective and select the new algorithmic family. |
| `GT` Greedy Trap | 13 | A standard greedy rule becomes incorrect under a new condition. | Detect the counterexample and use an exact or dynamic-programming method. |

## Metrics Beyond Correctness

AlgoBench separates "the code ran" from "the algorithm is appropriate for the shifted problem."

| Metric | Purpose |
| --- | --- |
| `pass@k` | Functional correctness under sampled generations. |
| `OPTT` | Fraction of correct solutions meeting the target time complexity. |
| `OPTS` | Fraction of correct solutions meeting the target space complexity. |
| `TRAPRATE` | Frequency of failures that reuse the invalidated source algorithm or a known trap pattern. |
| `GAPT` | Gap between functional success and target-complexity success. |
| `CONSENS` | Constraint-sensitivity analysis across increasing problem scales. |

These metrics are useful because many solutions that look correct on examples still fail the required asymptotic behavior of the generated problem.

## Repository Structure

```text
algobench/
|-- assets/                 # README figures
|-- data/
|   |-- final_benchmark.jsonl
|   |-- source_problems.jsonl
|   `-- final_benchmark_summary.json
|-- src/
|   |-- stage1_collect.py
|   |-- stage2_transform.py
|   |-- stage4_evaluate.py
|   |-- stage4_multimodel.py
|   |-- stage5_analyze.py
|   `-- stage6_generalization.py
|-- scripts/                # ablations, checker validation, figure/table generation
|-- results/                # model-level evaluation records
|-- analysis/               # processed tables and figure data
|-- run_pipeline.sh
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Installation

```bash
git clone git@github.com:Hik289/algobench.git
cd algobench

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set the API keys for the models you want to evaluate:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Quick Start

Inspect the released benchmark snapshot:

```bash
python - <<'PY'
import json
from collections import Counter

path = "data/final_benchmark.jsonl"
items = [json.loads(line) for line in open(path)]
print(f"{len(items)} shifted problems")
print(Counter(item["operator"] for item in items))
print(items[0]["id"], "-", items[0]["title"])
PY
```

Run the portable end-to-end pipeline:

```bash
bash run_pipeline.sh
```

If no API key is available, the pipeline still runs collection and transformation stages, then skips API-based model evaluation.

## Evaluate Models

Run the released multi-model evaluator:

```bash
python src/stage4_multimodel.py
```

Run a single-model evaluator:

```bash
python src/stage4_evaluate.py
```

Analyze results:

```bash
python src/stage5_analyze.py
python src/stage6_generalization.py
python scripts/recompute_tables.py
```

Precomputed model outputs and processed paper tables are included under [`results/`](results/) and [`analysis/`](analysis/).

## Data Format

Each line in `data/final_benchmark.jsonl` is one shifted problem:

| Field | Meaning |
| --- | --- |
| `id`, `title`, `operator` | Problem identity and transformation family. |
| `source_statement`, `source_constraints` | Original problem specification. |
| `source_reference_algorithm`, `source_complexity` | Known source solution and expected complexity. |
| `shifted_statement`, `shifted_constraints` | Generated shifted task. |
| `target_algorithm`, `target_complexity` | Expected algorithm and asymptotic requirements after the shift. |
| `old_solution_failure` | Why the source algorithm no longer works. |
| `trap_patterns` | Common invalid reuse patterns. |
| `shifted_examples` | Public examples for the shifted task. |

## Reproducing Paper Analyses

The repository includes JSON artifacts for the main tables and figures:

| Artifact | Description |
| --- | --- |
| `analysis/table_main.json` | Main model comparison on original and shifted tasks. |
| `analysis/table_optt_opts.json` | pass@k, OPTT, OPTS, and gap statistics. |
| `analysis/table_shift.json` | Breakdown by shift operator. |
| `analysis/table_ablation*.json` | Quality-gate ablation results. |
| `analysis/table_checker.json` | Complexity-checker validation. |
| `analysis/fig_consens_data.json` | Constraint magnitude sweep for CONSENS analysis. |
| `analysis/shift_metrics_per_problem.json` | Per-problem shift statistics. |

Figure/table regeneration scripts live in [`scripts/`](scripts/). Some archival plotting scripts write to paper-specific output folders; the main benchmark and evaluation paths are rooted at the repository directory.

## Recommended Evaluation Protocol

For contamination-aware evaluation with AlgoBench:

1. freeze the evaluated model version and decoding parameters;
2. define seed sources and transformation operators before generation;
3. generate more candidate tasks than needed;
4. apply old-solution failure, reference-solution, near-duplicate, and complexity checks;
5. separate public development tasks from private test tasks;
6. evaluate generated code in a sandbox with fixed time and memory limits;
7. report pass@k and complexity-aware metrics separately;
8. release generation metadata after evaluation when possible.

## Citation

```bibtex
@misc{song2026algobenchbenchmarkingalgorithmicadaptation,
  title={AlgoBench: Benchmarking Algorithmic Adaptation in Code Generation},
  author={Xinyuan Song and Zekun Cai and Liang Zhao},
  year={2026},
  eprint={2607.00062},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2607.00062}
}
```

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

Benchmark items may be derived from external programming-problem sources. Users are responsible for checking redistribution and evaluation rights for any added seed dataset.
