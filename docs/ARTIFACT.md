# Artifact Guide

Operational notes for reproducing `AlgoBench` from the public `algobench` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `assets/`: README and paper-facing visual assets.
- `analysis/`: Post-processing, table, and figure-generation scripts.
- `results/`: Small checked-in placeholders or result documentation; generated result folders remain local.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `bash run_pipeline.sh`
- `python scripts/eval_ablation.py`
- `python scripts/eval_ablation_v2.py`
- `python scripts/eval_checker.py`
- `python scripts/eval_gemini_o20.py`
- `python scripts/recompute_tables.py`
- `python scripts/run_t4_optt_fix.py`

## Figure Assets

- `assets/algobench_intuition.png`
- `assets/algobench_pipeline.png`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
