# Artifact Guide

This guide maps the public `algobench` repository to a reviewer-friendly artifact workflow for `AlgoBench`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `data/`: Small fixtures, schemas, manifests, or data-layout notes; large data should stay outside git.
- `assets/`: README and paper-facing visual assets.
- `analysis/`: Post-processing, table, and figure-generation scripts.
- `results/`: Small checked-in placeholders or result documentation; generated result folders remain local.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
```

If a smoke command is not tracked, use the README Quick Start with the smallest available seed, sample, or task count.

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

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

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
