#!/bin/bash
# ConstraintShift Pipeline Master Script
# Usage: bash run_pipeline.sh [OPENAI_API_KEY]

set -e
PYTHON="python3"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$BASE/logs/pipeline.log"
MILESTONE="$BASE/experiments/_milestones.md"

mkdir -p "$BASE/logs" "$BASE/experiments"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
mlog() { echo "[$(ts)] $1" | tee -a "$LOG"; }
milestone() { echo "$(ts) | $1 | $2" >> "$MILESTONE"; }

if [ -n "${1:-}" ]; then
    export OPENAI_API_KEY="$1"
fi

mlog "=== ConstraintShift Pipeline Start ==="
mlog "Python: $($PYTHON --version 2>&1)"
mlog "OpenAI key set: $([ -n "$OPENAI_API_KEY" ] && echo YES || echo NO)"

# Stage 1
mlog ">>> Stage 1: Collect seed problems"
$PYTHON "$BASE/src/stage1_collect.py" && milestone "STAGE1_COLLECT" "DONE" || { mlog "Stage 1 FAILED"; exit 1; }

# Stage 2
mlog ">>> Stage 2: Transform + quality gates"
$PYTHON "$BASE/src/stage2_transform.py" && milestone "STAGE2_TRANSFORM" "DONE" || { mlog "Stage 2 FAILED"; exit 1; }

# Check output
N=$(wc -l < "$BASE/data/final_benchmark.jsonl")
mlog "Benchmark: $N problems accepted"

# Stage 4
mlog ">>> Stage 4: LLM Evaluation (GPT-4o mini)"
if [ -z "$OPENAI_API_KEY" ]; then
    mlog "WARNING: OPENAI_API_KEY not set. Skipping Stage 4."
    mlog "Run manually: OPENAI_API_KEY=sk-... $PYTHON \"$BASE/src/stage4_evaluate.py\""
else
    $PYTHON "$BASE/src/stage4_evaluate.py" && milestone "STAGE4_EVALUATE" "DONE" || { mlog "Stage 4 FAILED"; exit 1; }
fi

# Stage 5
if [ -f "$BASE/results/main_results.json" ]; then
    mlog ">>> Stage 5: Analyze results"
    $PYTHON "$BASE/src/stage5_analyze.py" && milestone "STAGE5_ANALYZE" "DONE" || mlog "Stage 5 FAILED"
fi

mlog "=== Pipeline complete ==="
mlog "Results at: $BASE/results/"
ls -lh "$BASE/results/" 2>/dev/null || true
