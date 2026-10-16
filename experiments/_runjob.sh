#!/usr/bin/env bash
# Run one generated job script, logging to the results dir. Invoked by run.sh
# via xargs; not meant to be called directly.
set -uo pipefail
JOB="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$(cat "$REPO_ROOT/.work/results_dir" 2>/dev/null || echo "$REPO_ROOT/results/unknown")"
LOGDIR="$RESULTS/logs"
mkdir -p "$LOGDIR"
NAME="$(basename "$JOB" .sh)"
LOG="$LOGDIR/$NAME.log"

START=$(date +%s)
if bash "$JOB" >"$LOG" 2>&1; then
  printf 'ok    %-44s %4ss\n' "$NAME" "$(( $(date +%s) - START ))"
else
  printf 'FAIL  %-44s %4ss  -> %s\n' "$NAME" "$(( $(date +%s) - START ))" "$LOG"
  exit 1
fi
