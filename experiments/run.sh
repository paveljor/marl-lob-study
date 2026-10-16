#!/usr/bin/env bash
# Generate and run an experiment suite in parallel.
#
#   experiments/run.sh smoke          # ~2 min, proves the pipeline works
#   experiments/run.sh cheap          # the four cheap experiments
#   experiments/run.sh grid -j 8      # the 2D spine grid
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUITE="${1:-}"
[ -n "$SUITE" ] || { echo "usage: experiments/run.sh <smoke|cheap|grid> [-j N]" >&2; exit 1; }
shift || true

# Default concurrency: leave 2 cores for the OS, cap at 8. Each ABIDES sim
# spawns 2000+ background agents, so this is RAM-bound as much as CPU-bound -
# on a machine with little memory, pass -j 1.
# One thread per job. torch defaults to one thread per core, so N concurrent
# jobs ask for N*ncpu threads and the machine thrashes. Measured on an 8-core
# i7-9700K at -j8: 193 threads on 8 cores, load ~52, and a v10_n1 cell took
# 5824s. With these set it takes 212s - a 27x difference, entirely context
# switching. The policy is a 64x64 MLP and the bottleneck is the
# single-threaded ABIDES kernel, so intra-op parallelism buys nothing here;
# the win is running many sims at once, which is what -j does.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NOT nproc: GNU coreutils nproc honours OMP_NUM_THREADS, which is exported to 1
# a few lines above, so nproc would report 1 core and the default concurrency
# would collapse to -j 1 on every Linux box. getconf reports the real count.
# macOS never saw this because sysctl succeeds there and short-circuits.
NCPU="$( (sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2) )"
JOBS=$(( NCPU - 2 )); [ "$JOBS" -lt 1 ] && JOBS=1; [ "$JOBS" -gt 8 ] && JOBS=8
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2 ;;
    *)  echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Prefer the bootstrap venv; otherwise accept whatever environment is already
# active (e.g. a conda env on a machine that was set up by hand), as long as the
# imports actually resolve.
if [ -d .work/venv ]; then
  # shellcheck disable=SC1091
  . .work/venv/bin/activate
elif python -c "import marl_lob, abides_core, stable_baselines3" 2>/dev/null; then
  echo "note: no .work/venv - using the active environment ($(command -v python))"
else
  echo "ERROR: no usable environment. Run experiments/bootstrap.sh first, or" >&2
  echo "       activate an env that already has marl_lob + abides_core + sb3." >&2
  exit 1
fi

# bootstrap.sh normally records this; create one if we are running without it.
mkdir -p .work
if [ ! -s .work/results_dir ]; then
  RD="$REPO_ROOT/results/$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RD"
  echo "$RD" > .work/results_dir
fi

python experiments/matrix.py "$SUITE"
JOBDIR="experiments/jobs/$SUITE"
COUNT="$(ls "$JOBDIR"/*.sh 2>/dev/null | wc -l | tr -d ' ')"
RESULTS="$(cat .work/results_dir)"

echo
echo "suite:       $SUITE  ($COUNT jobs)"
echo "concurrency: -j $JOBS  (of $NCPU cores)"
echo "logs:        $RESULTS/logs"
echo

START=$(date +%s)
set +e
ls "$JOBDIR"/*.sh | xargs -P "$JOBS" -n1 experiments/_runjob.sh
RC=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

echo
printf 'suite %s finished in %dm%ds\n' "$SUITE" $(( ELAPSED / 60 )) $(( ELAPSED % 60 ))
[ "$RC" -ne 0 ] && echo "NOTE: at least one job failed - grep FAIL above, logs in $RESULTS/logs"
echo "collect results with: python experiments/collect.py $SUITE"
exit 0
