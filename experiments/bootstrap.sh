#!/usr/bin/env bash
# Set up a local environment for the current project and legacy suites.
#
# Environment and simulator files live under .work/; results live under results/.
# No system-wide packages are installed. Pip may use its normal shared cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO_ROOT/.work"
VENV="$WORK/venv"
ABIDES="$WORK/abides-jpmc-public"
ABIDES_PIN="f9cbe51342b7dedd9587e4e069040d68a5c6477f"
PATCH="$REPO_ROOT/patches/abides-jpmc-public-local.patch"
CONSTRAINTS="$REPO_ROOT/requirements/linux-py311-cpu.txt"

echo "==> repo:  $REPO_ROOT"
echo "==> work:  $WORK  (environment and simulator; results are separate)"

# --- python check -------------------------------------------------------------
PY="${PYTHON:-python3}"
PYV="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PYV" in
  3.11) : ;;
  *) echo "ERROR: tested setup requires Python 3.11, found $PYV ($PY). Set PYTHON=/path/to/python3.11" >&2
     exit 1 ;;
esac
echo "==> python $PYV ($("$PY" -c 'import sys; print(sys.executable)'))"

mkdir -p "$WORK"

# --- venv ---------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "==> creating venv"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install -q -U pip wheel

# --- ABIDES: clone at the pinned commit, then apply our patch -----------------
if [ ! -d "$ABIDES/.git" ]; then
  echo "==> cloning ABIDES"
  git clone -q https://github.com/jpmorganchase/abides-jpmc-public.git "$ABIDES"
fi
git -C "$ABIDES" fetch -q origin
git -C "$ABIDES" checkout -q "$ABIDES_PIN"

# The patch is required, not optional: without it str_to_ns is 1000x wrong and
# a "1 hour" simulation actually simulates 3.6 seconds. Results would be
# silently wrong rather than obviously broken.
if git -C "$ABIDES" apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "==> ABIDES patch already applied"
else
  echo "==> applying ABIDES patch"
  git -C "$ABIDES" apply "$PATCH"
fi
git -C "$ABIDES" apply --check --reverse "$PATCH" \
  || { echo "ERROR: ABIDES patch is not applied; refusing to continue" >&2; exit 1; }

# --- install ------------------------------------------------------------------
# abides-core / abides-markets declare no install_requires, so they pull nothing.
# ABIDES' top-level requirements.txt pins gym / ray / pomegranate which this
# project does not need and which no longer build cleanly - deliberately unused.
echo "==> installing abides-core + abides-markets (editable)"
python -m pip install -q -e "$ABIDES/abides-core" -e "$ABIDES/abides-markets"
echo "==> installing tested CPU PyTorch and simulator dependencies"
python -m pip install -q -c "$CONSTRAINTS" torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q -c "$CONSTRAINTS" coloredlogs psutil scipy tqdm
echo "==> installing this project + dev extras"
python -m pip install -q -c "$CONSTRAINTS" -e "$REPO_ROOT[dev]"
python -m pip check

# --- provenance ---------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="$REPO_ROOT/results/$(hostname -s)-$STAMP"
mkdir -p "$RESULTS"
python -m pip freeze > "$RESULTS/pip-freeze.txt"
{
  echo "host:        $(hostname -s)"
  echo "uname:       $(uname -a)"
  echo "python:      $(python -V 2>&1)"
  echo "repo_sha:    $(git -C "$REPO_ROOT" rev-parse --verify HEAD 2>/dev/null || echo uncommitted-source)"
  echo "abides_pin:  $ABIDES_PIN"
  echo "bootstrapped: $STAMP"
  if command -v sysctl >/dev/null 2>&1; then
    echo "cpu:         $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
    echo "ncpu:        $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
    echo "mem_bytes:   $(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
  fi
} > "$RESULTS/machine.txt"
echo "$RESULTS" > "$WORK/results_dir"

# --- verify -------------------------------------------------------------------
echo "==> verifying with the fast test suite"
( cd "$REPO_ROOT" && python -m pytest -q -m "not abides" ) | tail -3
echo "==> verifying ABIDES integration tests"
( cd "$REPO_ROOT" && python -m pytest -q -m abides ) | tail -3

cat <<EOF

bootstrap complete.

  activate:  . .work/venv/bin/activate
  results:   $RESULTS
  current:   see README.md for concurrent independent-PPO training/evaluation
  legacy:    experiments/run.sh smoke     # historical shared-policy suite
             see experiments/README.md before using cheap/grid suites
  cleanup:   experiments/teardown.sh
EOF
