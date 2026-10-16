#!/usr/bin/env bash
# Remove everything bootstrap.sh created, keeping results/.
#
# Deliberately conservative: it only ever deletes paths inside this clone, and
# prints exactly what it will remove before doing it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ASSUME_YES=0
[ "${1:-}" = "-y" ] && ASSUME_YES=1

TARGETS=".work runs experiments/jobs"

echo "clone: $REPO_ROOT"
echo
echo "will remove (results/ is kept):"
FOUND=0
for t in $TARGETS; do
  if [ -e "$t" ]; then
    printf '  %-20s %s\n' "$t" "$(du -sh "$t" 2>/dev/null | cut -f1)"
    FOUND=1
  fi
done
[ "$FOUND" -eq 0 ] && { echo "  (nothing to remove)"; }
echo
echo "will keep:"
[ -d results ] && printf '  %-20s %s\n' "results" "$(du -sh results | cut -f1)" \
               || echo "  (no results/ - nothing was run?)"

if [ "$ASSUME_YES" -ne 1 ]; then
  printf '\nproceed? [y/N] '
  read -r ans
  case "$ans" in y|Y) ;; *) echo "aborted"; exit 0 ;; esac
fi

for t in $TARGETS; do
  [ -e "$t" ] && rm -rf "$t" && echo "removed $t"
done

echo
echo "done. Nothing outside this clone was touched - no system packages, no"
echo "conda, no files in \$HOME. To remove the clone itself as well:"
echo
echo "    rm -rf \"$REPO_ROOT\""
echo
if [ -d results ]; then
  echo "Copy results off first if you want to keep them:"
  echo
  echo "    tar czf ~/marl-lob-results.tgz -C \"$REPO_ROOT\" results"
fi
