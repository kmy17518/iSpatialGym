#!/usr/bin/env bash
# Run validate_tables.py for one scene until every job has a result, re-launching after crashes (Isaac Sim segfaults).
# Usage: run_scene.sh <scene> [extra validate_tables.py args...]
set -u
SCENE="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/.." && pwd)"
WORK="${ISG_TABLE_WORK:-/tmp/isg_table_sel}"
PY="${ISG_PY:-$HOME/miniconda3/envs/ispatialgym/bin/python}"
mkdir -p "$WORK/logs"
LOG="$WORK/logs/${SCENE}$( [[ " $* " == *" --under "* ]] && echo _under ).log"
export OMNIGIBSON_HEADLESS=1
# validate against a frozen copy of the sampler (see validate_tables.py, ISG_TABLE_REPO) when one exists
SNAP="${ISG_TABLE_REPO:-$WORK/snapshot}"
if [[ -d "$SNAP/ispatialgym" ]]; then
  export ISG_TABLE_REPO="$SNAP"
  export PYTHONPATH="$SNAP${PYTHONPATH:+:$PYTHONPATH}"
fi
for round in 1 2 3 4 5 6; do
  echo "=== $(date '+%F %T') $SCENE round $round: $*" | tee -a "$LOG"
  "$PY" "$HERE/validate_tables.py" --scene "$SCENE" --out "$OUT" --work "$WORK" --skip-done "$@" >> "$LOG" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') $SCENE round $round exit $rc" | tee -a "$LOG"
  if grep -q "^\[table_selection [0-9:]*\] nothing to do" <(tail -n 200 "$LOG"); then
    echo "=== $SCENE complete" | tee -a "$LOG"
    exit 0
  fi
  # a clean exit that still had work means the last job list was finished in this round; check once more
  if [[ $rc -eq 0 ]]; then
    continue
  fi
  sleep 5
done
echo "=== $SCENE gave up after 6 rounds" | tee -a "$LOG"
exit 1
