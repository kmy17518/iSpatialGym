#!/bin/bash
# Stack probe lanes (2026-10-30): lid teleported onto the teapot's attachment point, N=1, 300 control steps; one variant per "name:args" item.
#   run_probe_stack.sh <lane> <appdata> "name:args" ["name:args" ...]
set -u
LANE=$1; APPDATA=$2; shift 2
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
INST=$HOME/vec_eval85/instances/eval/configuration_matching-attachment-lid-teapot-bar.json
OUT=$HOME/vec_eval85/next/stack
log() { echo "[$(date -Is)] [$LANE] $*"; }
for item in "$@"; do
  name=${item%%:*}; extra=${item#*:}; [ "$extra" = "$item" ] && extra=""
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/probe_stack.py --instances $INST --num-envs 1 --steps 300 --every 30 --stack $extra ) > $OUT/$name.log 2>&1
  rc=$?
  log "$name [$extra] rc=$rc $(grep -c 'PROBE step' $OUT/$name.log) step-lines; $(grep 'PROBE summary' $OUT/$name.log | sed 's/PROBE summary slot 0 //' | tr '\n' ';' | cut -c1-400)"
done
log "LANE DONE"
