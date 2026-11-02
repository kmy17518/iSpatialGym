#!/bin/bash
# Restore-phase probe (2026-10-31): where does the N=8 step-0 state depart from N=1? One alias, N=1 then N=8 same-instance, exit after the restore.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
TASK=${1:?alias}
INST=$HOME/vec_eval85/instances/eval/$TASK.json
VOUT=$HOME/vec_eval85/next/restore_probe${PROBE_SUFFIX:-}/$TASK; mkdir -p $VOUT
log() { echo "[$(date -Is)] $*"; }
run() { local name=$1; shift; rm -rf $VOUT/$name $VOUT/$name.jsonl; local t=$SECONDS
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85 ISG_RESTORE_PROBE=$VOUT/$name.jsonl ISG_RESTORE_PROBE_EXIT=1
    source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 --tag vec_next:restore_probe:$TASK "$@" --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  log "$name rc=$? seconds=$((SECONDS - t)) phases=$(wc -l < $VOUT/$name.jsonl 2>/dev/null)"; }
run n1 --num-envs 1
run n8same --num-envs 8 --same-instance-all-slots
log "RESTORE PROBE DONE"
