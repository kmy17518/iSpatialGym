#!/bin/bash
# Restore-phase probe variants (2026-10-31): name:num_envs:extra args; slot 0 vs the N=1 of the same variant.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
TASK=${1:?alias}; shift
INST=$HOME/vec_eval85/instances/eval/$TASK.json
VOUT=$HOME/vec_eval85/next/restore_probe_fixed/$TASK; mkdir -p $VOUT
log() { echo "[$(date -Is)] $*"; }
for spec in "$@"; do
  name=${spec%%:*}; rest=${spec#*:}; n=${rest%%:*}; extra=${rest#*:}; [ "$extra" = "$rest" ] && extra=""
  rm -rf $VOUT/$name $VOUT/$name.jsonl; t=$SECONDS
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85 ISG_RESTORE_PROBE=$VOUT/$name.jsonl ISG_RESTORE_PROBE_EXIT=1
    source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 --tag vec_next:restore_probe:$TASK --num-envs $n $extra --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  log "$name rc=$? seconds=$((SECONDS - t)) phases=$(wc -l < $VOUT/$name.jsonl 2>/dev/null) $(grep -o 'physics_overrides[^}]*}' $VOUT/$name.log | head -1 | cut -c1-160)"
done
log "VARIANT PROBE DONE"
