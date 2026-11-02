#!/bin/bash
# Sleep-assist validation (2026-10-30): the teapot alias, N=1 then N=8 same-instance, with STOCK solver iterations + --sleep-assist 30,1,3 in SLEEP mode with the proactive gripper wake (0.05 m); compare.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
TASK=configuration_matching-attachment-lid-teapot-bar
INST=$HOME/vec_eval85/instances/eval/$TASK.json
VOUT=$HOME/vec_eval85/next/assist/$TASK; mkdir -p $VOUT
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
run() { local name=$1; shift; rm -rf $VOUT/$name; local t=$SECONDS
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85 ISG_SLEEP_TRACE=1 ISG_SLEEP_TRACE_EVERY=100
    source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 --tag vec_next:assist:$TASK "$@" --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  local rc=$?
  log "$name rc=$rc seconds=$((SECONDS - t)) $(grep -c RUN_ISG_RESULT $VOUT/$name.log) results; $(grep -o 'world built in [0-9.]*s' $VOUT/$name.log | head -1); assist sleeps: $(grep -c SLEEP_ASSIST $VOUT/$name.log)"; }
run n1 --num-envs 1
run n8same --num-envs 8 --same-instance-all-slots
cd $NEXT
PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $VOUT/n8same --ref $VOUT/n1 > $VOUT/n8same/compare.txt 2>&1
PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $VOUT/n8same --ref $VOUT/n1 > $VOUT/n8same/verify_rules.log 2>&1
log "VERDICT $($PY -c "import json;d=json.load(open('$VOUT/n8same/verify.json'));print('PASS' if d['pass'] else 'FAIL', 'relaxed', d.get('pass_relaxed'), d['reasons'][:3])" 2>&1 | cut -c1-300)"
log "ASSIST TEST DONE"
