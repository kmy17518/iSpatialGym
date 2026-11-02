#!/bin/bash
# Round D on lambda (mental_rotation + object_scaling aliases): n1 + n8same under the final code (rigid sleep threshold 0.005 on scene AND task objects, stock solver iterations, --sleep-assist 30,1,3), N=1 parity vs round B's n1
# (stock threshold, local copies in roundB/runs_B), compare + verdict; synced into the cluster campaign folder (runs_D/, verify_D/).
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next/roundD; mkdir -p $VOUT/runs_D $VOUT/verify_D
REFB=$HOME/vec_eval85/next/roundB/runs_B
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
V=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$V/campaigns/vec_next
log() { echo "[$(date -Is)] $*"; }
for TASK in "$@"; do
  INST=$HOME/vec_eval85/instances/eval/$TASK.json; RUNS=$VOUT/runs_D/$TASK; mkdir -p $RUNS
  run() { local name=$1; shift; rm -rf $RUNS/$name; local t=$SECONDS
    ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85
      source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
      python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 ${ISG_ITERS:+--rigid-solver-iterations $ISG_ITERS} --sleep-assist ${ISG_ASSIST:-30,1,3} --tag vec_next:verifyD:$TASK "$@" --out $RUNS/$name ) > $RUNS/$name.log 2>&1
    local rc=$?
    echo "{\"rc\": $rc, \"seconds\": $((SECONDS - t)), \"host\": \"$(hostname -s)\", \"job\": \"lambda\"}" > $RUNS/$name.run.json
    log "$TASK $name rc=$rc seconds=$((SECONDS - t)) $(grep -c RUN_ISG_RESULT $RUNS/$name.log) results; $(grep -o 'world built in [0-9.]*s' $RUNS/$name.log | head -1)"; }
  run n1 --num-envs 1
  cd $NEXT
  if [ -d $REFB/$TASK/n1 ]; then
    PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $RUNS/n1 --ref $REFB/$TASK/n1 > $RUNS/n1/compare_vs_runs_B.txt 2>&1
    PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $RUNS/n1 --ref $REFB/$TASK/n1 > $RUNS/n1/verify_rules_vs_runs_B.log 2>&1
    cp $RUNS/n1/verify.json $RUNS/n1/verify_vs_runs_B.json 2>/dev/null
    log "$TASK N1_PARITY vs runs_B: $($PY -c "import json;d=json.load(open('$RUNS/n1/verify.json'));print('PASS' if d['pass'] else 'FAIL', d['reasons'][:2])" 2>&1 | cut -c1-200)"
  fi
  run n8same --num-envs 8 --same-instance-all-slots
  PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $RUNS/n8same --ref $RUNS/n1 > $RUNS/n8same/compare.txt 2>&1
  PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $RUNS/n8same --ref $RUNS/n1 > $RUNS/n8same/verify_rules.log 2>&1
  ISG_ROOMS_OVERRIDE=$C/inputs/rooms_override.json ISG_EVAL_ROOT=/vision/u/minyeong/projects/ispatialgym-eval-vec-next PYTHONPATH=$NEXT $PY scripts/vec_eval/next/verify_record.py --task $TASK --runs $RUNS --num-envs 8 \
    --out $VOUT/verify_D/$TASK.json --previous $HOME/vec_eval85/next/roundB/mirror_verify/$TASK.json --b1k $NEXT/Behavior-1K --worktree $NEXT > $RUNS/verify_record.log 2>&1
  log "$TASK VERDICT $($PY -c "import json;d=json.load(open('$VOUT/verify_D/$TASK.json'));print('PASS' if d['pass'] else 'FAIL', d['runs']['n8same'].get('reasons',[])[:2], 'n1_parity', d.get('n1_parity'))" 2>&1 | cut -c1-260)"
  rsync -a $RUNS/ minyeong@svl11:$C/runs_D/$TASK/ && rsync -a $VOUT/verify_D/$TASK.json minyeong@svl11:$C/verify_D/$TASK.json && log "$TASK synced"
done
log "ROUNDD_LAMBDA DONE"
