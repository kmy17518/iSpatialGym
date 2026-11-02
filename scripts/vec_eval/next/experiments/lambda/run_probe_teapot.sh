#!/bin/bash
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
INST=$HOME/vec_eval85/instances/eval/configuration_matching-attachment-lid-teapot-bar.json
OUT=$HOME/vec_eval85/next/probe
for thr in 0.005 0.5; do
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/probe_sleep.py --instances $INST --num-envs 1 --steps 240 --every 30 --sleep-threshold $thr ) > $OUT/teapot_thr$thr.log 2>&1
  echo "[$(date -Is)] thr=$thr rc=$?"
done
echo "TEAPOT PROBE DONE"
