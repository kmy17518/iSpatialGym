#!/bin/bash
# Stop a run's fleet loop once its last units have landed (the loop's status line shows 0 claimed and 0 open); detached on svl11:
#   nohup bash stop_loop_when_done.sh <run> > $EV/logs/stopper_<run>.log 2>&1 &
cd /vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1
RUN=${1:?run}
while true; do
  line=$(grep -E "units [0-9]+/[0-9]+ done" logs/loop_$RUN.log | tail -1)
  if echo "$line" | grep -q -E ", 0 claimed, 0 open"; then
    pid=$(cat logs/loop_$RUN.pid)
    echo "$(date -u +%FT%TZ) done: $line"
    kill "$pid" && echo "$(date -u +%FT%TZ) loop $pid stopped"
    exit 0
  fi
  sleep 60
done
