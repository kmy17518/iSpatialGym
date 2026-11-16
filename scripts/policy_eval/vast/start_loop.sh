#!/bin/bash
# Start (detached) the fleet loop of a policy-eval run: keep GPUS GPUs working while its units are open; log $EV/logs/loop_<run>.log,
# pid in $EV/logs/loop_<run>.pid (stop it with kill $(cat ...pid), never pkill -f: the pattern matches the ssh command too).
#   bash start_loop.sh <run> <gpus> [max $/GPU-h]
cd /vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1
RUN=${1:?run}; GPUS=${2:?gpus}; MAX=${3:-1.0}
nohup env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "source tool/creds.sh >/dev/null 2>&1; export ISG_DRIVER_MIN=580 ISGEVAL_SEARCH_EVERY_S=120 ISGEVAL_SEARCH_LIMIT=300; exec \$VPY -u code/eval_9eec9441/scripts/policy_eval/vast/vastctl_eval.py --run $RUN loop --gpus $GPUS --max $MAX --every 120" >> logs/loop_$RUN.log 2>&1 < /dev/null &
echo $! > logs/loop_$RUN.pid
echo "loop $RUN pid $(cat logs/loop_$RUN.pid) gpus $GPUS max $MAX"
