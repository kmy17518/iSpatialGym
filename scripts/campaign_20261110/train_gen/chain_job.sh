#!/bin/bash
# One Slurm job = a chain of train worker sources run one after another (round plan packs ~5 h per chain).
#   sbatch ... chain_job.sh <chain_dir>      (<chain_dir>/items.txt: one source job dir per line)
# Each item: its own run.sh (dispatch-time gate inside; GATE_SKIP exit 3), in its own process group, watched by
# giveup_watch.py (sub-template give-up + hard per-item limit). An item that cannot finish inside the remaining Slurm
# time is not started (status not_started_time) so the dispatcher re-queues it. Items already run are skipped.
set -uo pipefail
CHAIN=$(cd "$1" && pwd)
TOOLS=/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/tools
STATUS=$CHAIN/status.jsonl
END=${SLURM_JOB_END_TIME:-$(( $(date +%s) + 18000 ))}
note() { echo "{\"at\": $(date +%s), \"job\": \"${SLURM_JOB_ID:-local}\", $1}" >> $STATUS; }
note "\"event\": \"chain_start\", \"host\": \"$(hostname -s)\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\""
while read -r SRC; do
  [ -z "$SRC" ] && continue
  if [ -e "$SRC/online/state.json" ] || [ -e "$SRC/chain_claim" ]; then note "\"event\": \"item_skip_already_run\", \"source\": \"$SRC\""; continue; fi
  SECS=$(python3 -c "import json;print(json.load(open('$SRC/source.json'))['seconds'])")
  if [ $(( $(date +%s) + SECS + 120 )) -gt $END ]; then note "\"event\": \"item_not_started_time\", \"source\": \"$SRC\", \"seconds\": $SECS"; continue; fi
  if ! mkdir "$SRC/chain_claim" 2>/dev/null; then note "\"event\": \"item_claimed_elsewhere\", \"source\": \"$SRC\""; continue; fi
  echo "${SLURM_JOB_ID:-local} $(hostname -s) $(date +%s)" > "$SRC/chain_claim/owner"
  T0=$(date +%s)
  note "\"event\": \"item_start\", \"source\": \"$SRC\", \"seconds\": $SECS"
  ( unset SLURM_JOB_END_TIME; exec setsid bash "$SRC/run.sh" > "$SRC/slurm_chain_${SLURM_JOB_ID:-local}.log" 2>&1 ) &
  RPID=$!
  sleep 2
  PGID=$(ps -o pgid= -p $RPID 2>/dev/null | tr -d ' ')
  [ -z "$PGID" ] && PGID=$RPID
  python3 $TOOLS/giveup_watch.py --source "$SRC" --pgid "$PGID" > "$SRC/giveup_watch.log" 2>&1 &
  WPID=$!
  wait $RPID; RC=$?
  wait $WPID; WRC=$?
  note "\"event\": \"item_end\", \"source\": \"$SRC\", \"rc\": $RC, \"watch_rc\": $WRC, \"wall_s\": $(( $(date +%s) - T0 ))"
done < "$CHAIN/items.txt"
note "\"event\": \"chain_end\""
