#!/bin/bash
# Run reach-band jobs of a manifest on the local (shared) GPU, one process at a time, in the order given.
#   scripts/reach_bands/run_local.sh <manifest.json> <out root> <job id> [<job id> ...]
# Waits until the GPU has < 12000 MiB in use (polls every 60 s) before every job; appends "<id> rc=<rc> wall=<s>" to
# <out root>/done.txt; the job's files go to <out root>/<id>/ (same layout as the pod runner, so collect_bands.py can
# read it with --extra <id>=<dir>).
set -uo pipefail
manifest=$1; root=$2; shift 2
cd "$(dirname "$0")/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ISPATIALGYM_LOG_LEVEL=INFO
mkdir -p "$root"
for jid in "$@"; do
  args=$(python3 - "$manifest" "$jid" <<'EOF'
import json, shlex, sys
jobs = {j["id"]: j for j in json.load(open(sys.argv[1]))}
print(" ".join(shlex.quote(a) for a in jobs[sys.argv[2]]["args"]))
EOF
)
  [ -z "$args" ] && { echo "unknown job $jid"; continue; }
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 12000 ] && break
    sleep 60
  done
  out="$root/$jid"; mkdir -p "$out"
  echo "$(date -u +%H:%M:%S) start $jid" >> "$root/runner.log"
  t0=$(date +%s)
  eval OMNIGIBSON_HEADLESS=1 timeout 3000 python -m ispatialgym.sampling.reach_bands $args --out "$out" > "$out/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "$jid rc=$rc wall=$((t1-t0)) n_written=-1" >> "$root/done.txt"
  echo "$(date -u +%H:%M:%S) done $jid rc=$rc wall=$((t1-t0))" >> "$root/runner.log"
done
echo "LOCAL_QUEUE_DONE" >> "$root/runner.log"
