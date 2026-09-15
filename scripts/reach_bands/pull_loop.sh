#!/bin/bash
# Periodically rsync /root/samp of every live rb pod into runlogs/runpod/rb/<pod name>/ so that nothing is lost when a
# pod hits its TTL with jobs still queued (collect.py only pulls after JOB_DONE, i.e. after --close + an empty queue).
#   nohup scripts/reach_bands/pull_loop.sh runlogs/runpod/rb/pods.txt 600 > runlogs/runpod/rb/pull_loop.log 2>&1 &
set -uo pipefail
pods_file=$1; interval=${2:-600}
cd "$(dirname "$0")/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ispatialgym
while true; do
  for p in $(cat "$pods_file"); do
    name=$(python3 -c "import json; print(json.load(open('runlogs/runpod/fleet.json'))['pods'].get('$p', {}).get('name', '$p'))")
    term=$(python3 -c "import json; print(json.load(open('runlogs/runpod/fleet.json'))['pods'].get('$p', {}).get('terminated_at') or '')")
    [ -n "$term" ] && continue
    mkdir -p "runlogs/runpod/rb/$name"
    if timeout 600 python scripts/runpod/rp.py pull "$p" /root/samp/ "runlogs/runpod/rb/$name/" > /dev/null 2>&1; then
      echo "$(date -u +%FT%TZ) pulled $name ($(ls runlogs/runpod/rb/$name/*/*.npz 2>/dev/null | wc -l) band files)"
    else
      echo "$(date -u +%FT%TZ) pull failed for $name"
    fi
  done
  sleep "$interval"
done
