#!/bin/bash
# After the GPU queue and the relocalization recovery: first frames of the recovered relocalization candidates, the final
# episode -> instance match, lab sync. Log: lab_tools/pipeline_instances.log
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
RELOC=camera_relocalization-standard
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}'; }
isaac_procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2 > 3000 {n++} END {print n+0}'; }
wait_room() { while [ "$(isaac_procs)" -ge 2 ] || [ "$(free_mib)" -lt 10000 ]; do sleep 30; done; sleep 20; }
log "finish_instances: waiting for the recovery and the GPU queue"
RPID=$(cat datasets/eval_1003_regen/$RELOC/recover.pid 2>/dev/null)
while [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; do sleep 30; done
log "finish_instances: recovery done ($(grep -c '"written": true' datasets/eval_1003_regen/$RELOC/recover_progress.jsonl) exact recoveries)"
until grep -q "gpu_queue end" "$LOG"; do sleep 60; done
wait_room
log "$RELOC: first frames (+depth) of the recovered candidates"
python lab_tools/first_frames.py --alias $RELOC --inst-root datasets/eval_1003_regen/$RELOC --out datasets/first_frames/$RELOC --redo-missing-depth >> datasets/first_frames.$RELOC.log 2>&1
log "$RELOC: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames/$RELOC/index.json) ok)"
log "finish_instances: final match"
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json >> "$LOG" 2>&1
log "match rc=$?"
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "finish_instances end"
