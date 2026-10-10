#!/bin/bash
# First frames (RGB + depth) of every regenerated instance / candidate, then the episode -> instance matching and a lab sync.
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
samplers() { ps -eo args | grep -c '^python lab_tools/sample_chosen.p[y]'; }
frames() {  # frames <alias>
  while [ "$(samplers)" -gt 1 ]; do sleep 60; done
  log "$1: first frames (+depth)"
  python lab_tools/first_frames.py --alias "$1" --inst-root datasets/eval_1003_regen/$1 --out datasets/first_frames/$1 --redo-missing-depth >> datasets/first_frames.$1.log 2>&1
  log "$1: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames/$1/index.json) ok)"
}
log "pipeline2 start"
frames alignment-axial-board_game-breakfast_table
frames camera_relocalization-standard
frames configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table
D=configuration_matching-articulation_open_large_scale-dishwasher
until grep -q "end $D rc=" datasets/eval_1003_regen/$D/sample.log 2>/dev/null; do sleep 30; done
frames $D
log "matching"
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json >> "$LOG" 2>&1
log "match rc=$?"
act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "pipeline2 end"
