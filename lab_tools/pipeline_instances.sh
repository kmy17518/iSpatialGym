#!/bin/bash
# Instance pipeline after the samplers: per task, once its sampler chain has ended (sample.log "end <alias>") and the GPU
# has room (>= 11 GB free), render the first frames of its regenerated instances; when every task has frames, match them
# against the test set (lab/instances.json) and sync lab/. Log: lab_tools/pipeline_instances.log
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
TASKS="camera_relocalization-standard alignment-axial-board_game-breakfast_table configuration_matching-articulation_open_large_scale-dishwasher configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}'; }
samplers() { ps -eo args | grep -c '^python lab_tools/sample_chosen.p[y]'; }   # Isaac Sim sampler processes (a loading one has not taken its memory yet)
done_frames=""
log "pipeline start"
while true; do
  all_done=1
  for t in $TASKS; do
    case " $done_frames " in *" $t "*) continue;; esac
    all_done=0
    if grep -q "end $t rc=" datasets/eval_1003_regen/$t/sample.log 2>/dev/null; then
      n=$(ls datasets/eval_1003_regen/$t/inst*/*/*/instances/*-tro_state.json 2>/dev/null | wc -l)
      if [ "$(samplers)" -gt 1 ] || [ "$(free_mib)" -lt 11000 ]; then log "$t sampled ($n instances); waiting for GPU room ($(samplers) samplers, $(free_mib) MiB free)"; sleep 60; continue; fi
      log "$t: first frames of $n instances"
      python lab_tools/first_frames.py --alias "$t" --inst-root datasets/eval_1003_regen/$t --out datasets/first_frames/$t >> datasets/first_frames.$t.log 2>&1
      log "$t: first frames rc=$? ($(grep -c '" ok' datasets/first_frames/$t/index.json 2>/dev/null) ok entries)"
      done_frames="$done_frames $t"
    fi
  done
  if [ "$all_done" = 1 ]; then break; fi
  sleep 60
done
log "matching"
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json >> "$LOG" 2>&1
log "match rc=$?"
act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "pipeline end"
