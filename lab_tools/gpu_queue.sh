#!/bin/bash
# Instance-recovery GPU queue, two slots (at most two Isaac Sim processes next to the ACT server):
#   slot A: relocalization recovery (per-attempt RNG replay + goal-image match) -> its first frames
#   slot B: board-game lenient candidates (every index) -> dishwasher lenient candidates (suspect indices) -> their first frames
# then the episode -> instance match and a lab sync. Every Isaac launch waits for GPU room (>= 10 GB free, < 2 Isaac processes).
# Log: lab_tools/pipeline_instances.log
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
MAN=datasets/isg-init-eval/isg_meta/isg-init-eval.manifest.json
BOARD=alignment-axial-board_game-breakfast_table
RELOC=camera_relocalization-standard
DISH=configuration_matching-articulation_open_large_scale-dishwasher
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}'; }
isaac_procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2 > 3000 {n++} END {print n+0}'; }
wait_room() { while [ "$(isaac_procs)" -ge 2 ] || [ "$(free_mib)" -lt 10000 ]; do sleep 30; done; sleep 20; }
frames() { wait_room; log "$1: first frames (+depth)"; python lab_tools/first_frames.py --alias "$1" --inst-root datasets/eval_1003_regen/$1 --out datasets/first_frames/$1 --redo-missing-depth >> datasets/first_frames.$1.log 2>&1; log "$1: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames/$1/index.json) ok)"; }
lenient_all() { wait_room; log "$1: lenient candidates ($2)"; python lab_tools/sample_chosen.py --alias "$1" --manifest $MAN --out datasets/eval_1003_regen/$1 --seed 1003 --max-attempts 8 --lenient-candidates --lenient-all ${3:+--only $3} >> datasets/eval_1003_regen/$1/sample_lenient_all.log 2>&1; log "$1: lenient rc=$? ($(ls -d datasets/eval_1003_regen/$1/candidates/*/a* 2>/dev/null | wc -l) candidates)"; }

slot_a() {
  wait_room
  log "slot A: relocalization recovery"
  python lab_tools/reloc_recover.py --manifest $MAN --dataset datasets/isg-init-eval --out datasets/eval_1003_regen/$RELOC >> datasets/eval_1003_regen/$RELOC/recover.log 2>&1
  log "slot A: recovery rc=$? ($(grep -c '"written": true' datasets/eval_1003_regen/$RELOC/recover_progress.jsonl) recovered)"
  frames $RELOC
}
slot_b() {
  sleep 90  # let slot A take its memory first
  lenient_all $BOARD "every index"
  lenient_all $DISH "suspect indices" "$(cat lab_tools/dishwasher_suspect.txt)"
  frames $BOARD
  frames $DISH
}
log "gpu_queue start"
slot_a &
PA=$!
slot_b &
PB=$!
wait $PA $PB
log "gpu_queue: final match"
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json >> "$LOG" 2>&1
log "match rc=$?"
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "gpu_queue end"
