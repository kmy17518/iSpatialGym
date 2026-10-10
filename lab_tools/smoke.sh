#!/bin/bash
# Smoke test (plan §4 / task §5): 5 episodes per checkpoint (3 relocalization + 2 board game) for w1-early-copy05-lrg1e-3-s0 and
# w1-late-tagzero-lrg1e-4-s0, a goal-swap plumbing check (same world), then a two-evaluator memory test. Results under
# lab/smoke/<run>/, videos under runlogs/tierB/smoke/<run>/. Log: lab_tools/smoke.log
set -u
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/smoke.log
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
PY=python
ACTPY=act-isg/.venv/bin/python
Q=/home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized/datasets/isg-goal-coordination/m1/tierB-queue.json
RELOC=camera_relocalization-standard
BOARD=alignment-axial-board_game-breakfast_table
DISH=configuration_matching-articulation_open_large_scale-dishwasher
BLEND=configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table

pick() {  # pick <task> <n> -> first n verified episode indices of the task
  $PY - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open("lab/instances.json")); eps = d.get("episodes", d)
task, n = sys.argv[1], int(sys.argv[2])
rows = sorted((int(k), v) for k, v in eps.items())
sel = [e for e, v in rows if v.get("task") == task and v.get("verified")][:n]
print(" ".join(map(str, sel)))
EOF
}
entry() {  # entry <run> <field>
  $PY -c "import json,sys; q=json.load(open('$Q')); e=[x for x in q['entries'] if x['run']=='$1'][0]; print(e['$2'])"
}
start_server() {  # start_server <run>
  local run=$1 ckpt="act-isg/ckpts/$1/$(entry $1 file)"
  local pid; for pid in $(pgrep -f "serve_b1k.py --model-path"); do kill $pid; done; sleep 3
  mkdir -p runlogs/tierB/server
  (cd act-isg && CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python scripts/b1k/serve_b1k.py --model-path "ckpts/$run/$(entry $run file)" --port 8000 --device cuda >> "../runlogs/tierB/server/$run.log" 2>&1 & echo "$! $run" > ../runlogs/tierB/server/current.pid)
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/healthz >/dev/null && break; sleep 2; done
  log "server for $run up (pid $(cut -d' ' -f1 runlogs/tierB/server/current.pid))"
}
evaluate() {  # evaluate <run> <task> <episodes...> ; extra args via EXTRA
  local run=$1 task=$2; shift 2
  $PY lab_tools/eval_episodes.py --run "$run" --step 50000 --ckpt-repo "$(entry $run repo)" --ckpt-file "$(entry $run file)" --ckpt-sha256 "$(entry $run sha256)" \
     --ckpt-path "act-isg/ckpts/$run/$(entry $run file)" --server ws://127.0.0.1:8000 --server-args "serve_b1k.py --model-path ckpts/$run/$(entry $run file) --port 8000 --device cuda" \
     --task "$task" --lab-out "lab/smoke/$run" --raw-out "runlogs/tierB/smoke/$run" --tag smoke --episodes "$@" ${EXTRA:-} >> "runlogs/tierB/smoke/$run.$task.log" 2>&1
  log "evaluate $run $task $* rc=$?"
}

mkdir -p lab/smoke runlogs/tierB/smoke
RUN_A=w1-early-copy05-lrg1e-3-s0
RUN_B=w1-late-tagzero-lrg1e-4-s0
R_EPS=$(pick $RELOC 3); B_EPS=$(pick $BOARD 2); D_EPS=$(pick $DISH 1); L_EPS=$(pick $BLEND 1)
R_OTHER=$(pick $RELOC 5 | awk '{print $5}')
log "smoke: reloc=[$R_EPS] board=[$B_EPS] dish=[$D_EPS] blender=[$L_EPS] goal-swap other=$R_OTHER"

start_server $RUN_A
EXTRA="--goal-swap $(echo $R_EPS | awk '{print $1}') $R_OTHER" evaluate $RUN_A $RELOC $R_EPS
EXTRA="" evaluate $RUN_A $BOARD $B_EPS

start_server $RUN_B
EXTRA="" evaluate $RUN_B $RELOC $R_EPS
EXTRA="" evaluate $RUN_B $BOARD $B_EPS

# two evaluators at once (memory / throughput): dishwasher + blender, one episode each, on checkpoint A. Needs the GPU to itself
# (no other Isaac Sim process next to the server).
isaac_procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2 > 3000 {n++} END {print n+0}'; }
while [ "$(isaac_procs)" -ge 1 ]; do sleep 60; done
log "GPU free for the two-evaluator test ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
start_server $RUN_A
( while true; do echo "$(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> lab_tools/smoke_gpu_mem.log; sleep 10; done ) &
MON=$!
EXTRA="" evaluate $RUN_A $DISH $D_EPS &
P1=$!
sleep 20
EXTRA="" evaluate $RUN_A $BLEND $L_EPS &
P2=$!
wait $P1 $P2
kill $MON 2>/dev/null
log "parallel test done; GPU memory peak (MiB): $(awk '{print $2}' lab_tools/smoke_gpu_mem.log | sort -n | tail -1)"
log "smoke end"
