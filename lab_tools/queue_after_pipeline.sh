#!/bin/bash
# GPU job queue after pipeline2: board-game lenient candidates for every index (+ their first frames), the relocalization
# recovery's first frames, a dishwasher match-quality check (lenient candidates if the strict pass is not decisive), then the
# final episode -> instance match and a lab sync. One Isaac Sim process at a time next to the running relocalization recovery.
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
frames() { python lab_tools/first_frames.py --alias "$1" --inst-root datasets/eval_1003_regen/$1 --out datasets/first_frames/$1 --redo-missing-depth >> datasets/first_frames.$1.log 2>&1; log "$1: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames/$1/index.json) ok)"; }
lenient_all() { python lab_tools/sample_chosen.py --alias "$1" --manifest $MAN --out datasets/eval_1003_regen/$1 --seed 1003 --max-attempts 8 --lenient-candidates --lenient-all >> datasets/eval_1003_regen/$1/sample_lenient_all.log 2>&1; log "$1: lenient-all rc=$? ($(ls -d datasets/eval_1003_regen/$1/candidates/*/a* 2>/dev/null | wc -l) candidates)"; }
match() { python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out "$1" ${2:+--tasks $2} >> "$LOG" 2>&1; }
decisive() {  # decisive <report.json> <task> -> exit 0 when the depth metric separates every episode (relative margin > 0.5 and every episode's best differs)
  python - "$1" "$2" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))[sys.argv[2]]
sep = (r.get("metric_separability") or {}).get("depth_mismatch_frac(>5cm)")
ok = sep is not None and sep > 0.5 and (r.get("assignment_min_margin") or 0) > 0 and not r.get("unmapped_episodes")
print("decisive" if ok else f"not decisive (sep={sep}, min margin={r.get('assignment_min_margin')})")
sys.exit(0 if ok else 1)
EOF
}
log "queue start"
until grep -q "pipeline2 end" "$LOG"; do sleep 60; done
log "queue: board game lenient candidates for every index"
lenient_all $BOARD
frames $BOARD
RPID=$(cat datasets/eval_1003_regen/$RELOC/recover.pid 2>/dev/null)
while [ -n "$RPID" ] && kill -0 "$RPID" 2>/dev/null; do sleep 30; done
log "queue: relocalization recovery finished ($(grep -c '"written": true' datasets/eval_1003_regen/$RELOC/recover_progress.jsonl) recovered)"
frames $RELOC
mkdir -p /tmp/v20eval-match
match /tmp/v20eval-match/dish_check.json $DISH
if decisive /tmp/v20eval-match/instances_report.json $DISH >> "$LOG" 2>&1; then
  log "queue: dishwasher strict match decisive"
else
  log "queue: dishwasher lenient candidates for every index"
  lenient_all $DISH
  frames $DISH
fi
log "queue: final match"
match lab/instances.json
log "match rc=$?"
act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "queue end"
