#!/bin/bash
# Dishwasher per-attempt candidates for the unconfirmed indices, two Isaac Sim processes on disjoint halves of the list
# (resumable: indices with a candidates/<index>/ dir are skipped), then their first frames and the final match + sync.
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
MAN=datasets/isg-init-eval/isg_meta/isg-init-eval.manifest.json
DISH=configuration_matching-articulation_open_large_scale-dishwasher
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
ALL=$(cat lab_tools/dishwasher_suspect.txt)
TODO=""
for i in $ALL; do [ -d datasets/eval_1003_regen/$DISH/candidates/$i ] && [ "$(ls -d datasets/eval_1003_regen/$DISH/candidates/$i/a* 2>/dev/null | wc -l)" -ge 8 ] || TODO="$TODO $i"; done
set -- $TODO
N=$#; H=$(( (N + 1) / 2 ))
A="${@:1:$H}"; B="${@:$((H+1))}"
log "dishwasher lenient (parallel): $N indices left; A=[$A] B=[$B]"
run() { python lab_tools/sample_chosen.py --alias $DISH --manifest $MAN --out datasets/eval_1003_regen/$DISH --seed 1003 --max-attempts 8 --lenient-candidates --lenient-all --only $1 >> datasets/eval_1003_regen/$DISH/sample_lenient_$2.log 2>&1; log "dishwasher lenient $2 rc=$?"; }
[ -n "$A" ] && run "$A" A &
PA=$!
sleep 120
[ -n "$B" ] && run "$B" B &
PB=$!
wait $PA $PB
log "dishwasher lenient done ($(ls -d datasets/eval_1003_regen/$DISH/candidates/*/a* 2>/dev/null | wc -l) candidates)"
python lab_tools/first_frames.py --alias $DISH --inst-root datasets/eval_1003_regen/$DISH --out datasets/first_frames/$DISH --redo-missing-depth >> datasets/first_frames.$DISH.log 2>&1
log "$DISH: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames/$DISH/index.json) ok)"
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames --out lab/instances.json >> "$LOG" 2>&1
log "match rc=$?"
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")')\"" >> "$LOG" 2>&1
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "dish_lenient_parallel end"
