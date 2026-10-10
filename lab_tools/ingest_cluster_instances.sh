#!/bin/bash
# Ingest the cluster's eval_1003 dishwasher instance files once they are copied to
#   datasets/eval_1003_cluster/configuration_matching-articulation_open_large_scale-dishwasher/   (rsync of the cluster's
#   eval_1003/<task>/ tree, instance / template files only). Waits for the files, renders their first frames (separate frames
#   dir), matches them against the test episodes, fills the dishwasher episodes that are not yet verified in lab/instances.json
#   (already verified episodes keep their current instance), syncs lab/. The Tier B loop picks the new episodes up on its next
#   queue read. Log: lab_tools/pipeline_instances.log
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
LOG=lab_tools/pipeline_instances.log
DISH=configuration_matching-articulation_open_large_scale-dishwasher
SRC=datasets/eval_1003_cluster/$DISH
GLOB="$SRC/*/inst*/$DISH/*/instances/*-tro_state.json"
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }
isaac_procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, '$2 > 3000 {n++} END {print n+0}'; }
log "ingest: waiting for cluster instance files under $SRC"
while true; do
  n=$(ls $GLOB 2>/dev/null | wc -l); ok=1
  for t in $(ls $GLOB 2>/dev/null); do [ -f "$(dirname $(dirname $t))/template_meta.json" ] && [ -f "$(dirname $(dirname $t))/template.json" ] || ok=0; done
  [ "$n" -ge 50 ] && [ "$ok" = 1 ] && break
  sleep 60
done
sleep 120  # let the copy settle
log "ingest: $(ls $GLOB | wc -l) cluster TROs found; rendering first frames (waits for a GPU slot)"
while [ "$(isaac_procs)" -ge 2 ]; do sleep 60; done
python lab_tools/first_frames.py --alias $DISH --inst-root /dev/null --tro-glob "$GLOB" --out datasets/first_frames_cluster/$DISH >> datasets/first_frames_cluster.$DISH.log 2>&1
log "ingest: first frames rc=$? ($(grep -c '"ok": true' datasets/first_frames_cluster/$DISH/index.json) ok)"
mkdir -p /tmp/v20eval-match/cluster
python lab_tools/match_instances.py --dataset datasets/isg-init-eval --frames datasets/first_frames_cluster --out /tmp/v20eval-match/cluster/instances.json --tasks $DISH >> "$LOG" 2>&1
python - <<'EOF' >> "$LOG" 2>&1
import json
cur = json.load(open("lab/instances.json")); eps = cur["episodes"]
new = json.load(open("/tmp/v20eval-match/cluster/instances.json"))["episodes"]
filled = 0
for k, v in new.items():
    if eps.get(k, {}).get("verified"):
        continue  # keep the instance every checkpoint has been evaluated on
    if v.get("verified"):
        v = dict(v); v["method"] = "cluster_eval_1003_instance_files+first_frame_verify"; v["source"] = "cluster"
        eps[k] = v; filled += 1
cur["verified"] = sum(1 for v in eps.values() if v.get("verified")); cur["total"] = len(eps)
json.dump(cur, open("lab/instances.json", "w"), indent=1)
print(f"ingest: filled {filled} dishwasher episodes from the cluster files; verified {cur['verified']}/{cur['total']}")
EOF
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py set instances_verified "\"$(python -c 'import json; d=json.load(open("lab/instances.json")); print(f"{d[\"verified\"]}/{d[\"total\"]}")') (dishwasher completed from the cluster's eval_1003 instance files)\"" >> "$LOG" 2>&1
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py note "Dishwasher instances completed from the cluster's eval_1003 instance files (copied by the user); first frames verified the same way (lab/instances.json updated). The Tier B loop evaluates the newly verified episodes on each queued checkpoint." >> "$LOG" 2>&1
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> "$LOG" 2>&1
log "ingest end"
