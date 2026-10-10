#!/bin/bash
# Training episodes -> instances (relocalization phase, deliverable 3): recover the camera_relocalization-standard training
# instances of kmy17518/isg-init from the sampler's RNG (lab_tools/reloc_recover.py: CSV init->goal distance match over the
# candidate sweep seeds, init / goal view renders decide), render their first frames with the collector's sensor set
# (lab_tools/first_frames.py), verify them against the episodes' first frames like the test set (lab_tools/match_instances.py)
# and publish lab/instances-train.json. Idempotent: recovered indices are skipped.
#
#   lab_tools/train_instances.sh "<seeds>" <episode ...>        e.g. lab_tools/train_instances.sh "0 1 2" 8 12 15 19 28
#
# Log: lab_tools/train_instances.log. Needs ~10 GB of free GPU memory for each Isaac Sim step (waits for it).
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
SEEDS=$1; shift
EPISODES="$*"
LOG=lab_tools/train_instances.log
RELOC=camera_relocalization-standard
JOB=datasets/train_regen
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{print $2-$1}'; }
wait_room() { while [ "$(free_mib)" -lt 10000 ]; do sleep 30; done; }
mkdir -p "$JOB"
log "train_instances: seeds [$SEEDS] episodes [$EPISODES]"
wait_room
python lab_tools/reloc_recover.py --dataset datasets/isg-init --episodes $EPISODES --seeds $SEEDS --out "$JOB" >> "$JOB/recover.log" 2>&1
log "recover rc=$? ($(grep -c '"written": true' "$JOB/recover_progress.jsonl" 2>/dev/null) recovered so far)"
wait_room
python lab_tools/first_frames.py --alias $RELOC --inst-root "$JOB" --out datasets/first_frames_train/$RELOC --redo-missing-depth >> datasets/first_frames_train.$RELOC.log 2>&1
log "first frames rc=$? ($(python -c "import json; d=json.load(open('datasets/first_frames_train/$RELOC/index.json')); print(sum(1 for v in d.values() if v.get('ok')))" 2>/dev/null) ok)"
python lab_tools/match_instances.py --dataset datasets/isg-init --frames datasets/first_frames_train --out lab/instances-train.json --tasks $RELOC \
    --method-note "+reloc_recover(train sweep seed from the CSV distance match)" >> "$LOG" 2>&1
log "match rc=$?"
# goal-side check: the recovered instance's goal trunk (translate, tilt) -> torso joints vs the demo's final observation.state trunk_qpos
python - <<'EOF' >> "$LOG" 2>&1
import json, sys, numpy as np
sys.path.insert(0, ".")
from lab_tools.replay_episodes import load_episode_arrays
from ispatialgym.robot.conventions import torso_q
p = "lab/instances-train.json"
d = json.load(open(p))
for e, row in d["episodes"].items():
    if not row.get("instance") or row.get("goal_trunk_max_abs_dq") is not None:
        continue
    try:
        sc = json.load(open(row["tro"].replace("-tro_state.json", ".oracle.json")))
        g = sc["stances"]["goal"]
        _, st, _ = load_episode_arrays("datasets/isg-init", int(e))
        dq = float(np.abs(torso_q(g["translate"], g["tilt"]).numpy() - st[-1, 53:57]).max())
        row["goal_trunk_max_abs_dq"] = round(dq, 4)
        row["goal_verified_by_trunk"] = bool(dq < 0.02)
        row["recovered_sweep_seed"] = (json.load(open(row["tro"].replace("-tro_state.json", ".summary.json"))).get("recovered") or {}).get("base_seed")
    except Exception as ex:  # noqa: BLE001
        row["goal_trunk_check_error"] = repr(ex)
json.dump(d, open(p, "w"), indent=1)
print("goal trunk check:", {e: (r.get("goal_trunk_max_abs_dq"), r.get("recovered_sweep_seed")) for e, r in d["episodes"].items() if r.get("instance")})
EOF
env -u HF_HUB_OFFLINE act-isg/.venv/bin/python - <<'EOF' >> "$LOG" 2>&1
import json, sys
sys.path.insert(0, ".")
from lab_tools import lab_sync as LS
d = json.load(open("lab/instances-train.json"))
eps = d["episodes"]
st = LS.read_state()
st["instances_train_verified"] = f"{d['verified']}/{d['total']} (relocalization train episodes with a recovered, first-frame verified instance)"
LS.write_state(st)
LS.status()
print("instances_train_verified", st["instances_train_verified"])
EOF
log "train_instances end"
