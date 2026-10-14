#!/bin/bash
# One harness episode: run_isg --task <subtask> --instance-indices <instance> --mode policy --policy
# ispatialgym.harness.policy:HarnessPolicy, the instance loaded from the eval dataset, in the eval release's environment
# (scripts/eval_instances/env.sh: the warm Kit app data of the cluster rule, OMNIGIBSON_DATA_PATH = <checkout>/Behavior-1K/datasets).
# The step budget is run_isg's, max(300, 2 x the demo's length), unless harness_config.json sets max_steps.
# Reads <episode dir>/harness_config.json (python -m ispatialgym.harness.run); writes <episode dir>/{job.txt,run.log,code_paths.txt,run/<iid>/...}.
#   ISG_HARNESS_CODE=<checkout> sbatch [opts] scripts/harness/job_episode.sh <episode dir>
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
set -u
EP=$(realpath "$1")
CODE=${ISG_HARNESS_CODE:-/vision/u/minyeong/projects/ispatialgym-eval}
LIMIT=${ISG_HARNESS_TIMEOUT_S:-13800}
KIT_RC=1
ISG_EVAL_REPO=$CODE source $CODE/scripts/eval_instances/env.sh
trap 'kit_appdata_teardown' EXIT
export ISG_HARNESS_CONFIG=$EP/harness_config.json PYTHONDONTWRITEBYTECODE=1
# the conda env's editable installs point at another checkout: refuse to run unless PYTHONPATH wins for every package
python - "$CODE" > $EP/code_paths.txt 2>&1 <<'EOF'
import importlib.util, os, sys
root = os.path.realpath(sys.argv[1])
bad = 0
for name in ("ispatialgym", "omnigibson", "bddl"):
    origin = os.path.realpath(importlib.util.find_spec(name).origin)
    ok = origin.startswith(root + os.sep)
    bad += not ok
    print(f"{name} {origin} {'ok' if ok else 'OUTSIDE ' + root}")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || { echo "CODE_PATH_CHECK_FAILED"; cat $EP/code_paths.txt; exit 4; }
# run_isg's instance and budget arguments, one per line, from the episode's config
mapfile -t ARGS < <(python - "$ISG_HARNESS_CONFIG" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
if not c.get("task") or c.get("instance_index") is None:
    sys.exit("harness_config.json names no task / instance_index: prepare it with python -m ispatialgym.harness.run")
args = ["--task", c["task"], "--instance-indices", str(int(c["instance_index"])), "--settle-eval", "on" if c.get("settle_eval") else "off"]
for key, flag in (("max_steps", "--max-steps"), ("settle_steps", "--settle-steps"), ("instances_root", "--instances-root")):
    if c.get(key) is not None:
        args += [flag, str(c[key])]
print("\n".join(args))
EOF
)
[ ${#ARGS[@]} -gt 0 ] || { echo "BAD_HARNESS_CONFIG"; exit 3; }
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "ISG_HARNESS_EPISODE ${SLURM_JOB_ID:-local} ep=$EP host=$(hostname -s) gpu=\"$GPU\" isg=$(git -C $CODE rev-parse --short HEAD) b1k=$(git -C $CODE/Behavior-1K rev-parse --short=9 HEAD) limit_s=$LIMIT args=${ARGS[*]} start=$(date -Is)" | tee $EP/job.txt
t0=$(date +%s)
timeout $LIMIT python -m ispatialgym.oracle.run_isg "${ARGS[@]}" --mode policy --policy ispatialgym.harness.policy:HarnessPolicy \
  --out $EP/run --record video --record-view head,external --video-size 540 --no-goal-override > $EP/run.log 2>&1
KIT_RC=$?
echo "RUN_EXIT $KIT_RC wall_s=$(( $(date +%s) - t0 ))" | tee -a $EP/job.txt
