#!/bin/bash
# REPLAY_DIFF_20261115 final: wait for diff1 + diff1c + diff1d + diff1e, merge (train / standard-eval / held-out; one chunk per task family), verify
# against the curated tree, push to NEW private repos (owner 2026-11-15: new names, the earlier repos stay untouched).
set -uo pipefail
RV=/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1
C=$RV/code/rend_85d961ae; T=$C/scripts/batched_replay
source $RV/tool/creds.sh >/dev/null; export RV VPY
CPY=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
export PYTHONPATH=$C:$C/Behavior-1K/OmniGibson:$C/Behavior-1K/bddl3 OMNIGIBSON_DATA_PATH=$C/Behavior-1K/datasets HDF5_USE_FILE_LOCKING=FALSE
export HF_HOME=/tmp/isgrend_hf_$SLURM_JOB_ID HF_HUB_OFFLINE=1
RUNS=evalprod2,evalprod2t,evalprod2o,diff1,diff1c,diff1d,diff1e
EX=$RV/rows/v2_exclude_units.txt
cat $RV/rows/evalprod2_replaced_units.txt $RV/rows/diff1_retired_camera_units.txt $RV/rows/diff1c_retired_units.txt > $EX
# 1. wait: every unit of diff1 / diff1c / diff1d done (or dead) on R2 and pulled
while true; do
  for r in diff1 diff1c diff1d diff1e; do $CPY $T/sync.py --run $r pull 2>&1 | tail -1; done
  left=$($CPY - <<'PY'
import json, os, subprocess
RV = os.environ["RV"]; n = 0
for run in ("diff1", "diff1c", "diff1d", "diff1e"):
    units = [u["unit"] for u in json.load(open(f"{RV}/runs/{run}/units.json"))["units"]]
    ls = lambda sub: {l.split()[-1][:-5] for l in subprocess.run(os.environ["S5"].split() + ["ls", f"{os.environ['JOB_S3']}/runs/{run}/{sub}/*"], capture_output=True, text=True).stdout.splitlines() if l.strip()}
    done, dead = ls("done"), ls("dead")
    n += sum(1 for u in units if u not in dead and not (u in done and os.path.exists(f"{RV}/runs/{run}/out/{u}/.synced")))
print(n)
PY
)
  echo "$(date -u +%H:%M) units left $left"
  [ "$left" = 0 ] && break
  sleep 120
done
# 2. merges (in parallel)
M=$RV/merged/v2; mkdir -p $M; rm -rf $M/*.building
for s in train eval heldout; do
  rm -rf $M/$s-lerobot
  ( $CPY $T/merge_train.py --runs $RUNS --purpose $s --exclude-units $EX --name isg-$s-224-lerobot-v2 --out $M/$s-lerobot --load-test \
      > $RV/logs/v2_merge_$s.log 2>&1 && echo "MERGE $s OK" || echo "MERGE $s FAILED" ) &
done
for s in eval heldout; do
  rm -rf $M/$s-layout
  ( $CPY $T/export_eval.py --set $s --runs $RUNS --exclude-units $EX --out $M/$s-layout/ispatialgym-eval-instances \
      --demos-out $M/$s-layout/ispatialgym-eval-demos > $RV/logs/v2_export_$s.log 2>&1 && echo "EXPORT $s OK" || echo "EXPORT $s FAILED" ) &
done
wait
grep -hE 'selected|DONE |LeRobotDataset OK|goal images|instances,|load_index' $RV/logs/v2_merge_*.log $RV/logs/v2_export_*.log
# 3. verify
$CPY $T/verify_merge.py --train $M/train-lerobot --eval $M/eval-lerobot --eval-layout $M/eval-layout/ispatialgym-eval-instances \
  --heldout $M/heldout-lerobot --heldout-layout $M/heldout-layout/ispatialgym-eval-instances || { echo V2_VERIFY_FAILED; exit 1; }
# 4. push (new private repos)
unset HF_HUB_OFFLINE
for p in "train-lerobot isg-train-224-lerobot-v2" "eval-lerobot isg-standard-eval-224-lerobot-v2" "eval-layout isg-standard-eval-224-instances-v2" \
         "heldout-lerobot isg-heldout-sibling-224-lerobot-v2" "heldout-layout isg-heldout-sibling-224-instances-v2"; do
  set -- $p
  SRC=$M/$1 REPO=kmy17518/$2 bash $T/upload_hf.sh 2>&1 | grep -E 'DONE in|missing|refusing|Error' | sed "s|^|PUSH $2: |" || echo "PUSH $2 FAILED"
done
echo V2_ALL_DONE
