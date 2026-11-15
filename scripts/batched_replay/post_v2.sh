#!/bin/bash
# REPLAY_DIFF_20261115 after the merges: LeRobot load tests (node-local HF cache; a full /tmp fails them), verify, push to the new repos
set -uo pipefail
RV=/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1
C=$RV/code/rend_85d961ae; T=$C/scripts/batched_replay; M=$RV/merged/v2
source $RV/tool/creds.sh >/dev/null; export RV VPY
CPY=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
export PYTHONPATH=$C:$C/Behavior-1K/OmniGibson:$C/Behavior-1K/bddl3 HDF5_USE_FILE_LOCKING=FALSE
free=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); echo "node $(hostname) /tmp free ${free}G"
[ "$free" -lt 50 ] && { echo "POST_NO_TMP_SPACE"; exit 3; }
export HF_HOME=/tmp/isgrend_hf_$SLURM_JOB_ID HF_HUB_OFFLINE=1
trap 'rm -rf /tmp/isgrend_hf_$SLURM_JOB_ID' EXIT
for s in train eval heldout; do
  $CPY -c "import sys; sys.path.insert(0, '$T'); from merge_train import load_test; load_test('$M/$s-lerobot')" 2>&1 | grep -E 'LeRobotDataset OK|goal images|Error|Traceback' | sed "s/^/LOAD $s: /" || true
done
$CPY $T/verify_merge.py --train $M/train-lerobot --eval $M/eval-lerobot --eval-layout $M/eval-layout/ispatialgym-eval-instances \
  --heldout $M/heldout-lerobot --heldout-layout $M/heldout-layout/ispatialgym-eval-instances || { echo V2_VERIFY_FAILED; exit 1; }
unset HF_HUB_OFFLINE
for p in "train-lerobot isg-train-224-lerobot-v2" "eval-lerobot isg-standard-eval-224-lerobot-v2" "eval-layout isg-standard-eval-224-instances-v2" \
         "heldout-lerobot isg-heldout-sibling-224-lerobot-v2" "heldout-layout isg-heldout-sibling-224-instances-v2"; do
  set -- $p
  SRC=$M/$1 REPO=kmy17518/$2 bash $T/upload_hf.sh 2>&1 | grep -E 'DONE in|missing|refusing|Error' | sed "s|^|PUSH $2: |" || echo "PUSH $2 FAILED"
done
echo V2_ALL_DONE
