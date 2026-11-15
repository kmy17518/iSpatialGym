#!/bin/bash
# Batched replay (og-isgrend-20261115) R2 bundle: boot list, code snapshot, worker. Writes ONLY under $JOB_S3
# (s3://isg-demos/jobs/og-isgrend-20261115); every other prefix it names is read by the instances, never written.
#   bash setup_bundle.sh <code dir on the cluster, e.g. $RV/code/rend_<sha>> <worktree scripts/batched_replay/vast dir>
set -euo pipefail
source /vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1/tool/creds.sh >/dev/null
[ "$JOB_S3" = s3://isg-demos/jobs/og-isgrend-20261115 ] || { echo "wrong JOB_S3"; exit 1; }
CODE=${1:?code dir}; VW=${2:?vast worker dir}
NAME=$(basename "$CODE")
mkdir -p "$RV/bundle"
# boot list: the eval fleet's (env isg-1020-b1k parts + og-isgof-vast env_delta, behavior-1k assets: the 959 replay models + the
# rest, robot, task instances, rsync) plus the patched Behavior-1K the snapshots name; sampler stores are not needed
{
  $S5 cat "$EVAL_S3/bundle/boot_list.txt" | grep -v "/stores.tar.zst"
  echo "$EVAL_S3/bundle/code/b1k_6a6194600_p59fc36c6.tar.zst /"
} > "$RV/bundle/boot_list.txt"
$S5 cp "$RV/bundle/boot_list.txt" "$JOB_S3/bundle/boot_list.txt" >/dev/null
# code: the snapshot dir with its symlinks resolved, Behavior-1K left out (the worker links the unpacked one and /work/data)
if ! $S5 ls "$JOB_S3/bundle/code/$NAME.tar.zst" >/dev/null 2>&1; then
  tar -C "$CODE" -h --exclude=./Behavior-1K --exclude=./runlogs --exclude='*/__pycache__' -cf - . | zstd -q -T8 -10 \
    | $S5 pipe "$JOB_S3/bundle/code/$NAME.tar.zst" >/dev/null
  echo "code $NAME uploaded"
else
  echo "code $NAME already on R2"
fi
# worker: boot.sh + worker.py (the image's entrypoint runs boot.sh from bundle/worker.tar.gz)
tar -C "$VW" -czf "$RV/bundle/worker.tar.gz" boot.sh worker.py
$S5 cp "$RV/bundle/worker.tar.gz" "$JOB_S3/bundle/worker.tar.gz" >/dev/null
sha256sum "$RV/bundle/worker.tar.gz"
$S5 ls "$JOB_S3/bundle/*"
