#!/bin/bash
# i7-vast: write $JOB_S3/bundle/boot_list.txt (phase 1, parallel) and boot_list2.txt (phase 2, in order) from what is on R2:
# the replay's env parts / assets (read only) + this job's objects_rest, stores (phase 1); the cache bases (1b, after boot_ready);
# env delta, cache deltas (2, in order).
#   VTOOL=... bash boot_lists.sh [--no-ready]
set -euo pipefail
source ${VTOOL:-/vision/u/minyeong/projects/isg_home/oracle_fix/vast/tool}/creds.sh
ENVP=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
S5="s5cmd --endpoint-url $S3_ENDPOINT_URL"
B=$VJ/bundle
{
  for p in $($S5 cat $REPLAY_S3/bundle/env/parts.txt); do echo "$REPLAY_S3/bundle/env/$p $ENVP"; done
  echo "$REPLAY_S3/bundle/assets/b1k_base.tar.zst /work/data/behavior-1k-assets"
  for k in 0 1 2 3 4 5; do echo "$REPLAY_S3/bundle/assets/b1k_objects_$k.tar.zst /work/data/behavior-1k-assets"; done
  $S5 ls "$JOB_S3/bundle/assets/objects_rest_*" | awk -v p="$JOB_S3/bundle/assets/" '{print p $NF, "/work/data/behavior-1k-assets"}'
  echo "$REPLAY_S3/bundle/assets/robot.tar.zst /work/data"
  echo "$REPLAY_S3/bundle/assets/task_instances.tar.zst /work/data"
  echo "$JOB_S3/bundle/stores.tar.zst /"
  echo "$JOB_S3/bundle/tools/rsync.tar.zst /"   # rsync 3.2.7 (Ubuntu 24.04 debs, files only): the image has none
} > $B/boot_list.txt
$S5 ls "$JOB_S3/bundle/cache/*_base_*" 2>/dev/null | awk -v p="$JOB_S3/bundle/cache/" '{print p $NF, "/"}' > $B/boot_list1b.txt || true
{
  echo "$JOB_S3/bundle/env_delta.tar.zst /"
  $S5 ls "$JOB_S3/bundle/cache/*_delta_*" 2>/dev/null | awk -v p="$JOB_S3/bundle/cache/" '{print p $NF, "/"}' | sort  || true
} > $B/boot_list2.txt
$S5 cp $B/boot_list.txt $JOB_S3/bundle/boot_list.txt > /dev/null
$S5 cp $B/boot_list1b.txt $JOB_S3/bundle/boot_list1b.txt > /dev/null
$S5 cp $B/boot_list2.txt $JOB_S3/bundle/boot_list2.txt > /dev/null
wc -l $B/boot_list.txt $B/boot_list1b.txt $B/boot_list2.txt
# boot_ready: workers wait for it before phase 1b (--no-ready while a cache base is still being packed)
if [ "${1:-}" != --no-ready ]; then date -u +%FT%TZ | $S5 pipe $JOB_S3/bundle/boot_ready > /dev/null; echo "boot_ready written"; fi
