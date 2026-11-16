#!/bin/bash
# Policy-eval (og-isgeval-20261116) R2 bundle. Writes ONLY under $JOB_S3 (s3://isg-demos/jobs/og-isgeval-20261116); every other prefix
# it names (the render fleet's boot list, isg-1020-b1k textures / templates / env) is read by the instances, never written.
#   source $EV/tool/creds.sh; bash build_bundle.sh <code snapshot dir> [what...]     what: boot code data worker policy (default all)
set -euo pipefail
[ "$JOB_S3" = s3://isg-demos/jobs/og-isgeval-20261116 ] || { echo "wrong JOB_S3 $JOB_S3"; exit 1; }
CODE=${1:?code dir}; shift
WHAT=${*:-boot code data worker policy}
NAME=$(basename "$CODE")
CAMP=/vision/u/minyeong/projects/isg_home/campaign_20261110
M=$CAMP/training_generation/batched_replay_r1/merged/v2
STAGE=$EV/bundle_stage
mkdir -p $STAGE
for w in $WHAT; do case $w in
  boot)  # the render fleet's boot list: env (isg-1020-b1k), behavior-1k assets, the patched Behavior-1K (all read-only sources)
    $S5 cp "$REND_S3/bundle/boot_list.txt" $STAGE/boot_list.txt >/dev/null
    $S5 cp $STAGE/boot_list.txt "$JOB_S3/bundle/boot_list.txt" >/dev/null; echo "boot list: $(wc -l < $STAGE/boot_list.txt) entries" ;;
  code)  # the snapshot with its symlinks resolved, Behavior-1K and runlogs left out (the worker links the unpacked Behavior-1K)
    tar -C "$CODE" -h --exclude=./Behavior-1K --exclude=./runlogs --exclude='*/__pycache__' -cf - . | zstd -q -T8 -10 \
      | $S5 pipe "$JOB_S3/bundle/code/$NAME.tar.zst" >/dev/null; echo "code $NAME uploaded" ;;
  data)  # /work/evaldata: the eval instances + demos, the per-instance BDDL overlay, the training task list (alias -> task_index)
    cp $M/train-lerobot/meta/tasks.jsonl $STAGE/train_tasks.jsonl
    # the dataset: the 2026-11-16 re-export (BDDLs inside, files.bddl_root) unless EVAL_LAYOUT names another; the overlay stays for older layouts
    tar -h -cf - -C ${EVAL_LAYOUT:-$EV/dataset/eval-layout-v2r} ispatialgym-eval-instances ispatialgym-eval-demos -C $EV/inputs --transform 's|^bddl_overlay_v2|bddl_overlay|' \
        bddl_overlay_v2 -C $STAGE train_tasks.jsonl | zstd -q -T8 -6 | $S5 pipe "$JOB_S3/bundle/data/eval_v2.tar.zst" >/dev/null
    echo "data uploaded" ;;
  worker)
    tar -C "$CODE/scripts/policy_eval/vast" -czf $STAGE/worker.tar.gz boot.sh eval_worker.py
    $S5 cp $STAGE/worker.tar.gz "$JOB_S3/bundle/worker.tar.gz" >/dev/null; sha256sum $STAGE/worker.tar.gz ;;
  policy)  # serving checkouts + venvs + uv Pythons (absolute paths; packed on the workstation that built them)
    for k in pi act dp; do $S5 cp $STAGE/policy_env_$k.tar.zst "$JOB_S3/bundle/policy/policy_env_$k.tar.zst" >/dev/null; echo "policy env $k uploaded"; done ;;
esac; done
$S5 ls "$JOB_S3/bundle/*" | tail -20
