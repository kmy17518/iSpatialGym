#!/bin/bash
# Freeze a committed campaign revision into its own cluster root; old runtime/data remain read-only.
set -euo pipefail
REVISION=$(git rev-parse --verify "${1:-HEAD}^{commit}")
SHORT_REV=${REVISION:0:12}
CAMPAIGN_REMOTE=${CAMPAIGN_REMOTE:-minyeong@svl11}
CAMPAIGN_ROOT=/vision/u/minyeong/projects/isg_home/campaign_20261110
DEST="$CAMPAIGN_ROOT/code/trunk_$SHORT_REV"
if ssh "$CAMPAIGN_REMOTE" "test -d '$DEST'"; then
  FOUND=$(ssh "$CAMPAIGN_REMOTE" "cat '$DEST/.snapshot_commit'")
  [ "$FOUND" = "$REVISION" ] || { echo 'Existing snapshot has different provenance' >&2; exit 1; }
  echo "$DEST"
  exit 0
fi
ssh "$CAMPAIGN_REMOTE" "mkdir -p '$CAMPAIGN_ROOT/code'; mkdir '$DEST.tmp'"
git archive --format=tar "$REVISION" | ssh "$CAMPAIGN_REMOTE" "tar -x -C '$DEST.tmp'"
ssh "$CAMPAIGN_REMOTE" bash -s -- "$DEST" "$REVISION" <<'REMOTE'
set -euo pipefail
DEST=$1; REVISION=$2
FROZEN_B1K=/vision/u/minyeong/projects/isg_home/oracle_fix/code/b1k_6a6194600_p59fc36c6
SHARED_DATA=/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets
test -d "$FROZEN_B1K/OmniGibson"
mkdir "$DEST.tmp/Behavior-1K"
for ENTRY in "$FROZEN_B1K"/* "$FROZEN_B1K"/.[!.]*; do
  [ -e "$ENTRY" ] || continue
  [ "$(basename "$ENTRY")" = datasets ] && continue
  ln -s "$ENTRY" "$DEST.tmp/Behavior-1K/$(basename "$ENTRY")"
done
mkdir "$DEST.tmp/Behavior-1K/datasets"
for DATA in 2026-challenge-task-instances behavior-1k-assets omnigibson-robot-assets omnigibson.key; do
  ln -s "$SHARED_DATA/$DATA" "$DEST.tmp/Behavior-1K/datasets/$DATA"
done
ln -s "$DEST/ispatialgym/assets/datasets/ispatialgym-assets" "$DEST.tmp/Behavior-1K/datasets/ispatialgym-assets"
printf '%s\n' "$REVISION" > "$DEST.tmp/.snapshot_commit"
printf '%s\n' "${REVISION:0:12}" > "$DEST.tmp/.snapshot_isg"
printf '%s\n' '6a6194600+p59fc36c6' > "$DEST.tmp/.snapshot_b1k"
printf 'isg=%s b1k=6a6194600+p59fc36c6 created=%s\n' "$REVISION" "$(date -u +%FT%TZ)" > "$DEST.tmp/.snapshot.txt"
mv "$DEST.tmp" "$DEST"
chmod -R a-w "$DEST"
printf '%s\n' "$DEST"
REMOTE
