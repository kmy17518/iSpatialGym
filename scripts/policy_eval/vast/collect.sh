#!/bin/bash
# Pull a policy-eval run's done/ records from R2 and summarize it (per checkpoint: instances, classes, tasks, families, total).
#   source $EV/tool/creds.sh; bash collect.sh <run> [--outputs]
# --outputs: also the run folders (out/<ckpt>/<unit>.tar.zst) and each instance's full meta.json extracted to
# metas/<ckpt>/<instance id>/meta.json (the summary then reads all four verdicts and both settle modes' errors from them).
set -euo pipefail
RUN=${1:?run}
D=$EV/runs/$RUN
mkdir -p $D/done
$S5 sync "$JOB_S3/runs/$RUN/done/*" $D/done/ >/dev/null 2>&1 || true
if [ "${2:-}" = --outputs ]; then
  $S5 sync "$JOB_S3/runs/$RUN/out/*" $D/out/ >/dev/null 2>&1 || true
  for t in $D/out/*/*.tar.zst; do
    [ -e "$t" ] || continue
    ck=$(basename $(dirname $t)); mark=$D/metas/$ck/.$(basename $t).done
    [ -e $mark ] && continue
    mkdir -p $D/metas/$ck
    zstd -dc "$t" | tar -x -C $D/metas/$ck --wildcards '*/meta.json' 2>/dev/null && touch $mark
  done
fi
python3 "$(dirname "$0")/../summarize.py" --run-dir $D
echo "summary: $D/summary.md, $D/summary_<ckpt>.json, $D/instances.csv"
