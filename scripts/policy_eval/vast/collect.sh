#!/bin/bash
# Pull a policy-eval run's done/ records from R2 and summarize it (per checkpoint: instances, classes, tasks, families, total).
#   source $EV/tool/creds.sh; bash collect.sh <run> [--outputs]     (--outputs: also the run folders, out/<ckpt>/<unit>.tar.zst)
set -euo pipefail
RUN=${1:?run}
D=$EV/runs/$RUN
mkdir -p $D/done
$S5 sync "$JOB_S3/runs/$RUN/done/*" $D/done/ >/dev/null 2>&1 || true
[ "${2:-}" = --outputs ] && $S5 sync "$JOB_S3/runs/$RUN/out/*" $D/out/ >/dev/null 2>&1 || true
python3 "$(dirname "$0")/../summarize.py" --run-dir $D
echo "summary: $D/summary.md, $D/summary_<ckpt>.json, $D/instances.csv"
