#!/bin/bash
# Pull a Vast policy-eval run from R2 into its results dir and summarize it.
#   source $EV/tool/creds.sh; bash collect.sh <run> [--dest <results dir>]
# The results dir (default: spec.json results_dir = make_eval_run.py --results-dir, else $EV/runs/<run>/results) gets
# <ckpt>/<instance id>/ with every artifact of the instance's episode (collect_results.py); summarize.py then writes summary.md,
# summary_<ckpt>.json, instances.csv, groups.csv and qa.csv there (total / family / task class / task). Safe to rerun at any time.
set -euo pipefail
RUN=${1:?run}; shift
DEST=
while [ $# -gt 0 ]; do case $1 in --dest) DEST=$2; shift 2 ;; --outputs) shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; done
D=$EV/runs/$RUN
H=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $D/done $D/out
$S5 sync "$JOB_S3/runs/$RUN/done/*" $D/done/ >/dev/null 2>&1 || true
$S5 sync "$JOB_S3/runs/$RUN/out/*" $D/out/ >/dev/null 2>&1 || true
DEST=${DEST:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('results_dir') or '')" $D/spec.json)}
DEST=${DEST:-$D/results}
python3 "$H/collect_results.py" --run-dir $D --dest "$DEST"
python3 "$H/summarize.py" "$DEST" --quiet
echo "results: $DEST (summary.md, summary_<ckpt>.json, instances.csv, groups.csv, qa.csv)"
