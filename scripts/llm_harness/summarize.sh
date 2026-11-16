#!/bin/bash
# Summarize a results folder of run_eval.py and zip the summary (the file to send back).
#   bash scripts/llm_harness/summarize.sh <results folder>
# -> <results>/summary/ (summary.md, summary_<name>.json, instances.csv, groups.csv, qa.csv) and <results>/summary_<folder>_<date>.zip
set -euo pipefail
R=${1:?results folder}
R=$(cd "$R" && pwd)
H=$(cd "$(dirname "$0")/../.." && pwd)
python "$H/scripts/policy_eval/summarize.py" "$R" --out "$R/summary" --quiet
Z="$R/summary_$(basename "$R")_$(date +%Y%m%d-%H%M).zip"
(cd "$R" && python -c "import shutil, sys; shutil.make_archive(sys.argv[1][:-4], 'zip', '.', 'summary')" "$Z")
echo "summary: $R/summary/summary.md"
echo "send:    $Z"
