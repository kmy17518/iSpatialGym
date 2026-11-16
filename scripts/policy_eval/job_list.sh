#!/bin/bash
# Run run_isg on a list of eval instances, one process each, in one Slurm GPU job (timing / sanity runs).
#   sbatch ... --export=ALL,CODE=<code dir>,LIST=<file of tro paths>,OUT=<out dir>,ARGS_FILE=<file: run_isg args, one line> job_list.sh
set -uo pipefail
ARGS=$(cat "${ARGS_FILE:?args file}")
source "$CODE/scripts/policy_eval/job_env.sh"
trap 'kit_appdata_teardown' EXIT
echo "=== $(date -u +%FT%TZ) $(hostname) $USER $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) code=$CODE"
mkdir -p "$OUT"
while read -r tro; do
  [ -z "$tro" ] && continue
  t0=$(date +%s.%N)
  python -u -m ispatialgym.run_isg --instance "$tro" --out "$OUT" $ARGS
  rc=$?
  echo "JOBLIST_DONE rc=$rc wall_s=$(python3 -c "import time; print(round(time.time() - $t0, 1))") tro=$tro"
done < "$LIST"
KIT_RC=0
