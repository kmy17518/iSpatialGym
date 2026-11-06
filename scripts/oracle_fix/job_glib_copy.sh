#!/bin/bash
# oracle-fix 2026-11-04 (owner item 5): the grasp-library v2 pipeline (export + candidates + physics validation) for a manifest into the
# oracle-fix COPY of the library ($OFH/glib_v2), never the grasp-bands record. The grasp-bands env with GBH=$OFH (asset cache, job dirs)
# and the oracle-fix worktree as its root.
#   GB_MANIFEST=<jobs.json> sbatch -J isgof_glib_<tag> -A vision -p svl --exclude=svl9 --gres=gpu:titanrtx:1 --time=2:00:00 scripts/oracle_fix/job_glib_copy.sh
#SBATCH -c 12
#SBATCH --mem=40G
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/slurm/%x_%j.log
set -u
export GBH=/vision/u/minyeong/projects/isg_home/oracle_fix ISG_GB_ROOT=/vision/u/minyeong/projects/ispatialgym-oracle-fix
source $ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
M=${GB_MANIFEST:?}; OUT=$GBH/glib_v2; mkdir -p $OUT $GBH/glib_logs
echo "ISGOF_GLIB $SLURM_JOB_ID manifest $M out $OUT host $(hostname) start $(date -Is)"
python -u -m ispatialgym.grasping.pipeline all --jobs $M --root $OUT --workers ${SLURM_CPUS_PER_TASK:-8} > $GBH/glib_logs/glib_$SLURM_JOB_ID.log 2>&1
rc=$?; grep -q "GB_VALIDATE_DONE" $GBH/glib_logs/glib_$SLURM_JOB_ID.log || rc=${rc/#0/1}
echo "ISGOF_GLIB_DONE rc=$rc end $(date -Is)"
