#!/bin/bash
# i7-floating 2026-11-07: one shard of the grasp-library v2 FLOATING path (held-object grasps, no support; sibling of job_glib.sh):
#   GB_MANIFEST=<floating jobs.json> GB_NSHARD=N GB_TAG=<tag> GB_OUT=<out dir> ISG_GB_ROOT=<frozen snapshot> \
#     sbatch -A vision -p svl --exclude=svl9 --gres=gpu:titanrtx:1 --array=0-$((N-1)) -o <log> job_floating.sh
# Stages (GB_STAGES, default "export graspgen candidates validate"), each skipped per entry when its output exists:
#   export     - the collision geometry (pipeline export, Isaac env): <out>/glib/models/<id>/geometry.json
#   graspgen   - GraspGen proposals (GB_GG_ARGS: e.g. "--n-points 4096 --n-seeds 4 --num-grasps 4000 --topk 1000", the re-seed) in the graspgen env (scripts/grasp_bands/graspgen_floating.py; GRASPGEN_ENV, default the legacy run's
#                /vision/u/minyeong/projects/graspgen/env_graspgen.sh - the Franka Panda checkpoint): graspgen.npz / graspgen.json
#   candidates - the floating gates + legacy mode ranking (floating.generate; ISG_GLIB_GATE_DUMP=1 writes gate_dump.json)
#   validate   - in hand, gravity, the held motions (validate_floating): validation.json, library.json
# Jobs exit 0 even on failure: read the "ok" fields of the GB_* / GG lines in $GB_OUT/logs/floating_<shard>.log.
#SBATCH --job-name=isgof_float
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
set -u
GBH=${GBH:-/vision/u/minyeong/projects/isg_home/grasp_bands}
source ${ISG_GB_ROOT:?}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
I=${SLURM_ARRAY_TASK_ID:-0}; N=${GB_NSHARD:?}; TAG=${GB_TAG:?}; M=${GB_MANIFEST:?}
OUT=${GB_OUT:?}; mkdir -p $OUT/glib $OUT/logs
LOG=$OUT/logs/floating_$I.log
STAGES=${GB_STAGES:-export graspgen candidates validate}
echo "GB_JOB floating shard $I/$N tag $TAG host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $OUT start $(date +%s)" | tee -a $LOG
RC=0
for ST in $STAGES; do
  t0=$(date +%s)
  case $ST in
    export|candidates|validate)
      python -u -m ispatialgym.grasping.pipeline $ST --jobs $M --root $OUT/glib --shard $I/$N >> $LOG 2>&1; rc=$?
      grep -q "GB_$(echo $ST | tr a-z A-Z)_DONE" $LOG || rc=${rc/#0/1} ;;
    graspgen)
      # its own env (python 3.10, torch + GraspGen), in a subshell: the Isaac env above stays as it was for the next stages
      ( unset PYTHONPATH; source ${GRASPGEN_ENV:-/vision/u/minyeong/projects/graspgen/env_graspgen.sh}
        export MPLCONFIGDIR=${JB:-/tmp/$USER}/mpl TORCH_EXTENSIONS_DIR=${JB:-/tmp/$USER}/torch_ext
        python -u $R/scripts/grasp_bands/graspgen_floating.py --jobs $M --root $OUT/glib --shard $I/$N ${GB_GG_ARGS:-} ) >> $LOG 2>&1; rc=$?
      grep -q "GB_GRASPGEN_DONE" $LOG || rc=${rc/#0/1} ;;
    *) echo "unknown stage $ST"; rc=2 ;;
  esac
  [ $RC = 0 ] && RC=$rc
  echo "GB_STAGE $ST rc=$rc s=$(( $(date +%s) - t0 ))" | tee -a $LOG
done
echo "GB_JOB_DONE shard $I rc=$RC end $(date +%s)" | tee -a $LOG
