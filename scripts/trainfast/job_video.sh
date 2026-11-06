#!/bin/bash
# owner review videos (2026-11-06): the OPEN-LOOP replay (run_isg --mode replay: the recorded actions stepped with physics, not the
# kinematic state replay) of a fast-flow job's successful demos, recording the tiled view (left wrist | head | right wrist over the
# external camera, 720 px). The job's own ISPATIALGYM_* / ISG_* switches apply (e.g. a stable-scene copy); --no-goal-override: the job's
# TROs are read, never written. One run_isg process per demo.
# VIDEO_SCENE (owner 2026-11-06: "only the dining room seems to be loaded"): full (default) replays in the full template.json with its
# evaluation rooms (ISPATIALGYM_FAST_PARTIAL_SCENE=0) - a fast-flow demo is collected in the sampler's cropped world (template meta
# scene_crop, rooms / region mode: everything the program never comes near left out), and run_isg rebuilds that crop unless told not to;
# the dropped objects lie outside what the robot reaches, so the open-loop actions meet the same contacts. cropped: the collector's world.
#   sbatch -J isgof_vid_<n> -A vision -p svl --exclude=svl9 --gres=gpu:titanrtx:1 -c 8 --mem=32G --time=<T> job_video.sh <job dir> <snapshot> <out dir> [max demos]
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/slurm/%x_%j.log
set -u
JOBDIR=$1; CODE=$2; OUT=$3; MAXD=${4:-4}
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $CODE/scripts/trainfast/env_fast.sh
set -a; source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' $JOBDIR/job.env); set +a
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
VIDEO_SCENE=${VIDEO_SCENE:-full}
case "$VIDEO_SCENE" in
  full) export ISPATIALGYM_FAST_PARTIAL_SCENE=0 ;;
  cropped) ;;
  *) echo "VIDEO_SCENE=$VIDEO_SCENE: expected full or cropped" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_ORACLE_STILL_EXIT=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
mkdir -p $OUT
echo "VIDEO_START $SLURM_JOB_ID job=$JOBDIR isg=$(cat $CODE/.snapshot_isg 2>/dev/null) scene=$VIDEO_SCENE start=$(date -Is)" | tee -a $OUT/job.txt
n=0
for m in $(ls $JOBDIR/demo/*/meta.json 2>/dev/null); do
  [ $n -ge $MAXD ] && break
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$m')).get('success') else 1)" || continue
  d=$(dirname $m); iid=$(basename $d)
  tro=$(find $JOBDIR/inst -name "$iid-tro_state.json" | head -1)
  [ -n "$tro" ] && [ -f $d/demo.hdf5 ] || continue
  t0=$(date +%s)
  timeout -s INT -k 120 3600 python -m ispatialgym.oracle.run_isg --instance $tro --mode replay --replay $d/demo.hdf5 --record video --record-view tiled \
    --video-size 720 --no-goal-override --out $OUT/video > $OUT/replay_$iid.log 2>&1
  echo "VIDEO_DEMO $iid rc=$? wall_s=$(( $(date +%s) - t0 )) $(grep -h RUN_ISG_RESULT $OUT/replay_$iid.log | grep -oE '"status": "[a-z_]+"|"n_steps": [0-9]+' | tr '\n' ' ')" | tee -a $OUT/job.txt
  n=$(( n + 1 ))
done
KIT_RC=0
echo "VIDEO_END $SLURM_JOB_ID demos=$n end=$(date -Is)" | tee -a $OUT/job.txt
