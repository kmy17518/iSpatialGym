#!/bin/bash
# isgf_* (trainfast campaign, FAST flow): the sampler and KCOLLECT physics-only oracle collectors run CONCURRENTLY on one GPU from the
# train-fast worktree (ISGE_CODE). The sampler writes instances into $JOBDIR/inst; each collector is a long-lived
# `run_isg --queue` process claiming instances as they land (O_EXCL claim files), restarted when it exits for a new world (rc 75) or an
# error, until the sampler is done and every instance is claimed. Demos are hdf5 only (no render, no video): images come from the
# replay stage. Per-process Kit app-data copies (one Kit process per folder).
#   sbatch -J isgf_<key> -A vision -p svl --gres=gpu:titanrtx:1 --time=<T> job_fast.sh <job dir>   (job.env by submit_fast.py --flow fast)
#SBATCH -c 9
#SBATCH --mem=48G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1
F=/vision/u/minyeong/projects/isg_home/trainfast
T85=/vision/u/minyeong/projects/isg_home/train85
source $JOBDIR/job.env
# of-v2maps 2026-11-07 (shiyuc mode, submit_fast.py --shiyuc): OmniGibson copies its .kit into $EXP_PATH on every launch - the account's own
# overlay (job.env ISGF_EXP_PATH); the kit stores come through KIT_TPL_NFS_ROOT / KIT_TEX_STORE in job.env (kit_appdata.sh reads them)
[ -n "${ISGF_EXP_PATH:-}" ] && export EXP_PATH=$ISGF_EXP_PATH
# ... and git reads the snapshot's Behavior-1K commit for the kit template key (kit_appdata.sh: `git rev-parse`; another account's checkout
# is "dubious ownership" -> key ..._nob1k -> a cold Kit start, sytest_sy 2026-11-07): safe.directory by environment, no config file written
if [ "${ISGF_GIT_SAFE_ALL:-0}" = 1 ]; then export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'; fi
[ -n "${KIT_TPL_NFS_ROOT:-}" ] && export KIT_TPL_NFS_ROOT
[ -n "${KIT_TEX_STORE:-}" ] && export KIT_TEX_STORE
# one trunk (owner 2026-11-05): the scripts come from the job's frozen code snapshot (submit_fast.py writes ISGF_SCRIPTS=<snapshot>/scripts/trainfast;
# unset = the campaign's live $F/scripts) and the job's outputs go under its root (ISGF_ROOT: $F for production, $OFH for alias validation)
S=${ISGF_SCRIPTS:-$F/scripts}
ROOT=${ISGF_ROOT:-$F}
K=${KCOLLECT:-2}
T_START=$(date +%s)
JOB_END=${SLURM_JOB_END_TIME:-$(( T_START + TIME_LIMIT_S ))}
KIT_RC=1
export ISGE_BAND_STORE=1 ISGE_CODE
[ -n "${TRUNK_CHECK:-}" ] && export ISGE_BAND_TRUNK_CHECK=$TRUNK_CHECK
source $S/env_fast.sh
[ -n "${ISGE_BAND_TRUNK_CHECK:-}" ] && export ISPATIALGYM_BAND_TRUNK_CHECK=$ISGE_BAND_TRUNK_CHECK
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache
if [ "${START_PER_ATTEMPT:-0}" = 1 ]; then export ISPATIALGYM_START_PER_ATTEMPT=1; else unset ISPATIALGYM_START_PER_ATTEMPT; fi
if [ "${STILL_EXIT:-0}" = 1 ]; then export ISPATIALGYM_ORACLE_STILL_EXIT=1; else unset ISPATIALGYM_ORACLE_STILL_EXIT; fi
CUSTOMS_ARG=""
# the snapshot's customs for a start-per-attempt alias and for CUSTOMS=trunk (submit_fast.py's default since 2026-11-06: the validated
# settings live in the trunk's isg_customs.json); CUSTOMS=template: multiply_isg reads the template's own copy (the counted run's customs)
{ [ "${START_PER_ATTEMPT:-0}" = 1 ] || [ "${CUSTOMS:-template}" = trunk ]; } && CUSTOMS_ARG="--customs $ISGE_CODE/ispatialgym/assets/task_specs/isg_customs.json"
# ---- the fast flow's settings (shared by sampler and collectors; see docs/FAST_FLOW.md in the worktree) ----
source $S/fast_env.sh
# per-job overrides given at submission (submit_fast.py --extra-env NAME=VALUE lands in job.env) win over env_fast.sh / fast_env.sh: their
# ISPATIALGYM_* / ISG_* assignments are applied again here, after both (until 2026-11-05 only the names were re-exported, so an override of
# a variable those scripts set - SEEDED_SPHERE_FIT, GOAL_OCCLUSION, HEAD_CAMERA_ONLY, ... - was silently lost)
set -a; source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' $JOBDIR/job.env); set +a
isgf_grasps_record $JOBDIR   # of-v2maps: ISGF_GRASPS legacy | v2 (fast_env.sh), the GRASPS line in job.txt
# the campaign-wide sampling cache (start-draw cells, exact IK witnesses) instead of the job-private one
export ISPATIALGYM_SAMPLING_CACHE=${ISGF_SAMPLING_CACHE:-$F/cache/sampling}
mkdir -p $ISPATIALGYM_SAMPLING_CACHE
# i7-plateqtf 2026-11-07: the expanding reach map's corrections (v2/map_tilt.py) - this job's own overlay shards (one file per process,
# never the frozen maps; merged offline by `python -m ispatialgym.sampling.v2.map_tilt merge`)
export ISPATIALGYM_MAP_OVERLAY_SHARDS=${ISPATIALGYM_MAP_OVERLAY_SHARDS:-$JOBDIR/map_overlay}
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*|${ISGF_JB_NFS:-$F/jobs}/*) rm -rf "$JB" ;; esac; rm -rf ${OMNIGIBSON_APPDATA_PATH}.c* 2>/dev/null' EXIT
mkdir -p $JOBDIR/claims
echo "$SLURM_JOB_ID" > $JOBDIR/jobid
# 2026-11-01: sharing between this campaign's jobs is opt-in (ISPATIALGYM_FAST_BAND_SHARE=1; default off after the contamination of
# 2026-11-01): train85's published bands (static, isg-1020's) are read as before; the campaign's own published bands and the band
# evidence of earlier jobs only with the share on
BAND_SHARE=${ISPATIALGYM_FAST_BAND_SHARE:-0}
for B in ${ISGF_T85_BANDS:-$T85/bands_train}/$ALIAS_KEY $( [ "$BAND_SHARE" = 1 ] && echo $F/bands_train/$ALIAS_KEY ); do [ -d "$B" ] && cp -n --preserve=timestamps $B/* $ISPATIALGYM_REACH_BANDS/ 2>/dev/null; done
# the alias's static band stores (submit_fast.py BAND_STORES, e.g. the bands an oracle-fix run persisted with the campaign's grasp library):
# copied over every store above, so the store's band of a name wins (job_val.sh's OF_BANDS order); read only, never written (no sharing)
for B in ${BAND_STORES:-}; do [ -d "$B" ] && cp --preserve=timestamps $B/*.json $B/*.npz $ISPATIALGYM_REACH_BANDS/ 2>/dev/null; [ -d "$B" ] && echo "BAND_STORE $B $(ls $B/*.json 2>/dev/null | wc -l) bands" >> $JOBDIR/job.txt; done
if [ "$BAND_SHARE" = 1 ]; then python -m ispatialgym.sampling.v2.band_evidence fetch $ISPATIALGYM_REACH_BANDS $F/band_evidence/$ALIAS_KEY 2>&1 | tee -a $JOBDIR/job.txt; fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
HEAD="key=$ALIAS_KEY alias=$ALIAS host=$(hostname -s) gpu=\"$GPU\" isg=$(cat $R/.snapshot_isg 2>/dev/null || git -C $R rev-parse --short HEAD) b1k=$(cat $R/.snapshot_b1k 2>/dev/null || git -C $R/Behavior-1K rev-parse --short HEAD)"
echo "ISGF_SAMPLE $SLURM_JOB_ID $HEAD flow=fast k=$K variant=${CODE_VARIANT:-fast} start_per_attempt=${ISPATIALGYM_START_PER_ATTEMPT:-0} still_exit=${ISPATIALGYM_ORACLE_STILL_EXIT:-0} start_index=$START n=$N max_attempts=$MAX_ATTEMPTS limit_s=$SAMPLE_LIMIT_S job_end=$JOB_END appdata=$OMNIGIBSON_APPDATA_PATH jb=$JB start=$(date -Is)" | tee -a $JOBDIR/job.txt
env | grep "^ISPATIALGYM_\|^ISG_\|^OMNIGIBSON_" | sort > $JOBDIR/env.txt
DEADLINE=$(( JOB_END - DEMO_LIMIT_S - 420 ))
# ---- GPU memory monitor (one line per minute: used MiB per process) ----
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr -d ' ' | tr '\n' ' ')"; sleep 60; done ) >> $JOBDIR/gpu.log 2>/dev/null &
MONPID=$!
# ---- COLLECTOR_PIN (2026-11-02, default on since its job-level A/B): the sampler and each collector on CPUs of their own (taskset;
# scripts/cpu_split.py, ISPATIALGYM_FAST_COLLECTOR_CPUS per collector). Unpinned Kit processes on one job's CPUs slow each other: two
# de03 collectors stepped at 334 ms each against 95 alone, and at 63-65 pinned to 3 CPUs each (E16); pinned de03 jobs made 30.6-32.6
# successes per GPU-hour on TITAN against 12.9 unpinned, 52.7-64.0 on L40S against 13.0-14.7 ----
PINS=()
if [ "${ISPATIALGYM_FAST:-1}" != 0 ] && [ "${ISPATIALGYM_FAST_COLLECTOR_PIN:-1}" != 0 ]; then
  PINS=( $(python3 $S/cpu_split.py $K ${ISPATIALGYM_FAST_COLLECTOR_CPUS:-3}) )
  if [ ${#PINS[@]} -gt 0 ]; then
    echo "PIN sampler=${PINS[0]} $(for i in $(seq 1 $K); do printf "c$i=%s " ${PINS[$i]}; done)" | tee -a $JOBDIR/job.txt
  else
    echo "PIN off: $(nproc) CPUs are fewer than $(( (K + 1) * ${ISPATIALGYM_FAST_COLLECTOR_CPUS:-3} ))" | tee -a $JOBDIR/job.txt
  fi
fi
pin_of() { [ ${#PINS[@]} -gt 0 ] && echo "taskset -c ${PINS[$1]}"; }
# ---- sampler ----
t0=$(date +%s)
( export ISPATIALGYM_PHYSICS_ONLY=; unset ISPATIALGYM_PHYSICS_ONLY
  timeout -s INT -k 300 $SAMPLE_LIMIT_S $(pin_of 0) python -m ispatialgym.sampling.multiply_isg --split train --template $TEMPLATE --instances $N --start-index $START \
    --out $JOBDIR/inst --eval-run $EVAL_INDEX --no-write-back --max-attempts $MAX_ATTEMPTS $KEEP_SCENE $CUSTOMS_ARG --log-level INFO > $JOBDIR/sample.log 2>&1
  echo $? > $JOBDIR/sample.rc ) &
SPID=$!
# ---- collectors: K long-lived queue consumers, each restarted while work may still come ----
collector() {
  local i=$1 q=0 barren=0 rc=0 app=${OMNIGIBSON_APPDATA_PATH}.c$i
  rsync -a --exclude 'local/logs' $OMNIGIBSON_APPDATA_PATH/ $app/ 2>/dev/null || cp -r $OMNIGIBSON_APPDATA_PATH $app
  export OMNIGIBSON_APPDATA_PATH=$app TMPDIR=$app/tmp ISPATIALGYM_PHYSICS_ONLY=1
  mkdir -p $TMPDIR
  sleep $(( (i - 1) * 45 ))   # stagger the Kit starts
  while true; do
    [ $(date +%s) -ge $DEADLINE ] && { echo "COLLECT_DEADLINE c$i" | tee -a $JOBDIR/job.txt; break; }
    [ $barren -ge 3 ] && { echo "COLLECT_GIVE_UP c$i: three processes in a row ended without a result" | tee -a $JOBDIR/job.txt; break; }
    if [ -e $JOBDIR/sampling.done ] && [ "$(python $S/queue_watch.py $JOBDIR left)" -eq 0 ]; then echo "COLLECT_DONE c$i" | tee -a $JOBDIR/job.txt; break; fi
    q=$(( q + 1 ))
    local r0=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l) t1=$(date +%s)
    echo "ISGF_COLLECT $SLURM_JOB_ID c$i q=$q left=$(python $S/queue_watch.py $JOBDIR left) deadline=$DEADLINE start=$(date -Is)" | tee -a $JOBDIR/job.txt
    $(pin_of $i) python -m ispatialgym.oracle.run_isg --queue $JOBDIR/inst --claims $JOBDIR/claims --done-file $JOBDIR/sampling.done --poll-s 15 \
      --results $JOBDIR/results.jsonl --deadline $DEADLINE --mode oracle --out $JOBDIR/demo --record hdf5 $COLLECT_EXTRA $( [ -e $JOBDIR/prebuild.mismatch ] || echo $PREBUILD_ARG ) > $JOBDIR/collect_c${i}_q$q.log 2>&1 &
    local qpid=$! killed=""
    while kill -0 $qpid 2>/dev/null; do
      sleep 20
      local over=$(python $S/queue_watch.py $JOBDIR over $qpid $DEMO_LIMIT_S)
      if [ -n "$over" ]; then
        echo "DEMO_TIMEOUT $over c$i pid=$qpid limit_s=$DEMO_LIMIT_S" | tee -a $JOBDIR/job.txt; echo "$over" >> $JOBDIR/demo_timeouts.txt
        kill -INT $qpid 2>/dev/null; sleep 60; kill -KILL $qpid 2>/dev/null; killed=1; break
      fi
    done
    wait $qpid; rc=$?; [ -n "$killed" ] && rc=124
    local r1=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
    echo "COLLECT_EXIT c$i q=$q rc=$rc wall_s=$(( $(date +%s) - t1 )) results=$(( r1 - r0 ))" | tee -a $JOBDIR/job.txt
    if [ "$r1" -eq "$r0" ] && [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then barren=$(( barren + 1 )); else barren=0; fi
    # 2026-11-06: no break on rc alone - a process that released its claim for a fresh world (run_isg EXIT_QUEUE_RESTART 75) reaches the
    # shell as 0 (og.shutdown exits 0), so an instance landing as sampling.done appeared was never collected (fix1_t03nr N 1 jobs: the
    # prebuild-mismatch instance left unclaimed); the loop ends at its top check (done and nothing ready left unclaimed)
    [ "$rc" -eq 0 ] && [ -e $JOBDIR/sampling.done ] && [ "$(python $S/queue_watch.py $JOBDIR left)" -eq 0 ] && break
  done
  echo $rc > $JOBDIR/collect_c$i.rc
}
# COLLECTOR_PREBUILD (2026-10-31): the collectors build their world from an earlier job's template copy (else the counted eval template) while the
# sampler loads, instead of after the first instance lands; run_isg falls back (prebuild.mismatch) when the job template differs
PREBUILD_ARG=""
if [ "${ISPATIALGYM_FAST:-1}" != 0 ] && [ "${ISPATIALGYM_FAST_COLLECTOR_PREBUILD:-1}" != 0 ]; then
  # source: a template copy an EARLIER train job of this alias wrote (the sampler's output - scene variant and settle as this job's
  # sampler writes them), newest first, this campaign then train85; the counted eval template (the sampler's input) only when none
  # exists - it differs where the train sampler draws another scene variant (disembedding_03)
  # 2026-11-02: only a copy whose scene crop request is this job's (scripts/prebuild_crop.py) - the 11-01 packing fleet took full4's
  # SCENE_CROP=full copies for cropped jobs and every collector rebuilt its world after its first instance (prebuild.mismatch)
  SRC=""; NSKIP=0
  for d in $(ls -td $( [ "$ROOT" != "$F" ] && echo $ROOT/out/*/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID ) $F/out/*/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID /vision/u/minyeong/projects/isg_home/train85/out/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID 2>/dev/null) ${COUNTED_TEMPLATE:-}; do
    case "$d" in $JOBDIR/*) continue ;; esac
    # shiyuc mode (2026-11-07): a source this account cannot read completely is skipped (a partial copy would make every collector rebuild)
    [ -n "$(find "$d" -maxdepth 2 ! -readable -print -quit 2>/dev/null)" ] && continue
    if [ -f "$d/template.json" ] && ls $d/instances/*-tro_state.json >/dev/null 2>&1; then
      fit=$(python $S/prebuild_crop.py $d 2>/dev/null || echo "skip prebuild_crop.py failed")
      if [ "$fit" = fits ]; then SRC=$d; break; fi
      NSKIP=$(( NSKIP + 1 )); [ $NSKIP -le 3 ] && echo "PREBUILD_SKIP $d $fit" | tee -a $JOBDIR/job.txt
    fi
  done
  [ $NSKIP -gt 0 ] && echo "PREBUILD_SKIPPED $NSKIP source(s) with another crop request" | tee -a $JOBDIR/job.txt
  CT=""
  [ -n "$SRC" ] && CT=$(ls $SRC/instances/*-tro_state.json 2>/dev/null | head -1)
  if [ -n "$CT" ]; then
    PD=$JOBDIR/prebuild/$(basename $(dirname $SRC))/$(basename $SRC)
    mkdir -p $PD/instances
    rsync -a --exclude instances --exclude runs $SRC/ $PD/
    cp $CT $PD/instances/
    PREBUILD_ARG="--prebuild $PD/instances/$(basename $CT)"
    echo "PREBUILD source=$SRC $PREBUILD_ARG" | tee -a $JOBDIR/job.txt
  fi
fi
CPIDS=""
for i in $(seq 1 $K); do collector $i & CPIDS="$CPIDS $!"; done
wait $SPID
SAMPLE_RC=$(cat $JOBDIR/sample.rc 2>/dev/null || echo 1)
echo "SAMPLE_EXIT $SAMPLE_RC wall_s=$(( $(date +%s) - t0 )) written=$(grep -c ' written (attempt ' $JOBDIR/sample.log)" | tee -a $JOBDIR/job.txt
if [ "$BAND_SHARE" = 1 ]; then
  mkdir -p $F/bands_train/$ALIAS_KEY
  python $S/publish_job_bands.py "$ISPATIALGYM_REACH_BANDS" "$F/bands_train/$ALIAS_KEY" "$t0" "$SLURM_JOB_ID" 2>&1 | tee -a $JOBDIR/job.txt
  python -m ispatialgym.sampling.v2.band_evidence publish $ISPATIALGYM_REACH_BANDS $F/band_evidence/$ALIAS_KEY $SLURM_JOB_ID 2>&1 | tee -a $JOBDIR/job.txt
fi
ISGF_PREDEMO_ROOT=${ISGF_PREDEMO_ROOT:-} python $S/train_predemo.py $JOBDIR 2>&1 | tee -a $JOBDIR/job.txt
touch $JOBDIR/sampling.done
wait $CPIDS
kill $MONPID 2>/dev/null
QRC=0; for i in $(seq 1 $K); do r=$(cat $JOBDIR/collect_c$i.rc 2>/dev/null || echo 1); [ "$r" -ne 0 ] && QRC=$r; done
python $S/fast_check.py $JOBDIR 2>&1 | tee -a $JOBDIR/job.txt
# i7-memory (owner 2026-11-08 part C): this job's outcome shard for the per-alias memory (<job dir>/alias_memory_shard.jsonl; one writer per job)
python -m ispatialgym.sampling.alias_memory record $JOBDIR 2>&1 | tail -1 | sed 's/^/ALIAS_MEMORY_SHARD /' | tee -a $JOBDIR/job.txt
case "$SAMPLE_RC" in 0|2) [ "$QRC" -eq 0 ] && KIT_RC=0 ;; esac
echo "ISGF_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T_START )) kit_rc=$KIT_RC end=$(date -Is)" | tee -a $JOBDIR/job.txt
