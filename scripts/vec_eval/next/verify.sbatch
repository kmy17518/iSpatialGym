#!/bin/bash
#SBATCH -J vnverify
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:20:00
#SBATCH --exclude=viscam11,viscam14,viscam15,viscam16,viscam-hgx-1,viscam-hgx-2
#SBATCH -o /vision/u/minyeong/projects/isg_home/vec_eval85/campaigns/vec_next/logs/%x_%A_%a.log
# Campaign vec_next (2026-10-29): clone-count re-verification of one alias at N=8 on one 48 GB GPU with the integrated evaluator
# ($EVAL: the vec_speed levers + the co-location assisted-grasp fix + the rooms override + the prototype-once world build; driver defaults:
# centred grid 5 m, wall clock off with the stall guard, monitors off, video composed on the sink thread, lean step; replay mode keeps
# the external camera). One job = one alias: ISG_TASK=<alias>, or ISG_TASK_LIST=<file> + the array index. Phases (ISG_PHASES, default all):
#   n1      : the eval instance at N=1 (the reference; same GPU, same process environment)      -> <runs>/<alias>/n1
#   n8      : the same demo in every slot at N=8 (--same-instance-all-slots)                    -> <runs>/<alias>/n8same
#   compare : compare_slots + verify_rules of n8same against n1; the campaign's verdict file    -> $V/campaigns/vec_next/verify/<alias>.json
#   sbatch -A viscam -p viscam --gres=gpu:1 --constraint=48G --export=ALL,ISG_TASK=object_scaling-apple verify.sbatch
#   sbatch -A viscam -p viscam --gres=gpu:1 --constraint=48G --array=0-52%16 --export=ALL,ISG_TASK_LIST=$V/campaigns/vec_next/inputs/aliases.txt verify.sbatch
#   round C: --export=ALL,ISG_TASK_LIST=...,ISG_B1K_DIR=...,ISG_RUNS_DIR=runs_C,ISG_VERIFY_DIR=verify_C,ISG_REF_RUNS_DIR=runs_B,ISG_PHASES=n1+compare_n1+n8+compare
set -u
V=/vision/u/minyeong/projects/isg_home/vec_eval85
EVAL=/vision/u/minyeong/projects/ispatialgym-eval-vec-next   # not 'W': creds.sh (sourced below) sets W to its own work dir (smoke job 18189590)
B1K=${ISG_B1K_DIR:-$EVAL/Behavior-1K}                         # the Behavior-1K checkout (overlay state) this round runs; round B: Behavior-1K-vec-next-b
RUNS_DIR=${ISG_RUNS_DIR:-runs}; VERIFY_DIR=${ISG_VERIFY_DIR:-verify}   # round B writes runs_B/ and verify_B/
EXTRA_ARGS=${ISG_EXTRA_ARGS:-}                                       # e.g. "--physics-override rigid_sleep_threshold=0.0005" (round C)
REF_RUNS_DIR=${ISG_REF_RUNS_DIR:-}                                   # compare_n1 phase: this round's n1 vs <REF_RUNS_DIR>/<alias>/n1 (physics parity)
C=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache
ENVP=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
CAMP=$V/campaigns/vec_next
if [ -n "${ISG_TASK_LIST:-}" ]; then TASK=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$ISG_TASK_LIST"); else TASK=${ISG_TASK:?}; fi
[ -n "$TASK" ] || { echo "no task for index ${SLURM_ARRAY_TASK_ID:-0}"; exit 2; }
PHASES=${ISG_PHASES:-n1,n8,compare}; PHASES=${PHASES//[+:]/,}; N=${ISG_N:-8}   # sbatch --export splits on commas: pass phases as n1+compare_n1+n8+compare
echo "[$(date -Is)] job ${SLURM_JOB_ID:-local} on $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1) cpus=${SLURM_CPUS_PER_TASK:-?} task=$TASK N=$N phases=$PHASES"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
case "$DRV" in 57[0-9].*|58[0-9].*) ;; *) echo "driver $DRV is not 570-580; exiting"; exit 0;; esac
source /vision/u/minyeong/projects/isg_home/train85/work/vast_replay/bin/creds.sh         # R2 keys for the asset staging (never printed)
RUNS=$CAMP/$RUNS_DIR/$TASK; mkdir -p "$RUNS" $CAMP/$VERIFY_DIR
# node-local work dir (assets, caches); /dev/shm when /tmp is full
if [ "$(df --output=avail -k /tmp 2>/dev/null | tail -1)" -gt 100000000 ] 2>/dev/null; then WORK=/tmp/minyeong/isg_vnext_${SLURM_JOB_ID:-$$}; else WORK=/dev/shm/minyeong/isg_vnext_${SLURM_JOB_ID:-$$}; fi
mkdir -p "$WORK/home/.cache" "$WORK/inductor_cache" "$WORK/triton_cache" "$WORK/cuda_cache" || { echo "cannot create $WORK"; exit 3; }
# Kit app-data from the NFS template of this GPU (cluster_tools helper), textures from the vec store
export WT=$EVAL KIT_TEX_STORE=$C/kit_textures_vec
echo "b1k: $B1K head $(git -C $B1K rev-parse --short=9 HEAD 2>&1) overlay files $(git -C $B1K status --short | wc -l) (git $(git --version 2>&1 | cut -c1-30))"   # the app-data template key needs it (smoke job 18189590 got nob1k)
source /vision/u/minyeong/projects/cluster_tools/kit_appdata.sh; kit_appdata_setup
KIT_RC=1
cleanup() { kit_appdata_teardown; case "$WORK" in /tmp/minyeong/isg_vnext_*|/dev/shm/minyeong/isg_vnext_*) rm -rf -- "$WORK";; esac; }
trap cleanup EXIT
# assets on node-local disk (c3's recipe), staged ONCE PER NODE into a shared cache dir (flock; .complete marker; never removed - several
# array tasks land on one node and each streaming its own 30 GB copy from R2 gave truncated streams and filled /dev/shm, job 18190376);
# when /tmp is full the per-job /dev/shm copy of before is used and removed at exit
t0=$SECONDS; S5="s5cmd --endpoint-url $S3_ENDPOINT_URL --numworkers 16"; B=s3://isg-demos/isg-1020-b1k
stage_assets() {  # stage_assets <dir>: every part extracted under <dir>/behavior-1k-assets or <dir>; returns 1 on failure
  local root=$1 part dst attempt part_ok
  mkdir -p "$root/behavior-1k-assets" || return 1
  set -o pipefail
  for part in b1k_base b1k_objects_0 b1k_objects_1 b1k_objects_2 b1k_objects_3 b1k_objects_4 b1k_objects_5 robot task_instances; do
    case $part in robot|task_instances) dst=$root;; *) dst=$root/behavior-1k-assets;; esac
    part_ok=0
    for attempt in 1 2 3; do
      if ($S5 cat "$B/bundle/assets/$part.tar.zst" | /vision/u/minyeong/miniconda3/bin/zstd -d -q | tar -x --no-same-owner -C "$dst"); then part_ok=1; break; fi
      echo "[$(date -Is)] assets: $part attempt $attempt failed"; sleep $((30 * attempt))
    done
    if [ $part_ok = 0 ]; then set +o pipefail; return 1; fi
  done
  set +o pipefail
  return 0
}
ok=0
case "$WORK" in
  /tmp/minyeong/*)
    ASSETS=/tmp/minyeong/isg_assets_isg-1020-b1k; mkdir -p "$ASSETS"
    exec 9> "$ASSETS.lock"; flock 9
    if [ -f "$ASSETS/.complete" ]; then ok=1; echo "[$(date -Is)] assets: node cache $ASSETS reused"
    else rm -rf "$ASSETS"/*; if stage_assets "$ASSETS"; then touch "$ASSETS/.complete"; ok=1; else echo "assets: node cache staging failed"; fi; fi
    flock -u 9; exec 9>&-
    DATA_SRC=$ASSETS;;
  *)
    if stage_assets "$WORK/data"; then ok=1; else echo "assets: /dev/shm staging failed"; fi
    DATA_SRC=$WORK/data;;
esac
DATA=$B1K/datasets
if [ $ok = 1 ] && [ -d "$DATA_SRC/behavior-1k-assets/objects" ]; then
  mkdir -p "$WORK/datasets"
  for d in behavior-1k-assets omnigibson-robot-assets 2026-challenge-task-instances; do ln -sfn "$DATA_SRC/$d" "$WORK/datasets/$d"; done
  ln -sfn /vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets/omnigibson.key "$WORK/datasets/omnigibson.key"
  ln -sfn $EVAL/ispatialgym/assets/datasets/ispatialgym-assets "$WORK/datasets/ispatialgym-assets"
  for d in ispatialgym-eval-instances ispatialgym-eval-demos; do ln -sfn /vision/u/minyeong/projects/ispatialgym-eval/Behavior-1K/datasets/$d "$WORK/datasets/$d"; done
  DATA=$WORK/datasets
  echo "[$(date -Is)] assets ready ($DATA_SRC) in $((SECONDS - t0)) s"
else
  echo "[$(date -Is)] assets: local staging failed; NFS datasets"
fi
# the worker's process environment (eval_worker.proc_env)
export PATH=$ENVP/bin:$PATH CONDA_PREFIX=$ENVP OMNI_CRASHREPORTER_ENABLED=0
export PYTHONPATH=$EVAL:$B1K/OmniGibson:$B1K/bddl3 OMNIGIBSON_DATA_PATH=$DATA
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1
export OMNIGIBSON_TRACK_STATE_UPDATES=1 OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0
export ISPATIALGYM_SHARED_GOAL_EVAL=1 ISPATIALGYM_FAST_GEOMETRY=1 ISPATIALGYM_FAST_ASSIGN=1 ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1 HOME=$WORK/home XDG_CACHE_HOME=$WORK/home/.cache TORCHINDUCTOR_CACHE_DIR=$WORK/inductor_cache TRITON_CACHE_DIR=$WORK/triton_cache
export NUMBA_CACHE_DIR=$WORK/home/numba MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUDA_CACHE_PATH=$WORK/cuda_cache CUDA_CACHE_MAXSIZE=$((4 * 1024 * 1024 * 1024))
export ISG_VEC_ROOT=$V
# the adopted rooms override (campaign rooms 2026-10-28; frozen copy in this campaign's inputs): both the N=1 reference and the clones load it
export ISG_ROOMS_OVERRIDE=${ISG_ROOMS_OVERRIDE:-$CAMP/inputs/rooms_override.json}
cd "$EVAL" || exit 3
python -c "import ispatialgym.oracle.run_isg_vec" || { echo "driver import failed (PYTHONPATH=$PYTHONPATH cwd=$(pwd))"; exit 3; }
INST=$V/instances/eval/$TASK.json
COMMON="--mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag vec_next:verify:$TASK $EXTRA_ARGS"
run() { local name=$1; shift; echo "[$(date -Is)] $name start: $*"; local t=$SECONDS; python -u -m ispatialgym.oracle.run_isg_vec $COMMON "$@" > "$RUNS/$name.log" 2>&1; local rc=$?
        echo "[$(date -Is)] $name rc=$rc seconds=$((SECONDS - t)) $(grep -c RUN_ISG_RESULT "$RUNS/$name.log") results; $(grep -o 'world built in [0-9.]*s' "$RUNS/$name.log" | head -1); $(grep RUN_ISG_VEC_BATCH "$RUNS/$name.log" | tail -1 | cut -c1-160)"
        echo "{\"rc\": $rc, \"seconds\": $((SECONDS - t)), \"host\": \"$(hostname -s)\", \"job\": \"${SLURM_JOB_ID:-local}\"}" > "$RUNS/$name.run.json"; return $rc; }
RC=0
case ",$PHASES," in *,n1,*) rm -rf "$RUNS/n1"; run n1 --num-envs 1 --out "$RUNS/n1" || RC=1;; esac
case ",$PHASES," in *,n8,*) rm -rf "$RUNS/n8same"; run n8same --num-envs $N --same-instance-all-slots --out "$RUNS/n8same" || RC=1;; esac
case ",$PHASES," in *,compare_n1,*)
  if [ -n "$REF_RUNS_DIR" ] && [ -d "$CAMP/$REF_RUNS_DIR/$TASK/n1" ]; then
    PYTHONPATH=$EVAL python $EVAL/scripts/vec_eval/compare_slots.py "$RUNS/n1" --ref "$CAMP/$REF_RUNS_DIR/$TASK/n1" > "$RUNS/n1/compare_vs_${REF_RUNS_DIR}.txt" 2>&1 || echo "compare_slots(n1) rc=$?"
    PYTHONPATH=$EVAL python $EVAL/scripts/vec_eval/verify_rules.py "$RUNS/n1" --ref "$CAMP/$REF_RUNS_DIR/$TASK/n1" > "$RUNS/n1/verify_rules_vs_${REF_RUNS_DIR}.log" 2>&1 || echo "verify_rules(n1) rc=$?"
    cp "$RUNS/n1/verify.json" "$RUNS/n1/verify_vs_${REF_RUNS_DIR}.json" 2>/dev/null || true
    echo "[$(date -Is)] N1_PARITY vs $REF_RUNS_DIR: $(python -c "import json;d=json.load(open('$RUNS/n1/verify.json'));print('PASS' if d['pass'] else 'FAIL', d['reasons'][:2])" 2>&1)"
  else echo "compare_n1: no reference $CAMP/$REF_RUNS_DIR/$TASK/n1"; fi;; esac
case ",$PHASES," in *,compare,*)
  PYTHONPATH=$EVAL python $EVAL/scripts/vec_eval/compare_slots.py "$RUNS/n8same" --ref "$RUNS/n1" > "$RUNS/n8same/compare.txt" 2>&1 || echo "compare_slots rc=$?"
  PYTHONPATH=$EVAL python $EVAL/scripts/vec_eval/verify_rules.py "$RUNS/n8same" --ref "$RUNS/n1" > "$RUNS/n8same/verify_rules.log" 2>&1 || echo "verify_rules rc=$?"
  PYTHONPATH=$EVAL ISG_EVAL_ROOT=$EVAL python $EVAL/scripts/vec_eval/next/verify_record.py --task "$TASK" --runs "$RUNS" --num-envs $N --out "$CAMP/$VERIFY_DIR/$TASK.json" --previous "$V/mirror/verify/$TASK.json" --b1k "$B1K" || RC=1
  echo "[$(date -Is)] VERDICT $(python -c "import json;d=json.load(open('$CAMP/$VERIFY_DIR/$TASK.json'));print('PASS' if d['pass'] else 'FAIL', d['runs'].get('n8same',{}).get('reasons',[])[:3])" 2>&1)";; esac
KIT_RC=$RC
echo "[$(date -Is)] verify done rc=$RC task=$TASK"
exit $RC
