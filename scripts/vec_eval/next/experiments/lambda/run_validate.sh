#!/bin/bash
# Validation of the build-only changes of 2026-10-30 (clone-only batched Fabric sync, collision-mesh cache): (1) N=2 world state after the build,
# optimised vs stock switches, must be identical; (2) timers of the optimised apple N=8 build (slope); (3) apple / hamper N=8 replays against the
# existing N=1 references (verify_rules). Warm app-data of the speed checkout for every run.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next; mkdir -p $VOUT/validate $VOUT/timers $VOUT/after2  # not R: env_lambda.sh redefines R
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
build() {  # build <tag> <N> <instance> [env assignments...]
  local tag=$1 n=$2 inst=$3; shift 3
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --no-profile --dump-state $VOUT/validate/$tag.npz --out $VOUT/validate ) > $VOUT/validate/$tag.log 2>&1
  log "$tag rc=$? $(grep BUILD_PROFILE $VOUT/validate/$tag.log | grep -o 'build_s=[0-9.]*')"
  grep ISG_VEC_BUILD_TIMER $VOUT/validate/$tag.log > $VOUT/validate/$tag.timers.txt
}
log "state check: apple N=2 optimised vs stock switches"
build apple_n2_opt 2 $APPLE
build apple_n2_stock 2 $APPLE ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
$PY - <<'PYEOF'
import numpy as np, os
R = os.path.expanduser("~/vec_eval85/next/validate")
a, b = np.load(f"{R}/apple_n2_opt.npz"), np.load(f"{R}/apple_n2_stock.npz")
for k in a.files:
    x, y = a[k], b[k]
    same = x.shape == y.shape and np.array_equal(x, y)
    print(f"STATE_CHECK {k}: shape {x.shape} identical={same} max_abs_diff={float(np.max(np.abs(x - y))) if x.shape == y.shape else 'shape differs'}")
PYEOF
log "hamper N=2 optimised vs stock"
build hamper_n2_opt 2 $HAMPER
build hamper_n2_stock 2 $HAMPER ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
$PY - <<'PYEOF'
import numpy as np, os
R = os.path.expanduser("~/vec_eval85/next/validate")
a, b = np.load(f"{R}/hamper_n2_opt.npz"), np.load(f"{R}/hamper_n2_stock.npz")
for k in a.files:
    x, y = a[k], b[k]
    same = x.shape == y.shape and np.array_equal(x, y)
    print(f"STATE_CHECK {k}: shape {x.shape} identical={same} max_abs_diff={float(np.max(np.abs(x - y))) if x.shape == y.shape else 'shape differs'}")
PYEOF
log "timers apple N=8 optimised"
build apple_n8_opt 8 $APPLE
run_one() {  # run_one <out dir> <log> <instance> <N>
  local out=$1 lg=$2 inst=$3 n=$4
  rm -rf $out
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $inst --same-instance-all-slots --num-envs $n --layout grid-centered --scene-margin 5 \
      --wallclock 0 --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:validate --out $out ) > $lg 2>&1
  log "$(basename $out) rc=$? $(grep -c RUN_ISG_RESULT $lg) results; $(grep -o 'world built in [0-9.]*s' $lg | head -1); $(grep RUN_ISG_VEC_BATCH $lg | tail -1 | grep -o 'batch_steps/s=[0-9.]* env_steps/s=[0-9.]*')"
}
log "replays N=8 optimised"
run_one $VOUT/after2/apple_n8 $VOUT/after2/apple_n8.log $APPLE 8
run_one $VOUT/after2/hamper_n8 $VOUT/after2/hamper_n8.log $HAMPER 8
cd $NEXT
for t in apple hamper; do
  PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $VOUT/after2/${t}_n8 --ref $VOUT/after/${t}_n1 > $VOUT/after2/${t}_n8/compare.txt 2>&1
  PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $VOUT/after2/${t}_n8 --ref $VOUT/after/${t}_n1 > $VOUT/after2/${t}_n8/verify_rules.log 2>&1
  log "$t N=8 vs N=1: $($PY -c "import json;d=json.load(open('$VOUT/after2/${t}_n8/verify.json'));print('PASS' if d['pass'] else 'FAIL', d['reasons'][:3])")"
done
log "VALIDATE DONE"
