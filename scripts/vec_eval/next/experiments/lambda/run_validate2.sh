#!/bin/bash
# Validation, part 2 (the first script was edited while running and died at the replays): apple / hamper N=8 replays with the optimised build
# against the N=1 references (verify_rules), then the stock-vs-stock / opt-vs-opt control builds at N=2.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next; mkdir -p $VOUT/validate $VOUT/after2
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
run_one() { local out=$1 lg=$2 inst=$3 n=$4; rm -rf $out
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $inst --same-instance-all-slots --num-envs $n --layout grid-centered --scene-margin 5 \
      --wallclock 0 --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:validate --out $out ) > $lg 2>&1
  log "$(basename $out) rc=$? $(grep -c RUN_ISG_RESULT $lg) results; $(grep -o 'world built in [0-9.]*s' $lg | head -1); $(grep RUN_ISG_VEC_BATCH $lg | tail -1 | grep -o 'batch_steps/s=[0-9.]* env_steps/s=[0-9.]*')"; }
build() { local tag=$1 n=$2 inst=$3; shift 3
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --no-profile --dump-state $VOUT/validate/$tag.npz --out $VOUT/validate ) > $VOUT/validate/$tag.log 2>&1
  log "$tag rc=$? $(grep BUILD_PROFILE $VOUT/validate/$tag.log | grep -o 'build_s=[0-9.]*')"; }
log "replays N=8 optimised"
run_one $VOUT/after2/apple_n8 $VOUT/after2/apple_n8.log $APPLE 8
run_one $VOUT/after2/hamper_n8 $VOUT/after2/hamper_n8.log $HAMPER 8
cd $NEXT
for t in apple hamper; do
  PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $VOUT/after2/${t}_n8 --ref $VOUT/after/${t}_n1 > $VOUT/after2/${t}_n8/compare.txt 2>&1
  PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $VOUT/after2/${t}_n8 --ref $VOUT/after/${t}_n1 > $VOUT/after2/${t}_n8/verify_rules.log 2>&1
  log "$t N=8 vs N=1: $($PY -c "import json;d=json.load(open('$VOUT/after2/${t}_n8/verify.json'));print('PASS' if d['pass'] else 'FAIL', d['reasons'][:3])" 2>&1)"
done
log "control builds"
mv $NEXT/validate/*.npz $VOUT/validate/ 2>/dev/null; rmdir $NEXT/validate 2>/dev/null
build apple_n2_stock2 2 $APPLE ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
build apple_n2_opt2 2 $APPLE
build hamper_n2_stock2 2 $HAMPER ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
$PY - <<'PYEOF'
import numpy as np, os
D = os.path.expanduser("~/vec_eval85/next/validate")
def cmp(a, b, label):
    A, B = np.load(f"{D}/{a}.npz"), np.load(f"{D}/{b}.npz")
    for k in A.files:
        x, y = A[k], B[k]; d = np.abs(x - y)
        print(f"STATE_CHECK {label} {k}: identical={np.array_equal(x, y)} differing={int(np.count_nonzero(d))} max_abs_diff={float(d.max())}")
cmp("apple_n2_stock", "apple_n2_stock2", "apple stock-vs-stock")
cmp("apple_n2_opt", "apple_n2_opt2", "apple opt-vs-opt")
cmp("apple_n2_opt", "apple_n2_stock", "apple opt-vs-stock")
cmp("hamper_n2_stock", "hamper_n2_stock2", "hamper stock-vs-stock")
cmp("hamper_n2_opt", "hamper_n2_stock", "hamper opt-vs-stock")
PYEOF
log "VALIDATE2 DONE"
