#!/bin/bash
# Control for the state check: two STOCK-switch builds of the same N=2 world compared with each other (run-to-run noise of the build itself:
# a physics step and the warm renders follow the load), apple and hamper. If stock-vs-stock differs at the same 1e-5 / 1e-12 level as
# optimised-vs-stock, the batching difference is within the build's own noise.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next; mkdir -p $VOUT/validate
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
build() { local tag=$1 n=$2 inst=$3; shift 3
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --no-profile --dump-state $VOUT/validate/$tag.npz --out $VOUT/validate ) > $VOUT/validate/$tag.log 2>&1
  log "$tag rc=$? $(grep BUILD_PROFILE $VOUT/validate/$tag.log | grep -o 'build_s=[0-9.]*')"; }
# the earlier pairs' npz landed in the eval root (clobbered R); move them next to the new ones
mv $NEXT/validate/*.npz $VOUT/validate/ 2>/dev/null; rmdir $NEXT/validate 2>/dev/null
build apple_n2_stock2 2 $APPLE ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
build hamper_n2_stock2 2 $HAMPER ISG_VEC_BATCH_USD_EDITS=0 ISG_VEC_MESH_CACHE=0
build apple_n2_opt2 2 $APPLE
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
log "CONTROL DONE"
