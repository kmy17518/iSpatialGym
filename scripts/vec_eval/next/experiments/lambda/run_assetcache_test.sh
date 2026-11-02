#!/bin/bash
# Persistent asset-geometry cache test (2026-10-31): builds with state dumps; cache off / fill / read must be bit-identical.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
O=$NEXT/scripts/vec_eval/b1k_overlay/OmniGibson/omnigibson; B=/home/minyeongk/Desktop/projects/Behavior-1K-vec-next-6a6194600/OmniGibson/omnigibson
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
V=$HOME/vec_eval85/next/options; AC=$HOME/vec_eval85/next/asset_cache_test
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json; HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
log() { echo "[$(date -Is)] $*"; }
cp $O/prims/rigid_prim.py $B/prims/rigid_prim.py; cp $O/envs/env_base.py $B/envs/env_base.py; cp $O/utils/isg_asset_cache.py $B/utils/isg_asset_cache.py
log "overlay copied $(md5sum $B/prims/rigid_prim.py $B/envs/env_base.py $B/utils/isg_asset_cache.py | cut -c1-8 | tr '\n' ' ')"
rm -rf $AC; mkdir -p $AC
build() { local tag=$1 inst=$2 n=$3; shift 3
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --no-profile --dump-state $V/$tag.npz --out $V ) > $V/$tag.log 2>&1
  log "$tag rc=$? $(grep -o 'build_s=[0-9.]*' $V/$tag.log | head -1) $(grep -o 'ISG_ASSET_CACHE build.*' $V/$tag.log | cut -c1-120) $(grep 'ISG_VEC_BUILD_TIMER \(scene\[0\].add_objects\|sim.import_play\)' $V/$tag.log | awk '{print $2"="$3}' | tr '\n' ' ')"; }
build ac_apple_n1_off $APPLE 1
build ac_apple_n1_fill $APPLE 1 ISG_VEC_ASSET_CACHE_DIR=$AC
build ac_apple_n1_read $APPLE 1 ISG_VEC_ASSET_CACHE_DIR=$AC
build ac_apple_n8_read $APPLE 8 ISG_VEC_ASSET_CACHE_DIR=$AC
build ac_hamper_n1_off $HAMPER 1
build ac_hamper_n1_fill $HAMPER 1 ISG_VEC_ASSET_CACHE_DIR=$AC
build ac_hamper_n1_read $HAMPER 1 ISG_VEC_ASSET_CACHE_DIR=$AC
ls -la $AC | tail -5
$PY - <<'PYEOF'
import numpy as np, os
D = os.path.expanduser("~/vec_eval85/next/options")
def cmp(a, b, label):
    try: A, B = np.load(f"{D}/{a}.npz"), np.load(f"{D}/{b}.npz")
    except Exception as e: print("STATE_CHECK", label, "load failed", e); return
    bad = [k for k in A.files if k not in B.files or A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
    print(f"STATE_CHECK {label}: {'IDENTICAL' if not bad else 'DIFFERS in ' + str(bad)} ({len(A.files)} scenes)")
cmp("ac_apple_n1_off", "apple_n1_new", "apple n1 cache-off vs previous code")
cmp("ac_apple_n1_fill", "ac_apple_n1_off", "apple n1 fill vs off")
cmp("ac_apple_n1_read", "ac_apple_n1_off", "apple n1 read vs off")
cmp("ac_apple_n8_read", "apple_n8_new", "apple n8 read vs previous code")
cmp("ac_hamper_n1_fill", "ac_hamper_n1_off", "hamper n1 fill vs off")
cmp("ac_hamper_n1_read", "ac_hamper_n1_off", "hamper n1 read vs off")
PYEOF
log "ASSETCACHE TEST DONE"
