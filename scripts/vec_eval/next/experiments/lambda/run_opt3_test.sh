#!/bin/bash
# Option 3 test (2026-10-31): copy the overlay (volume / collision-data caches, lazy clone emitters), then
#  (a) N=1 build state vs the current code's N=1 (cache off = stock path for scene 0 anyway) - must be identical,
#  (b) N=8 build time + state vs apple_n8_norepl (old code), (c) the new overlay with the caches off (ISG_VEC_MESH_CACHE=0 ISG_VEC_LAZY_EMITTERS=0).
set -u
until grep -q "SAP TEST DONE" $HOME/vec_eval85/next/sap_test.out 2>/dev/null; do sleep 30; done
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
O=$NEXT/scripts/vec_eval/b1k_overlay/OmniGibson/omnigibson; B=/home/minyeongk/Desktop/projects/Behavior-1K-vec-next-6a6194600/OmniGibson/omnigibson
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
V=$HOME/vec_eval85/next/options; APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
build() { local tag=$1 n=$2; shift 2
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
    python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs $n --no-profile --dump-state $V/$tag.npz --out $V ) > $V/$tag.log 2>&1
  log "$tag rc=$? $(grep -o 'build_s=[0-9.]*' $V/$tag.log | head -1) $(grep 'ISG_VEC_BUILD_TIMER \(sim.import_play\|env.final_play\|scene\[1\].add_objects\|env.load_robots\)' $V/$tag.log | awk '{print $2"="$3}' | tr '\n' ' ')"; }
build apple_n1_old 1
cp $O/prims/rigid_prim.py $B/prims/rigid_prim.py; cp $O/objects/usd_object.py $B/objects/usd_object.py; cp $O/simulator.py $B/simulator.py; cp $O/envs/env_base.py $B/envs/env_base.py
log "overlay copied $(md5sum $B/prims/rigid_prim.py $B/objects/usd_object.py | cut -c1-8 | tr '\n' ' ')"
build apple_n1_new 1
build apple_n8_new 8
build apple_n8_new_cacheoff 8 ISG_VEC_MESH_CACHE=0 ISG_VEC_LAZY_EMITTERS=0
$PY - <<'PYEOF'
import numpy as np, os
D = os.path.expanduser("~/vec_eval85/next/options")
def cmp(a, b, label):
    try:
        A, B = np.load(f"{D}/{a}.npz"), np.load(f"{D}/{b}.npz")
    except Exception as e:
        print("STATE_CHECK", label, "load failed", e); return
    for k in A.files:
        x = A[k]; y = B[k] if k in B.files else None
        if y is None or x.shape != y.shape: print(f"STATE_CHECK {label} {k}: shape {x.shape} vs {None if y is None else y.shape}"); continue
        d = np.abs(x - y); print(f"STATE_CHECK {label} {k}: identical={np.array_equal(x, y)} differing={int(np.count_nonzero(d))} max_abs_diff={float(d.max())}")
cmp("apple_n1_new", "apple_n1_old", "n1 new-vs-old")
cmp("apple_n8_new", "apple_n8_norepl", "n8 new-vs-old")
cmp("apple_n8_new_cacheoff", "apple_n8_norepl", "n8 new(caches off)-vs-old")
PYEOF
log "OPT3 TEST DONE"
