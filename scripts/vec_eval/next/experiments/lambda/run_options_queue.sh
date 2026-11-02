#!/bin/bash
# Options 1-3 measurements on lambda (2026-10-31), one GPU job at a time.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
O=$NEXT/scripts/vec_eval/b1k_overlay/OmniGibson/omnigibson; B=/home/minyeongk/Desktop/projects/Behavior-1K-vec-next-6a6194600/OmniGibson/omnigibson
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
V=$HOME/vec_eval85/next/options; mkdir -p $V
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
log() { echo "[$(date -Is)] $*"; }
cp $O/simulator.py $B/simulator.py; cp $O/envs/env_base.py $B/envs/env_base.py; log "overlay copied: $(md5sum $B/simulator.py $B/envs/env_base.py | cut -c1-8 | tr '\n' ' ')"
build() { local tag=$1 n=$2; shift 2
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA "$@"; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
    python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs $n --no-profile --dump-state $V/$tag.npz --out $V ) > $V/$tag.log 2>&1
  log "$tag rc=$? $(grep -o 'build_s=[0-9.]*' $V/$tag.log | head -1) $(grep -o 'ISG_PHYSX_REPLICATE play.*' $V/$tag.log | tr '\n' ' ') $(grep 'ISG_VEC_BUILD_TIMER \(sim.import_play\|env.final_play\)' $V/$tag.log | tr '\n' ' ')"; }
# option 2: physics replication, state identity
build apple_n8_norepl 8 ISG_PHYSX_REPLICATE=0
build apple_n8_repl 8 ISG_PHYSX_REPLICATE=1
$PY - <<'PYEOF'
import numpy as np, os
D = os.path.expanduser("~/vec_eval85/next/options")
try:
    A, B = np.load(f"{D}/apple_n8_norepl.npz"), np.load(f"{D}/apple_n8_repl.npz")
    for k in A.files:
        x, y = A[k], B[k] if k in B.files else None
        if y is None or x.shape != y.shape: print(f"STATE_CHECK repl {k}: shape {x.shape} vs {None if y is None else y.shape}"); continue
        d = np.abs(x - y); print(f"STATE_CHECK repl-vs-norepl {k}: identical={np.array_equal(x, y)} differing={int(np.count_nonzero(d))} max_abs_diff={float(d.max())}")
except Exception as e:
    print("STATE_CHECK failed", repr(e))
PYEOF
# option 3: where a clone's object setup goes (py-spy, N=4)
( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
  py-spy record -r 50 --format raw -o $V/pyspy_apple_n4.txt -- python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs 4 --no-profile --out $V ) > $V/pyspy_apple_n4.log 2>&1
log "pyspy rc=$? $(wc -l < $V/pyspy_apple_n4.txt 2>/dev/null) stacks"
# option 1: K processes x N scenes
for kn in "1 8" "2 4" "4 2" "8 1"; do set -- $kn; $HOME/vec_eval85/next/run_mp.sh object_scaling-apple $1 $2 > /dev/null 2>&1; log "mp k$1 n$2: $(cat $HOME/vec_eval85/next/mp/object_scaling-apple/k$1n$2/summary.txt)"; done
log "OPTIONS QUEUE DONE"
