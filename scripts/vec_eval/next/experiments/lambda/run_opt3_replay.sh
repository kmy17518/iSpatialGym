#!/bin/bash
# SAP broadphase test (2026-10-31): per alias n1 (default MBP), n1, n8same; full replays; slot 0 vs n1, n1 vs n1.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
PY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/python
log() { echo "[$(date -Is)] $*"; }
for TASK in "$@"; do
  INST=$HOME/vec_eval85/instances/eval/$TASK.json; VOUT=$HOME/vec_eval85/next/opt3/$TASK; mkdir -p $VOUT
  for spec in "n1:1:" "n8same:8:--same-instance-all-slots"; do
    name=${spec%%:*}; rest=${spec#*:}; n=${rest%%:*}; extra=${rest#*:}; rm -rf $VOUT/$name; t=$SECONDS
    ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85
      source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
      python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 --tag vec_next:sap:$TASK --num-envs $n $extra --out $VOUT/$name ) > $VOUT/$name.log 2>&1
    log "$TASK $name rc=$? seconds=$((SECONDS - t)) $(grep -o 'world built in [0-9.]*s' $VOUT/$name.log | head -1) $(grep RUN_ISG_VEC_BATCH $VOUT/$name.log | tail -1 | grep -o 'env_steps/s=[0-9.]*')"
  done
  cd $NEXT
  PYTHONPATH=$NEXT $PY scripts/vec_eval/compare_slots.py $VOUT/n8same --ref $VOUT/n1 > $VOUT/n8same/compare.txt 2>&1
  PYTHONPATH=$NEXT $PY scripts/vec_eval/verify_rules.py $VOUT/n8same --ref $VOUT/n1 > $VOUT/n8same/verify_rules.log 2>&1
  log "$TASK VERDICT n8_sap vs n1: $($PY -c "import json;d=json.load(open('$VOUT/n8same/verify.json'));print('PASS' if d['pass'] else 'FAIL', 'relaxed', d.get('pass_relaxed'), [r[-150:] for r in d['reasons'][:2]])" 2>&1 | cut -c1-400)"
  $PY - "$VOUT" <<'PYEOF'
import glob, sys, numpy as np
V = sys.argv[1]
def ld(p): z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
def first(z, z1):
    keys = [k for k in z1 if (k.startswith("object_pose__") or k in ("joint_positions", "base_pose")) and k in z]
    T = min(len(z1["step"]), len(z["step"]))
    for t in range(T):
        if any(not np.array_equal(z[k][t], z1[k][t]) for k in keys): return t, T
    return "all", T
n1 = ld(glob.glob(f"{V}/n1/*/trajectory.npz")[0]); n1s = ld(glob.glob(f"{V}/n1/*/trajectory.npz")[0])
print("OPT3_CHECK n1 == n1 until", first(n1s, n1))
for i, d in enumerate(sorted(glob.glob(f"{V}/n8same/*.slot*"))):
    print(f"OPT3_CHECK slot {i} == n1 until", first(ld(d + "/trajectory.npz"), n1s))
PYEOF
done
log "OPT3 REPLAY DONE"
