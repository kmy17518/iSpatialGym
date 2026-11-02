#!/bin/bash
# Multi-process measurement (2026-10-31, option 1): K concurrent processes x N scenes each (same demo in every slot), one app-data copy per
# process, GPU memory / CPU load sampled every 2 s. Output: ~/vec_eval85/next/mp/<alias>/k<K>n<N>/{p*.log, gpu.csv, summary.txt}
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
TASK=${1:?alias}; K=${2:?K}; N=${3:?N}
INST=$HOME/vec_eval85/instances/eval/$TASK.json
OUT=$HOME/vec_eval85/next/mp/$TASK/k${K}n${N}; rm -rf $OUT; mkdir -p $OUT
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1),$(cut -d' ' -f1 /proc/loadavg)"; sleep 2; done ) > $OUT/gpu.csv &
MON=$!
T0=$(date +%s)
pids=()
for k in $(seq 0 $((K - 1))); do
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$HOME/vec_eval85/next/appdata_mp/p$k ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_VEC_ROOT=$HOME/vec_eval85 ${ISG_MP_ENV:-}
    source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 \
      --tag vec_next:mp:$TASK --num-envs $N --same-instance-all-slots --out $OUT/p$k ) > $OUT/p$k.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
T1=$(date +%s); kill $MON 2>/dev/null
python3 - "$OUT" "$K" "$N" "$((T1 - T0))" <<'PY' | tee $OUT/summary.txt
import re, sys, glob
out, K, N, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
builds, sps, eps, ok = [], [], [], 0
for f in sorted(glob.glob(f"{out}/p*.log")):
    t = open(f, errors="replace").read()
    m = re.search(r"world built in ([0-9.]+)s", t); builds.append(float(m[1]) if m else None)
    m = re.search(r"env_steps/s=([0-9.]+)", t); eps.append(float(m[1]) if m else None)
    ok += "RUN_ISG_VEC_DONE" in t
mem = [int(l.split(",")[1]) for l in open(f"{out}/gpu.csv") if l.count(",") >= 3 and l.split(",")[1].strip().isdigit()]
load = [float(l.split(",")[3]) for l in open(f"{out}/gpu.csv") if l.count(",") >= 3]
print(f"K={K} N={N} scenes={K*N} done={ok}/{K} wall={wall}s builds={builds} env_steps/s per process={eps} sum={sum(e for e in eps if e):.1f} gpu_mem_peak={max(mem) if mem else None} MiB load_peak={max(load) if load else None}")
PY
