#!/bin/bash
# Pod-side throughput benchmark: how many concurrent sampler processes a card sustains and at what per-process speed.
#   bench.sh <K>      K = concurrent sampler processes for the concurrent phase (2 on 24 GB, 4 on 48 GB, 8 on 96 GB)
# Phases (all with the on-disk IK cache off so every run does the cold cuRobo work):
#   warm    1 x IORS ontop (first Isaac Sim launch: shader / extension caches; excluded from throughput)
#   single  1 x IORS ontop, then 1 x camera relocalization (renders), sequential -> per-process baseline
#   conc    K jobs at once, alternating IORS ontop / camera relocalization, staggered by 20 s -> aggregate throughput
# Output: /root/bench/result.json (+ nvidia-smi samples in /root/bench/gpu_*.csv, per-job logs in /root/bench/*.log).
set -uo pipefail
K="${1:-2}"
source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym
cd /root/ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ISPATIALGYM_SAMPLING_CACHE=off ISPATIALGYM_LOG_LEVEL=WARNING
B=/root/bench; mkdir -p "$B"
IORS=inter_object_relation_satisfaction-0
CAM=camera_relocalization-0

job() {  # job <phase> <idx> <problem> <seed>  -> appends a line "phase idx problem seed wall_s n_written" to $B/jobs.txt
  local phase=$1 idx=$2 problem=$3 seed=$4
  local out="$B/${phase}_${idx}"
  local t0=$(date +%s.%N)
  python -m ispatialgym.sampling.sample_instances --problem "$problem" --n-instances 2 --max-templates 1 --seed "$seed" --out "$out" --overwrite \
      > "$B/${phase}_${idx}.log" 2>&1
  local rc=$? t1=$(date +%s.%N)
  local nw=$(python - "$out" <<'EOF'
import glob, json, sys
reps = sorted(glob.glob(sys.argv[1] + "/sampling_report_*.json"))
if not reps:
    print(-1)
else:
    d = json.load(open(reps[-1]))
    # report layout: {"problems": {<problem>: {<template>: {"n_written": n, "time_s": t}}}}
    print(sum(int(t.get("n_written", 0)) for prob in d.get("problems", {}).values() for t in prob.values()))
EOF
)
  echo "$phase $idx $problem $seed $(python -c "print(round($t1-$t0,1))") $nw rc=$rc" >> "$B/jobs.txt"
}

gpu_sampler() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 > "$B/gpu_$1.csv" 2>/dev/null & echo $!; }

: > "$B/jobs.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$B/gpu_name.txt"
lscpu | grep -E "^CPU\(s\)|Model name" > "$B/cpu.txt"; free -g | head -2 > "$B/mem.txt"

echo "== warm"; P=$(gpu_sampler warm); job warm 0 $IORS 100; kill $P 2>/dev/null
echo "== single"; P=$(gpu_sampler single); job single 0 $IORS 200; job single 1 $CAM 201; kill $P 2>/dev/null
echo "== conc K=$K"; P=$(gpu_sampler conc); T0=$(date +%s.%N)
for i in $(seq 0 $((K-1))); do
  if (( i % 2 == 0 )); then job conc $i $IORS $((300+i)) & else job conc $i $CAM $((300+i)) & fi
  sleep 20
done
wait; T1=$(date +%s.%N); kill $P 2>/dev/null
echo "conc_total_wall_s $(python -c "print(round($T1-$T0,1))")" >> "$B/jobs.txt"

python - "$B" "$K" <<'EOF'
import csv, json, os, statistics as st, sys
B, K = sys.argv[1], int(sys.argv[2])
rows = [l.split() for l in open(f"{B}/jobs.txt") if l.strip()]
jobs = [r for r in rows if r[0] in ("warm", "single", "conc")]
tot = [r for r in rows if r[0] == "conc_total_wall_s"]
def gpu(phase):
    try:
        rs = list(csv.reader(open(f"{B}/gpu_{phase}.csv")))
        util = [float(r[1].strip().rstrip('%')) for r in rs if len(r) >= 3]
        mem = [float(r[2].strip().split()[0]) for r in rs if len(r) >= 3]
        return {"util_mean": round(st.mean(util), 1), "util_p90": round(sorted(util)[int(0.9 * len(util)) - 1], 1), "mem_peak_mb": max(mem)} if util else {}
    except Exception:
        return {}
res = {"gpu": open(f"{B}/gpu_name.txt").read().strip(), "cpu": open(f"{B}/cpu.txt").read().strip(), "K": K,
       "jobs": [{"phase": r[0], "idx": int(r[1]), "problem": r[2], "seed": int(r[3]), "wall_s": float(r[4]), "n_written": int(r[5]), "rc": r[6]} for r in jobs],
       "conc_total_wall_s": float(tot[0][1]) if tot else None, "gpu_stats": {p: gpu(p) for p in ("warm", "single", "conc")}}
single = {j["problem"]: j["wall_s"] for j in res["jobs"] if j["phase"] == "single"}
conc = [j for j in res["jobs"] if j["phase"] == "conc"]
if conc and single:
    slow = [j["wall_s"] / single[j["problem"]] for j in conc if j["problem"] in single]
    res["conc_slowdown_mean"] = round(st.mean(slow), 2)
    # throughput: jobs per hour of wall time (normalised to the single-process job mix)
    res["single_jobs_per_h"] = round(3600 / st.mean(single.values()), 2)
    res["conc_jobs_per_h"] = round(len(conc) / (res["conc_total_wall_s"] / 3600), 2) if res["conc_total_wall_s"] else None
json.dump(res, open(f"{B}/result.json", "w"), indent=2)
print(json.dumps(res, indent=2))
EOF
