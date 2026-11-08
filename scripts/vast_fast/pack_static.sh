#!/bin/bash
#SBATCH -J isgof_vpack
#SBATCH -A vision
#SBATCH -p svl
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --exclude=svl9
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/vast/logs/%x_%j.log
# i7-vast: the bundles the fast-flow workers need beyond the replay's (which stay read-only at $REPLAY_S3) -> $JOB_S3/bundle/:
#   assets/objects_rest_NN.tar.zst  every behavior-1k-assets model the replay's 959-model packs lack (encrypted as on disk; mtimes kept)
#   env_delta.tar.zst               env files changed since the replay env packs (2026-10-21), at their absolute paths
#   stores.tar.zst                  the read-only band / reach-volume stores env_fast.sh + job_fast.sh copy, kit_appdata.sh
#   cache/sampling_base_NN.tar.zst  a base copy of the campaign's sampling cache (exact-IK witnesses, start-draw cells), which the jobs
#                                   read AND write; cache/sampling.time = the listing time (a delta takes newer files)
# NOT shipped: the asset-geometry cache ($OFH/asset_cache: link-local collision points derived from the decrypted BEHAVIOR meshes - no
# decrypted asset data leaves the cluster); the instances start it empty (bit-identical demos, a slower first world load).
# Every archive is rooted at / (absolute cluster paths) unless named otherwise; boot_list.txt says where each one goes.
# Reads only NFS sources; writes only R2 ($JOB_S3) and $VJ/bundle (lists).
#   sbatch pack_static.sh [part ...]   parts: objects env stores sampling (default: all)
set -uo pipefail
source ${VTOOL:-/vision/u/minyeong/projects/isg_home/oracle_fix/vast/tool}/creds.sh   # VTOOL: a deployed tool copy (deploy.sh)
D=/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets
ENV=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
B=$VJ/bundle
mkdir -p $B
t0=$SECONDS
log() { echo "[$(date -u +%FT%TZ) +$(( SECONDS - t0 ))s] $*"; }
export -f log; export t0 S3_ENDPOINT_URL JOB_S3
PARTS="${*:-objects env stores sampling}"
fail=0
log "START host=$(hostname -s) parts=$PARTS"

# put <list file> <key> <tar -C dir> [zstd level]: tar of the list (mtimes kept) | zstd | R2
put() {
  if tar -C "$3" --format=posix -cf - -T "$1" | zstd -q -T2 "-${4:-3}" | s5cmd --endpoint-url $S3_ENDPOINT_URL pipe "$JOB_S3/bundle/$2" > /dev/null; then
    log "PUT $2 ($(wc -l < "$1") entries)"
  else
    log "FAILED $2"; return 1
  fi
}
export -f put

# split <listing (size TAB path)> <prefix> <k>: size-balanced lists <prefix>_NN.txt
split_lists() {
  python3 - "$1" "$2" "$3" <<'EOF'
import sys
src, pre, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
items = sorted(((int(s), p) for s, p in (l.split("\t", 1) for l in open(src).read().splitlines() if l)), reverse=True)
parts = [[0, []] for _ in range(k)]
for s, p in items:
    part = min(parts, key=lambda x: x[0]); part[0] += s; part[1].append(p)
for i, (s, ps) in enumerate(parts):
    open(f"{pre}_{i:02d}.txt", "w").write("\n".join(sorted(ps)) + "\n")
    print(i, len(ps), round(s / 1e9, 2), "GB")
print("total", len(items), round(sum(s for s, _ in items) / 1e9, 2), "GB")
EOF
}

if [[ " $PARTS " == *" objects "* ]]; then
  # every model dir objects/<category>/<model> not in the replay packs (scan/models_all.json: the 959 + 1 models it packed)
  python3 - $REPLAY_WORK/scan/models_all.json $D/behavior-1k-assets > $B/objects_rest.sizes <<'EOF'
import json, os, sys
have = {(c, m) for d, c, m, _ in json.load(open(sys.argv[1]))["models"] if d == "behavior-1k-assets"}
A = sys.argv[2]
for c in sorted(os.listdir(os.path.join(A, "objects"))):
    for m in sorted(os.listdir(os.path.join(A, "objects", c))):
        if (c, m) in have:
            continue
        tot = sum(os.lstat(os.path.join(r, f)).st_size for r, _, fs in os.walk(os.path.join(A, "objects", c, m)) for f in fs)
        print(f"{tot}\tobjects/{c}/{m}")
EOF
  split_lists $B/objects_rest.sizes $B/objects_rest 16
  ls $B/objects_rest_*.txt | xargs -P 8 -I{} bash -c 'f={}; n=$(basename $f .txt); put $f assets/$n.tar.zst '"$D/behavior-1k-assets"' 1' || fail=1
fi

if [[ " $PARTS " == *" env "* ]]; then
  # files the env gained or changed after the replay env packs (msgpack 1.1.1, Isaac Sim's apps/ copies of the OmniGibson kit + logo)
  (cd / && find ${ENV#/} -newermt "2026-10-21 00:00" ! -type d -not -path '*/__pycache__/*' -not -name '*.pyc') > $B/env_delta.txt
  put $B/env_delta.txt env_delta.tar.zst / 3 || fail=1
fi

if [[ " $PARTS " == *" stores "* ]]; then
  # read in place by every cluster job (env_fast.sh / job_fast.sh defaults): restored at the same paths on the instance
  printf '%s\n' \
    vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/reach_bands \
    vision/u/minyeong/projects/isg_home/eval85/bands_eval \
    vision/u/minyeong/projects/isg_home/.cache/ispatialgym-vecenv/reach_volume \
    vision/u/minyeong/projects/isg_home/train85/bands_train \
    vision/u/minyeong/projects/cluster_tools/kit_appdata.sh > $B/stores.txt
  put $B/stores.txt stores.tar.zst / 3 || fail=1
fi

cache_base() {  # name dir
  local name=$1 dir=$2
  date -u +"%Y-%m-%d %H:%M:%S" > $B/cache_$name.time   # the listing time: a later delta takes files newer than it
  python3 - "$dir" > $B/cache_$name.sizes <<'EOF'
import os, sys
d = sys.argv[1]
for e in os.scandir(d):
    if e.is_file(follow_symlinks=False) and not e.name.startswith("."):
        print(f"{e.stat(follow_symlinks=False).st_size}\t{os.path.relpath(e.path, '/')}")
EOF
  split_lists $B/cache_$name.sizes $B/cache_${name}_base 16
  ls $B/cache_${name}_base_*.txt | xargs -P 16 -I{} bash -c 'f={}; n=$(basename $f .txt); put $f cache/${n#cache_}.tar.zst / 3' || fail=1
  s5cmd --endpoint-url $S3_ENDPOINT_URL cp $B/cache_$name.time "$JOB_S3/bundle/cache/${name}.time" > /dev/null
}
[[ " $PARTS " == *" sampling "* ]] && cache_base sampling /vision/u/minyeong/projects/isg_home/trainfast/cache/sampling

s5cmd --endpoint-url $S3_ENDPOINT_URL ls "$JOB_S3/bundle/*" > $B/r2_listing.txt 2>&1
s5cmd --endpoint-url $S3_ENDPOINT_URL du --humanize "$JOB_S3/bundle/*" 2>&1 | tail -1
log "END fail=$fail"
exit $fail
