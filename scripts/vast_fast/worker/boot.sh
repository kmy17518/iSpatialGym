#!/bin/bash
# i7-vast instance boot for the round-4 fast-flow jobs (started by the isg-replay:v3 image's entrypoint from /opt/isg/run with
# ISG_S3 = this job's prefix s3://isg-demos/jobs/og-isgof-vast-20261108): the bundles of $ISG_S3/bundle/boot_list.txt (the replay's env,
# assets, robot, task instances at $REPLAY_S3, read only; this job's env delta, remaining objects, stores, caches) onto the instance
# disk at the cluster's absolute paths, the GPU / env checks, the cluster layout (datasets links, conda.sh, git, kit templates), then
# worker.py: it runs the job_fast.sh jobs the coordinator assigns to this instance until it is told to stop or stays idle.
# The instance destroys itself at the end (or when a check fails) with its CONTAINER_API_KEY; logs go to $ISG_S3/logs/<instance>/.
set -u
RUN=/opt/isg/run
LOG=/var/log/isg
mkdir -p $LOG /work
exec > >(tee -a $LOG/boot.log) 2>&1
t0=$SECONDS
log() { echo "[$(date -u +%FT%TZ) +$(( SECONDS - t0 ))s] $*"; }
if [ -n "${CONTAINER_ID:-}" ]; then ISG_INSTANCE=$CONTAINER_ID; else ISG_INSTANCE=$(hostname); fi
export ISG_INSTANCE
export ENVP=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
export REPLAY_S3=${REPLAY_S3:-s3://isg-demos/isg-1020-b1k}
S5="s5cmd --endpoint-url $S3_ENDPOINT_URL --numworkers 64"
[ -s /etc/machine-id ] || cat /proc/sys/kernel/random/uuid | tr -d '-' > /etc/machine-id
mkdir -p /var/lib/dbus && { [ -e /var/lib/dbus/machine-id ] || cp /etc/machine-id /var/lib/dbus/machine-id; }
export LANG=C.UTF-8 LC_ALL=C.UTF-8
log "BOOT instance=$ISG_INSTANCE cpus=$(nproc) mem=$(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo) disk=$(df -h --output=avail /work | tail -1)"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null
shm_kb=$(df -k --output=size /dev/shm | tail -1)
[ "${shm_kb:-0}" -lt 16000000 ] && mount -o remount,size=24G /dev/shm 2>/dev/null && log "remounted /dev/shm at 24G"
log "shm $(df -h --output=size /dev/shm | tail -1) nofile $(ulimit -n) nproc $(ulimit -u) cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) mem.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null) tmp $(df -h --output=avail /tmp | tail -1)"

upload_logs() { $S5 cp "$LOG/*" "$ISG_S3/logs/$ISG_INSTANCE/" > /dev/null 2>&1; }
heartbeat() {
  printf '{"instance": "%s", "phase": "%s", "detail": "%s", "time": %s}\n' "$ISG_INSTANCE" "$1" "${2:-}" "$(date +%s)" \
    | $S5 pipe "$ISG_S3/hb/$ISG_INSTANCE.json" > /dev/null 2>&1
}
self_destroy() {
  [ -n "${HB_PID:-}" ] && kill $HB_PID 2>/dev/null
  log "SELF_DESTROY $*"
  heartbeat destroyed "$*"
  upload_logs
  if [ "${ISG_NO_DESTROY:-0}" = 1 ]; then log "ISG_NO_DESTROY=1: staying up"; sleep infinity; fi
  # the key reaches curl on stdin (-K -), not on its command line
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    for i in 1 2 3 4 5 6; do
      code=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" | curl -s -K - -o /dev/null -w '%{http_code}' \
        -X DELETE "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/")
      log "destroy attempt $i: HTTP $code"
      [ "$code" = 200 ] && break
      sleep 20
    done
  fi
  sleep infinity
}
heartbeat boot

# 1. GPU: RT cores, driver 570-580 (the replay stack's tested range)
if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv,noheader > $LOG/gpus.txt 2>&1; then
  cat $LOG/gpus.txt; self_destroy "nvidia-smi failed"
fi
cat $LOG/gpus.txt
drv=$(head -1 $LOG/gpus.txt | awk -F', ' '{print $3}'); major=${drv%%.*}
[ "$major" -ge 570 ] 2>/dev/null && [ "$major" -le 580 ] || self_destroy "driver $drv outside 570-580"
if grep -E -i 'A100|H100|H200|V100|B200|A10G|T4' $LOG/gpus.txt > /dev/null; then self_destroy "GPU without RT cores"; fi
ldconfig 2>/dev/null
glx=$(ldconfig -p | grep -E 'libGLX_nvidia\.so\.0|libEGL_nvidia\.so\.0|libnvidia-glcore' | awk '{print $NF}' | tr '\n' ' ')
log "nvidia graphics libs: ${glx:-none}"
echo "$glx" | grep -q libGLX_nvidia && echo "$glx" | grep -q libEGL_nvidia || self_destroy "no NVIDIA GLX/EGL libraries"
cu1=$(ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')
if [ -n "$cu1" ] && [ ! -e "$(dirname "$cu1")/libcuda.so" ]; then
  ln -s "$(basename "$cu1")" "$(dirname "$cu1")/libcuda.so" && ldconfig 2>/dev/null && log "added libcuda.so -> $cu1"
fi

# 2. the bundles (boot_list.txt: "<s3 url> <dest dir>" per archive; 8 at a time, a stalled download retried)
fetch() {  # url dest
  mkdir -p "$2"
  for try in 1 2 3 4; do
    local f=/work/dl/$(echo "$1" | tr '/:' '__')
    mkdir -p /work/dl
    if timeout 600 s5cmd --endpoint-url $S3_ENDPOINT_URL --retry-count 20 cp -c 4 -p 64 "$1" "$f" > /dev/null \
        && case "$1" in *.zst) zstd -d -q -c "$f" | tar -x --no-same-owner -C "$2" ;; *) tar -x --no-same-owner -C "$2" -f "$f" ;; esac; then
      rm -f "$f"; log "fetched $1"; return 0
    fi
    rm -f "$f"
    log "FETCH_RETRY $1 (try $try)"
  done
  log "FETCH_FAILED $1"; return 1
}
export -f fetch log; export S3_ENDPOINT_URL t0
heartbeat boot fetch
( while sleep 60; do heartbeat boot "fetch/unpack +$(( SECONDS - t0 ))s disk_used=$(df -k --output=used /work | tail -1 | awk '{printf "%.1fGB", $1/1048576}')"; done ) &
HB_PID=$!
( sleep 3000; self_destroy "boot timeout after 50 min (slow host)" ) & WD_PID=$!
$S5 cp "$ISG_S3/bundle/boot_list.txt" /work/boot_list.txt > /dev/null || self_destroy "boot list fetch failed"
mkdir -p /work/data
# the BEHAVIOR asset key: straight to its private location (never in a bundle, a log or the image)
$S5 cp "$REPLAY_S3/secrets/omnigibson.key" /work/data/omnigibson.key > /dev/null 2>&1 || self_destroy "key fetch failed"
chmod 600 /work/data/omnigibson.key
xargs -a /work/boot_list.txt -P 8 -L 1 bash -c 'fetch "$0" "$1"' || self_destroy "bundle fetch failed"
# phase 1b (parallel): the cache bases, once the cluster says the bundle set is complete (bundle/boot_ready; boot_lists.sh)
for i in $(seq 1 80); do $S5 ls "$ISG_S3/bundle/boot_ready" > /dev/null 2>&1 && break; [ $i = 1 ] && log "waiting for bundle/boot_ready"; sleep 30; done
$S5 ls "$ISG_S3/bundle/boot_ready" > /dev/null 2>&1 || self_destroy "bundle/boot_ready never appeared"
if $S5 cp "$ISG_S3/bundle/boot_list1b.txt" /work/boot_list1b.txt > /dev/null 2>&1; then
  xargs -a /work/boot_list1b.txt -P 8 -L 1 bash -c 'fetch "$0" "$1"' || self_destroy "bundle fetch failed (1b)"
fi
# then, in order, what overrides phase 1 (the env delta over the replay env parts, the cache deltas over their bases)
if $S5 cp "$ISG_S3/bundle/boot_list2.txt" /work/boot_list2.txt > /dev/null 2>&1; then
  while read -r url dest; do [ -n "$url" ] && { fetch "$url" "$dest" || self_destroy "bundle fetch failed ($url)"; }; done < /work/boot_list2.txt
fi
rm -rf /work/dl

# 3. the cluster layout the job scripts expect
D=/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets
mkdir -p $D
for d in behavior-1k-assets omnigibson-robot-assets 2026-challenge-task-instances omnigibson.key; do ln -sfn /work/data/$d $D/$d; done
# texture store (KIT_TEX_STORE default of kit_appdata.sh) = the local store the scene packs unpack into; kit templates local
LC=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache
mkdir -p $LC /work/texstore/.scenes $LC/kit_templates
[ -e $LC/kit_textures ] || ln -s /work/texstore $LC/kit_textures
# conda: env_fast.sh sources miniconda3/etc/profile.d/conda.sh and activates the env; only the env prefix is shipped, so a stand-in
# sets what the cluster's `conda activate` sets (CONDA_*, PATH, the env's activate.d) - without isg_threads.sh, a no-op on the
# cluster (no CFS quota) that would pin every job here to CPUs 0..quota-1 (worker.py pins each job to CPUs of its own)
mkdir -p /vision/u/minyeong/miniconda3/etc/profile.d /vision/u/minyeong/miniconda3/condabin
cat > /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh <<'EOF'
# i7-vast stand-in for miniconda3's conda.sh on a Vast instance (activate only)
export CONDA_EXE=/vision/u/minyeong/miniconda3/bin/conda _CE_M= _CE_CONDA= CONDA_PYTHON_EXE=/vision/u/minyeong/miniconda3/bin/python
export _CONDA_ROOT=/vision/u/minyeong/miniconda3 _CONDA_EXE=/vision/u/minyeong/miniconda3/bin/conda
conda() {
  case "${1:-}" in
    activate)
      local p=/vision/u/minyeong/miniconda3/envs/${2:-base}
      [ -d "$p" ] || { echo "conda stand-in: no env $p" >&2; return 1; }
      export CONDA_PREFIX=$p CONDA_DEFAULT_ENV=${2:-base} CONDA_SHLVL=1 CONDA_PROMPT_MODIFIER="(${2:-base}) "
      export PATH=$p/bin:/vision/u/minyeong/miniconda3/condabin:$PATH
      local f
      for f in $p/etc/conda/activate.d/*.sh; do
        case "$(basename "$f")" in isg_threads.sh) continue ;; esac
        [ -f "$f" ] && . "$f"
      done ;;
    deactivate) ;;
    *) echo "conda stand-in: only activate" >&2; return 1 ;;
  esac
}
EOF
install -m 755 $RUN/git /usr/local/bin/git
touch /etc/isg_git_heads
# Kit app-data template for this GPU + driver + Behavior-1K 6a6194600: the replay's published ones (read only), else none (cold)
KEY=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9.\n' '_')_6a6194600
for src in "$REPLAY_S3/kit_tpl/$KEY/unit.tar.zst" "$REPLAY_S3/kit_tpl/$KEY/warm.tar.zst" "$REPLAY_S3/bundle/kit_templates/$KEY.tar.zst"; do
  if $S5 ls "$src" > /dev/null 2>&1 && fetch "$src" $LC/kit_templates/$KEY.tmp; then
    rm -f $LC/kit_templates/$KEY.tmp/.complete; touch $LC/kit_templates/$KEY.tmp/.complete; mv -T $LC/kit_templates/$KEY.tmp $LC/kit_templates/$KEY
    log "kit template $KEY from $src"; break
  fi
  rm -rf $LC/kit_templates/$KEY.tmp
done
[ -d $LC/kit_templates/$KEY ] || log "kit template: none for $KEY (cold Kit starts)"
# the texture pack every scene shares
[ -f /work/texstore/.scenes/common.complete ] || {
  if $S5 cp "$REPLAY_S3/bundle/textures/common/parts.txt" /work/tex_common.parts > /dev/null 2>&1; then
    for p in $(awk '{print $1}' /work/tex_common.parts); do fetch "$REPLAY_S3/bundle/textures/common/$p" /work/texstore || self_destroy "texture fetch failed"; done
    touch /work/texstore/.scenes/common.complete
  fi; }
log "bundle unpacked: env $(du -sh $ENVP | cut -f1), data $(du -sh -L /work/data/behavior-1k-assets | cut -f1), disk left $(df -h --output=avail /work | tail -1), tmp avail $(df -h --output=avail /tmp | tail -1)"

# 4. env self-test (a script file: Kit's crash reporter trips over newlines in python -c)
heartbeat boot selftest
export PATH=$ENVP/bin:$PATH PYTHONDONTWRITEBYTECODE=1
cat > /work/selftest.py <<'EOF'
import torch, h5py, numpy
assert torch.cuda.is_available(), "torch sees no CUDA device"
print("torch", torch.__version__, "cuda", torch.cuda.device_count(), torch.cuda.get_device_name(0), "h5py", h5py.__version__)
x = torch.randn(1024, 1024, device="cuda"); print("matmul", float((x @ x).sum()) == float((x @ x).sum()))
EOF
python /work/selftest.py > $LOG/selftest.log 2>&1 || { cat $LOG/selftest.log; self_destroy "env self-test failed"; }
for t in rsync taskset timeout git python3 awk find xargs hostname md5sum; do command -v $t > /dev/null || self_destroy "missing tool $t"; done
rsync --version | head -1 >> $LOG/selftest.log
cat $LOG/selftest.log
log "self-test ok"
kill $HB_PID $WD_PID 2>/dev/null
heartbeat ready
upload_logs
# 5. the worker (exit 0: stopped / idle; 3: unusable host)
python -u $RUN/worker.py >> $LOG/worker.log 2>&1
rc=$?
log "worker exited rc=$rc"
self_destroy "worker exit rc=$rc"
