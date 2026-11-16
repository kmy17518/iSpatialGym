#!/bin/bash
# Vast.ai instance boot of the 2026-11-16 policy eval (og-isgeval-; adapted from scripts/batched_replay/vast/boot.sh). Started by the
# image's entrypoint from /opt/isg/run: pull the boot list ($ISG_S3/bundle/boot_list.txt) onto the instance disk, check the GPUs and the
# env, then run eval_worker.py, which evaluates the units of run $ISG_RUN (runs/<run>/spec.json) until none is left. The instance
# destroys itself at the end (or when a check fails); logs go to $ISG_S3/runs/$ISG_RUN/logs/<instance>/.
set -u
RUN=/opt/isg/run
LOG=/var/log/isg
mkdir -p $LOG /work
exec > >(tee -a $LOG/boot.log) 2>&1
t0=$SECONDS
log() { echo "[$(date -u +%FT%TZ) +$(( SECONDS - t0 ))s] $*"; }
# worker id: the Vast instance id, or rp-<pod id> on RunPod (the second provider: same image, same R2 queue)
if [ -n "${CONTAINER_ID:-}" ]; then ISG_INSTANCE=$CONTAINER_ID
elif [ -n "${RUNPOD_POD_ID:-}" ]; then ISG_INSTANCE=rp-$RUNPOD_POD_ID
else ISG_INSTANCE=$(hostname); fi
export ISG_INSTANCE
export ENVP=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
S5="s5cmd --endpoint-url $S3_ENDPOINT_URL --numworkers 64"
# what a host OS has and a bare container lacks: a machine id and a UTF-8 locale (C.UTF-8 is built into glibc)
[ -s /etc/machine-id ] || cat /proc/sys/kernel/random/uuid | tr -d '-' > /etc/machine-id
mkdir -p /var/lib/dbus && { [ -e /var/lib/dbus/machine-id ] || cp /etc/machine-id /var/lib/dbus/machine-id; }
export LANG=C.UTF-8 LC_ALL=C.UTF-8
log "BOOT instance=$ISG_INSTANCE tag=${ISG_TAG:-} cpus=$(nproc) mem=$(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo) disk=$(df -h --output=avail /work | tail -1)"
log "env: $(for v in CONTAINER_ID CONTAINER_API_KEY RUNPOD_POD_ID RUNPOD_API_KEY; do printf '%s=%s ' $v "$([ -n "${!v:-}" ] && echo set || echo unset)"; done)"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null
shm_kb=$(df -k --output=size /dev/shm | tail -1)
[ "${shm_kb:-0}" -lt 8000000 ] && mount -o remount,size=16G /dev/shm 2>/dev/null && log "remounted /dev/shm at 16G"
log "shm $(df -h --output=size /dev/shm | tail -1) nofile $(ulimit -n) nproc $(ulimit -u) cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) mem.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"

RUNS3="$ISG_S3/runs/${ISG_RUN:?ISG_RUN}"
upload_logs() { $S5 cp "$LOG/*" "$RUNS3/logs/$ISG_INSTANCE/" > /dev/null 2>&1; }
heartbeat() {  # phase [detail]
  printf '{"instance": "%s", "tag": "%s", "phase": "%s", "detail": "%s", "time": %s}\n' "$ISG_INSTANCE" "${ISG_TAG:-}" "$1" "${2:-}" "$(date +%s)" \
    | $S5 pipe "$RUNS3/hb/$ISG_INSTANCE.json" > /dev/null 2>&1
}
self_destroy() {
  [ -n "${HB_PID:-}" ] && kill $HB_PID 2>/dev/null
  log "SELF_DESTROY $*"
  heartbeat destroyed "$*"
  upload_logs
  if [ "${ISG_NO_DESTROY:-0}" = 1 ]; then log "ISG_NO_DESTROY=1: staying up"; sleep infinity; fi
  # the keys reach curl on stdin (-K -), not on its command line
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    for i in 1 2 3 4 5 6; do
      code=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" | curl -s -K - -o /dev/null -w '%{http_code}' \
        -X DELETE "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/")
      log "destroy attempt $i: HTTP $code"
      [ "$code" = 200 ] && break
      sleep 20
    done
  elif [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    # RunPod puts a key scoped to this pod into its environment; podTerminate stops billing and deletes the pod
    for i in 1 2 3 4 5 6; do
      out=$(printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl -s -K - -H 'Content-Type: application/json' \
        --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" https://api.runpod.io/graphql)
      log "terminate attempt $i: ${out:0:200}"
      echo "$out" | grep -q '"podTerminate"' && ! echo "$out" | grep -q '"errors"' && break
      sleep 20
    done
  fi
  sleep infinity
}
heartbeat boot

# 1. GPUs: RT-core models only, driver 570-580 (or up to ISG_DRIVER_MAX: newer branches are being tested)
if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv,noheader > $LOG/gpus.txt 2>&1; then
  cat $LOG/gpus.txt; self_destroy "nvidia-smi failed"
fi
cat $LOG/gpus.txt
drv=$(head -1 $LOG/gpus.txt | awk -F', ' '{print $3}'); major=${drv%%.*}
# pi checkpoints serve with JAX 0.11 / CUDA 13: driver >= 580 (ISG_DRIVER_MIN)
[ "$major" -ge "${ISG_DRIVER_MIN:-570}" ] 2>/dev/null && [ "$major" -le "${ISG_DRIVER_MAX:-580}" ] || self_destroy "driver $drv outside ${ISG_DRIVER_MIN:-570}-${ISG_DRIVER_MAX:-580}"
if grep -E -i 'A100|H100|H200|V100|B200|A10G|T4' $LOG/gpus.txt > /dev/null; then self_destroy "GPU without RT cores"; fi
# Kit renders through Vulkan / EGL: the container toolkit must have injected the driver's graphics libraries
ldconfig 2>/dev/null
glx=$(ldconfig -p | grep -E 'libGLX_nvidia\.so\.0|libEGL_nvidia\.so\.0|libnvidia-glcore' | awk '{print $NF}' | tr '\n' ' ')
log "nvidia graphics libs: ${glx:-none} $(ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null | tr '\n' ' ')"
{ cat /proc/driver/nvidia/version; ls -la /dev/nvidia* /dev/dri 2>&1; ldconfig -p | grep -i -E 'nvidia|vulkan|libEGL|libGLX|libGL\.'; ulimit -a;
  grep -E '^Cap|^Seccomp' /proc/self/status; uname -a; cat /etc/os-release | head -2; env | grep -E '^NVIDIA|^VK_|^CUDA'
  echo "environment: names and value lengths"; python3 -c 1 2>/dev/null; awk 'BEGIN {for (k in ENVIRON) printf "%s %d\n", k, length(ENVIRON[k])}' | sort; } > $LOG/diag.txt 2>&1
echo "$glx" | grep -q libGLX_nvidia && echo "$glx" | grep -q libEGL_nvidia || self_destroy "no NVIDIA GLX/EGL libraries (NVIDIA_DRIVER_CAPABILITIES lacks graphics?)"
# triton links its CUDA helper with -lcuda: some hosts inject only libcuda.so.1 (ld: cannot find -lcuda)
cu1=$(ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')
if [ -n "$cu1" ] && [ ! -e "$(dirname "$cu1")/libcuda.so" ]; then
  ln -s "$(basename "$cu1")" "$(dirname "$cu1")/libcuda.so" && ldconfig 2>/dev/null && log "added libcuda.so -> $cu1"
fi

# 2. the bundle, 8 archives at a time (4 connections each): download (s5cmd cp, retried per part) -> zstd -> tar on the instance disk (mtimes
#    kept); a stalled download is killed after 5 min and retried (tar rewrites the same files)
fetch() {  # url-or-key dest
  mkdir -p "$2"
  local url=$1; case $url in s3://*) ;; *) url="$ISG_S3/$1" ;; esac
  for try in 1 2 3 4; do
    local f=/work/dl/$(echo "$1" | tr '/' '_')
    mkdir -p /work/dl
    if timeout 300 s5cmd --endpoint-url $S3_ENDPOINT_URL --retry-count 20 cp -c 4 -p 64 "$url" "$f" > /dev/null \
        && case "$1" in *.zst) zstd -d -q -c "$f" | tar -x --no-same-owner -C "$2" ;; *) tar -x --no-same-owner -C "$2" -f "$f" ;; esac; then
      rm -f "$f"; log "fetched $1"; return 0
    fi
    rm -f "$f"
    log "FETCH_RETRY $1 (try $try)"
  done
  log "FETCH_FAILED $1"; return 1
}
export -f fetch log; export ISG_S3 S3_ENDPOINT_URL t0
heartbeat boot fetch
( while sleep 60; do heartbeat boot "fetch/unpack +$(( SECONDS - t0 ))s disk_used=$(du -s /work $ENVP 2>/dev/null | awk '{s+=$1} END {printf "%.1fGB", s/1048576}')"; done ) &
HB_PID=$!
( sleep ${ISG_BOOT_TIMEOUT_S:-2700}; self_destroy "boot timeout after ${ISG_BOOT_TIMEOUT_S:-2700} s (slow host)" ) & WD_PID=$!
$S5 cp "$ISG_S3/bundle/boot_list.txt" /work/fetch_list.txt > /dev/null || self_destroy "boot list fetch failed"
mkdir -p /work/data
$S5 cp "${ISG_KEY_S3:?ISG_KEY_S3}" /work/data/omnigibson.key > /dev/null || self_destroy "key fetch failed"
chmod 600 /work/data/omnigibson.key
{ $S5 cp "$RUNS3/spec.json" /work/units.json; } > /dev/null || self_destroy "spec fetch failed"
# pre-converted Kit textures (read-only packs of the isg-1020-b1k replay): the robot's ("common") and house_single_floor's at
# boot; the worker fetches another scene's pack before its first unit (ensure_scene_textures)
for scene in common house_single_floor; do
  if $S5 cp "$ISG_TEX_S3/$scene/parts.txt" /work/tex_$scene.parts > /dev/null 2>&1; then
    awk -v s=$scene -v b="$ISG_TEX_S3" '{print b "/" s "/" $1, "/work/texstore"}' /work/tex_$scene.parts >> /work/fetch_list.txt
  fi
done
mkdir -p /work/texstore/.scenes
xargs -a /work/fetch_list.txt -P 8 -L 1 bash -c 'fetch "$0" "$1"' || self_destroy "bundle fetch failed"
for f in /work/tex_*.parts; do [ -f "$f" ] && { scene=${f#/work/tex_}; touch /work/texstore/.scenes/${scene%.parts}.complete; }; done
find /work/texstore -maxdepth 1 -type f -exec chmod 444 {} + 2>/dev/null
log "texture store: $(ls /work/texstore/.scenes 2>/dev/null | tr '\n' ' ') $(find /work/texstore -maxdepth 1 -name '*.ovtex.desc' | wc -l) textures"
log "bundle unpacked: env $(du -sh $ENVP | cut -f1), data $(du -sh /work/data | cut -f1), disk left $(df -h --output=avail /work | tail -1)"

# 3. the env: unpacked at its own prefix (nothing to relocate); conda-unpack only if the bundle came from conda-pack
heartbeat boot unpack
if [ -x $ENVP/bin/conda-unpack ]; then $ENVP/bin/conda-unpack > $LOG/conda_unpack.log 2>&1 || { tail $LOG/conda_unpack.log; self_destroy "conda-unpack failed"; }; fi
export PATH=$ENVP/bin:$PATH CONDA_PREFIX=$ENVP PYTHONDONTWRITEBYTECODE=1
python - <<'EOF' > $LOG/selftest.log 2>&1 || { cat $LOG/selftest.log; self_destroy "env self-test failed"; }
import torch, h5py, av, pyarrow, boto3, lerobot
assert torch.cuda.is_available(), "torch sees no CUDA device"
print("torch", torch.__version__, "cuda devices", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("lerobot", lerobot.__version__ if hasattr(lerobot, "__version__") else "?", "av", av.__version__, "boto3", boto3.__version__)
EOF
cat $LOG/selftest.log
log "self-test ok"

# 4. the worker (replays units until none is left; exit 0 = finished normally)
kill $HB_PID $WD_PID 2>/dev/null
heartbeat boot worker
python -u $RUN/eval_worker.py >> $LOG/worker.log 2>&1
rc=$?
log "worker exited rc=$rc"
[ $rc = 3 ] && self_destroy "unusable: worker exit rc=3"
self_destroy "worker exit rc=$rc"
