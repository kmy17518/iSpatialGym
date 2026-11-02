#!/bin/bash
# Vast.ai instance boot for the vectorized ISG eval (started by the image's entrypoint from /opt/isg/run): the replay bundle
# (env, Behavior-1K 6a6194600, assets, textures) from $ISG_BUNDLE_S3 and the eval bundle (code, datasets, instances, goals,
# ACT code) from $ISG_S3 onto the instance disk, GPU / env checks, then eval_worker.py until no unit is left. The instance
# destroys itself at the end (or when a check fails); logs go to $ISG_S3/logs/<instance>/.
set -u
RUN=/opt/isg/run
LOG=/var/log/isg
mkdir -p $LOG /work
exec > >(tee -a $LOG/boot.log) 2>&1
t0=$SECONDS
log() { echo "[$(date -u +%FT%TZ) +$(( SECONDS - t0 ))s] $*"; }
if [ -n "${CONTAINER_ID:-}" ]; then ISG_INSTANCE=$CONTAINER_ID
elif [ -n "${RUNPOD_POD_ID:-}" ]; then ISG_INSTANCE=rp-$RUNPOD_POD_ID
else ISG_INSTANCE=$(hostname); fi
export ISG_INSTANCE
export ENVP=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs
export ISG_BUNDLE_S3=${ISG_BUNDLE_S3:-s3://isg-demos/isg-1020-b1k}
S5="s5cmd --endpoint-url $S3_ENDPOINT_URL --numworkers 64"
[ -s /etc/machine-id ] || cat /proc/sys/kernel/random/uuid | tr -d '-' > /etc/machine-id
mkdir -p /var/lib/dbus && { [ -e /var/lib/dbus/machine-id ] || cp /etc/machine-id /var/lib/dbus/machine-id; }
export LANG=C.UTF-8 LC_ALL=C.UTF-8
log "BOOT instance=$ISG_INSTANCE tag=${ISG_TAG:-} cpus=$(nproc) mem=$(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo) disk=$(df -h --output=avail /work | tail -1)"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null
shm_kb=$(df -k --output=size /dev/shm | tail -1)
[ "${shm_kb:-0}" -lt 8000000 ] && mount -o remount,size=16G /dev/shm 2>/dev/null && log "remounted /dev/shm at 16G"

upload_logs() { $S5 cp "$LOG/*" "$ISG_S3/logs/$ISG_INSTANCE/" > /dev/null 2>&1; }
heartbeat() {
  printf '{"instance": "%s", "tag": "%s", "phase": "%s", "detail": "%s", "time": %s}\n' "$ISG_INSTANCE" "${ISG_TAG:-}" "$1" "${2:-}" "$(date +%s)" \
    | $S5 pipe "$ISG_S3/hb/$ISG_INSTANCE.json" > /dev/null 2>&1
}
self_destroy() {
  [ -n "${HB_PID:-}" ] && kill $HB_PID 2>/dev/null
  log "SELF_DESTROY $*"
  heartbeat destroyed "$*"
  upload_logs
  if [ "${ISG_NO_DESTROY:-0}" = 1 ]; then log "ISG_NO_DESTROY=1: staying up"; sleep infinity; fi
  if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    for i in 1 2 3 4 5 6; do
      code=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" | curl -s -K - -o /dev/null -w '%{http_code}' \
        -X DELETE "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/")
      log "destroy attempt $i: HTTP $code"
      [ "$code" = 200 ] && break
      sleep 20
    done
  elif [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
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

# 1. GPUs: RT-core models only, driver 570-580
if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv,noheader > $LOG/gpus.txt 2>&1; then
  cat $LOG/gpus.txt; self_destroy "nvidia-smi failed"
fi
cat $LOG/gpus.txt
drv=$(head -1 $LOG/gpus.txt | awk -F', ' '{print $3}'); major=${drv%%.*}
[ "$major" -ge 570 ] 2>/dev/null && [ "$major" -le "${ISG_DRIVER_MAX:-580}" ] || self_destroy "driver $drv outside 570-${ISG_DRIVER_MAX:-580}"
if grep -E -i 'A100|H100|H200|V100|B200|A10G|T4' $LOG/gpus.txt > /dev/null; then self_destroy "GPU without RT cores"; fi
ldconfig 2>/dev/null
glx=$(ldconfig -p | grep -E 'libGLX_nvidia\.so\.0|libEGL_nvidia\.so\.0|libnvidia-glcore' | awk '{print $NF}' | tr '\n' ' ')
log "nvidia graphics libs: ${glx:-none}"
echo "$glx" | grep -q libGLX_nvidia && echo "$glx" | grep -q libEGL_nvidia || self_destroy "no NVIDIA GLX/EGL libraries (NVIDIA_DRIVER_CAPABILITIES lacks graphics?)"
cu1=$(ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')
if [ -n "$cu1" ] && [ ! -e "$(dirname "$cu1")/libcuda.so" ]; then
  ln -s "$(basename "$cu1")" "$(dirname "$cu1")/libcuda.so" && ldconfig 2>/dev/null && log "added libcuda.so -> $cu1"
fi

# 2. the bundles
fetch() {  # url dest
  mkdir -p "$2"
  for try in 1 2 3 4; do
    local f=/work/dl/$(echo "$1" | tr '/:' '__')
    mkdir -p /work/dl
    if timeout 420 s5cmd --endpoint-url $S3_ENDPOINT_URL --retry-count 20 cp -c 4 -p 64 "$1" "$f" > /dev/null \
        && case "$1" in *.zst) zstd -d -q -c "$f" | tar -x --no-same-owner -C "$2" ;; *) tar -x --no-same-owner -C "$2" -f "$f" ;; esac; then
      rm -f "$f"; log "fetched $1"; return 0
    fi
    rm -f "$f"
    log "FETCH_RETRY $1 (try $try)"
  done
  log "FETCH_FAILED $1"; return 1
}
export -f fetch log; export ISG_S3 ISG_BUNDLE_S3 S3_ENDPOINT_URL t0
heartbeat boot fetch
( while sleep 60; do heartbeat boot "fetch/unpack +$(( SECONDS - t0 ))s disk_used=$(du -s /work $ENVP 2>/dev/null | awk '{s+=$1} END {printf "%.1fGB", s/1048576}')"; done ) &
HB_PID=$!
( sleep 2400; self_destroy "boot timeout after 40 min (slow host)" ) & WD_PID=$!
$S5 cp "$ISG_BUNDLE_S3/bundle/env/parts.txt" /work/env_parts.txt > /dev/null || self_destroy "env part list fetch failed"
mkdir -p /work/data
$S5 cp "$ISG_BUNDLE_S3/secrets/omnigibson.key" /work/data/omnigibson.key > /dev/null || self_destroy "key fetch failed"
chmod 600 /work/data/omnigibson.key
$S5 cp "$ISG_S3/units.json" /work/units.json > /dev/null || self_destroy "units fetch failed"
{
  for part in $(cat /work/env_parts.txt); do echo "$ISG_BUNDLE_S3/bundle/env/$part $ENVP"; done
  echo "$ISG_BUNDLE_S3/bundle/b1k_code.tar.zst /work/b1k"
  echo "$ISG_BUNDLE_S3/bundle/assets/b1k_base.tar.zst /work/data/behavior-1k-assets"
  for k in 0 1 2 3 4 5; do echo "$ISG_BUNDLE_S3/bundle/assets/b1k_objects_$k.tar.zst /work/data/behavior-1k-assets"; done
  echo "$ISG_BUNDLE_S3/bundle/assets/robot.tar.zst /work/data"
  echo "$ISG_BUNDLE_S3/bundle/assets/task_instances.tar.zst /work/data"
  echo "$ISG_S3/bundle/eval.tar.zst /"
  for scene in ${ISG_PREFETCH_SCENES:-house_single_floor common}; do
    if $S5 cp "$ISG_BUNDLE_S3/bundle/textures/$scene/parts.txt" /work/tex_$scene.parts > /dev/null 2>&1; then
      awk -v s=$scene -v b=$ISG_BUNDLE_S3 '{print b "/bundle/textures/" s "/" $1, "/work/texstore"}' /work/tex_$scene.parts
    fi
  done
} > /work/fetch_list.txt
mkdir -p /work/texstore/.scenes
xargs -a /work/fetch_list.txt -P 8 -L 1 bash -c 'fetch "$0" "$1"' || self_destroy "bundle fetch failed"
for scene in ${ISG_PREFETCH_SCENES:-house_single_floor common}; do
  [ -f /work/tex_$scene.parts ] && touch /work/texstore/.scenes/$scene.complete
done
find /work/texstore -maxdepth 1 -type f -exec chmod 444 {} + 2>/dev/null
# the eval checkout's Behavior-1K: the bundle's code and data at the paths the checkout expects
E=/vision/u/minyeong/projects/ispatialgym-eval-vec
rm -rf $E/Behavior-1K; mkdir -p $E/Behavior-1K/datasets
for d in OmniGibson bddl3 docs; do ln -s /work/b1k/$d $E/Behavior-1K/$d; done
for d in behavior-1k-assets omnigibson-robot-assets 2026-challenge-task-instances omnigibson.key; do ln -s /work/data/$d $E/Behavior-1K/datasets/$d; done
ln -s $E/ispatialgym/assets/datasets/ispatialgym-assets $E/Behavior-1K/datasets/ispatialgym-assets
for d in ispatialgym-eval-instances ispatialgym-eval-demos; do ln -s /vision/u/minyeong/projects/ispatialgym-eval/Behavior-1K/datasets/$d $E/Behavior-1K/datasets/$d; done
# the vec-eval overlay on Behavior-1K 6a6194600 (AttachedTo's stop callbacks keyed by scene)
bash $E/scripts/vec_eval/apply_b1k_overlay.sh /work/b1k
log "bundle unpacked: env $(du -sh $ENVP | cut -f1), data $(du -sh /work/data | cut -f1), eval $(du -sh $E | cut -f1), disk left $(df -h --output=avail /work | tail -1)"

# 3. the env self-test (msgpack is not in the replay env: the wheel rides in the eval bundle)
heartbeat boot unpack
export PATH=$ENVP/bin:$PATH CONDA_PREFIX=$ENVP PYTHONDONTWRITEBYTECODE=1
python -c "import msgpack" 2>/dev/null || pip install -q --no-deps --no-index /vision/u/minyeong/projects/isg_home/vec_eval85/fleet/wheels/msgpack*.whl || self_destroy "msgpack install failed"
python - <<'EOF' > $LOG/selftest.log 2>&1 || { cat $LOG/selftest.log; self_destroy "env self-test failed"; }
import torch, h5py, av, boto3, msgpack, websockets, cv2
assert torch.cuda.is_available(), "torch sees no CUDA device"
print("torch", torch.__version__, "cuda devices", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("av", av.__version__, "msgpack", msgpack.version, "websockets", websockets.__version__, "cv2", cv2.__version__)
EOF
cat $LOG/selftest.log
log "self-test ok"
# 3b. Hugging Face must be reachable (the checkpoints come from there; one 8-GPU host could not download anything, 2026-10-23)
HF_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 60 -L -r 0-1023 "https://huggingface.co/kmy17518/act-isg1020-objscale-bs1024-s0/resolve/main/.b1k-uploader.json" || echo 000)
case "$HF_CODE" in 200|206) log "huggingface reachable ($HF_CODE)";; *) self_destroy "huggingface unreachable (http $HF_CODE)";; esac

# 4. the worker
kill $HB_PID $WD_PID 2>/dev/null
heartbeat boot worker
python -u $RUN/eval_worker.py >> $LOG/worker.log 2>&1
rc=$?
log "worker exited rc=$rc"
self_destroy "worker exit rc=$rc"
