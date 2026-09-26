#!/bin/bash
# Pod-side bootstrap for iSpatialGym worker pods (RunPod PyTorch image, Ubuntu 24.04 / CUDA 12.8).
# Mirrors docs/runpod_setup.md sections 3-5, unattended. Environment (passed by scripts/runpod/rp.py bootstrap):
#   GH_TOKEN          GitHub token with repo read access (private repo clone over HTTPS; not written to disk)
#   HF_TOKEN          Hugging Face token (BEHAVIOR asset download; kept in the environment only)
#   ISG_TTL_SECONDS   self-terminate after this many seconds (secondary guard; the local watchdog is the primary one)
#   ISG_JOB           optional job to run after the install, e.g. "bench:2" (see bench.sh)
#   ISG_REPO_REF      git ref of kmy17518/iSpatialGym to check out (default main)
# Markers: /root/BOOTSTRAP_DONE (install + verify ok), /root/BOOTSTRAP_FAILED, /root/JOB_DONE.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/root/isg_bootstrap.log
T0=$(date +%s)
phase() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }

# single instance: a repeated launch (retry, harness re-run) must not start a second install
exec 9>/root/isg_boot/.lock
if ! flock -n 9; then echo "another bootstrap is running; exiting"; exit 0; fi
[ -f /root/BOOTSTRAP_DONE ] && [ -z "${ISG_FORCE:-}" ] && { echo "bootstrap already done"; exit 0; }

# secrets / parameters from the root-only env file written over stdin by rp.py (never on a command line)
if [ -f /root/isg_boot/env.sh ]; then
  set -a; source /root/isg_boot/env.sh; set +a
fi

# 0. secondary self-terminate guard (runpodctl is preinstalled on RunPod images). SSH sessions do not inherit the
#    container's environment, so import the RUNPOD_* variables (pod id, pod-scoped API key) from PID 1.
while IFS= read -r -d '' kv; do
  case "$kv" in RUNPOD_*) export "$kv";; esac
done < /proc/1/environ
if [ -n "${ISG_TTL_SECONDS:-}" ] && command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  nohup bash -c "sleep ${ISG_TTL_SECONDS}; runpodctl remove pod ${RUNPOD_POD_ID}" >/root/isg_selfterminate.log 2>&1 &
  phase "self-terminate armed in ${ISG_TTL_SECONDS}s (pod ${RUNPOD_POD_ID})"
else
  phase "WARNING: self-terminate not armed (runpodctl=$(command -v runpodctl || echo none), pod=${RUNPOD_POD_ID:-none})"
fi

# 1. system libs Isaac Sim needs on this image
phase "apt"
apt-get update -qq >/dev/null && apt-get install -y -qq libegl1 libglu1-mesa libxt6 time rsync >/dev/null || { phase "apt failed"; touch /root/BOOTSTRAP_FAILED; exit 1; }

# 2. Miniforge on the local disk
if [ ! -d /root/miniforge3 ]; then
  phase "miniforge"
  curl -fsSL --retry 3 --max-time 600 -o /root/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    || { phase "miniforge download failed"; touch /root/BOOTSTRAP_FAILED; exit 1; }
  bash /root/miniforge.sh -b -p /root/miniforge3 >/dev/null
fi
source /root/miniforge3/etc/profile.d/conda.sh

# 3. repo (shallow clone over HTTPS with the token; the token is removed from the remote URL right after)
if [ ! -d /root/ispatialgym/.git ]; then
  phase "clone"
  git clone -q --depth 1 --branch "${ISG_REPO_REF:-main}" "https://x-access-token:${GH_TOKEN}@github.com/kmy17518/iSpatialGym.git" /root/ispatialgym \
    || { phase "clone failed"; touch /root/BOOTSTRAP_FAILED; exit 1; }
  git -C /root/ispatialgym remote set-url origin https://github.com/kmy17518/iSpatialGym.git
fi
cd /root/ispatialgym
# a leftover empty / partial Behavior-1K directory (interrupted earlier run) makes setup.sh refuse to clone it
if [ -d Behavior-1K ] && [ ! -f Behavior-1K/setup.sh ]; then rm -rf Behavior-1K; fi

# 4. install (conda env, OmniGibson + Isaac Sim, datasets, cuRoboV2)
if [ ! -f /root/INSTALL_DONE ]; then
  phase "setup.sh"
  ./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --dev --primitives \
      --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos > /root/isg_install.log 2>&1 \
    || { phase "setup.sh failed (see /root/isg_install.log)"; tail -30 /root/isg_install.log; touch /root/BOOTSTRAP_FAILED; exit 1; }
  touch /root/INSTALL_DONE
fi
# tokens are no longer needed once the repo is cloned and the datasets are downloaded
[ -f /root/isg_boot/env.sh ] && shred -u /root/isg_boot/env.sh 2>/dev/null
unset GH_TOKEN HF_TOKEN

# 5. verify
phase "verify"
conda activate ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
python -c "import curobo, os; assert os.path.exists(os.path.join(os.path.dirname(curobo.__file__), 'content/configs/task/ik/lbfgs_ik.yml')); print('curobo', curobo.__version__)" \
  || { phase "curobo verify failed"; touch /root/BOOTSTRAP_FAILED; exit 1; }
python -c "from ispatialgym.sampling import grasp_library as GL; print(len(GL.load_grasps('pack_of_kielbasa__dgodnn__t')), 'grasps')" \
  || { phase "grasp library verify failed"; touch /root/BOOTSTRAP_FAILED; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 5b. thread cap (owner 2026-09-25, docs/runpod_operations.md §9): the container runs under a CFS CPU quota (10-27 CPUs) but sees every
# host CPU, so torch / numba / BLAS / Kit size their pools to the host and the container is throttled in 57-100 % of its 100-ms
# windows; pinning to floor(quota) CPUs + library caps: 0 % throttling, instance load 570 -> 331 s, sim step 0.59 -> 0.25 s/step
# (runlogs/runpod/artic_integration/threads_ab/README.md). Installed as a conda activation hook so every `conda activate ispatialgym`
# (remote_run runs it before each job) exports the caps, pins the shell and its children and sets ISG_THREAD_CAP; written atomically.
phase "thread cap"
_hookdir=/root/miniforge3/envs/ispatialgym/etc/conda/activate.d
mkdir -p "$_hookdir"
cat > "$_hookdir/.isg_threads.sh.tmp" <<'EOF_ISG_THREADS'
# isg_threads.sh - cap the thread pools and pin the CPUs to the container's CFS quota (does nothing without a quota)
_isg_q=""; _isg_p=""
if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
  _isg_q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); _isg_p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)
elif [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r _isg_q _isg_p < /sys/fs/cgroup/cpu.max
fi
if [ -n "$_isg_q" ] && [ "$_isg_q" != "-1" ] && [ "$_isg_q" != "max" ] && [ -n "$_isg_p" ] && [ "$_isg_p" -gt 0 ] 2>/dev/null; then
  _isg_n=$(( _isg_q / _isg_p )); [ "$_isg_n" -lt 1 ] && _isg_n=1
  _isg_t=$(( _isg_n - 1 )); [ "$_isg_t" -lt 1 ] && _isg_t=1
  export OMP_NUM_THREADS=$_isg_t MKL_NUM_THREADS=$_isg_t OPENBLAS_NUM_THREADS=$_isg_t NUMBA_NUM_THREADS=$_isg_t
  taskset -cp 0-$(( _isg_n - 1 )) $$ > /dev/null 2>&1
  export ISG_THREAD_CAP="cpus=0-$(( _isg_n - 1 )) threads=$_isg_t quota=$(awk -v q="$_isg_q" -v p="$_isg_p" 'BEGIN{printf "%.1f", q/p}')"
fi
unset _isg_q _isg_p _isg_n _isg_t
EOF_ISG_THREADS
mv -f "$_hookdir/.isg_threads.sh.tmp" "$_hookdir/isg_threads.sh"
chmod 644 "$_hookdir/isg_threads.sh"
_cap_check=$(bash -c 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && echo "${ISG_THREAD_CAP:-no quota}"')
phase "thread cap: ${_cap_check}"
phase "install ok"
touch /root/BOOTSTRAP_DONE

# 6. optional job
if [ -n "${ISG_JOB:-}" ]; then
  case "${ISG_JOB}" in
    bench:*) K="${ISG_JOB#bench:}"; phase "job bench K=${K}"; bash /root/isg_boot/bench.sh "${K}" > /root/isg_bench.log 2>&1; phase "bench exit $?";;
    inv:*)   S="${ISG_JOB#inv:}"; phase "job scene inventory ${S}"; mkdir -p /root/inv
             ( cd /root/ispatialgym && python -m ispatialgym.sampling.scene_inventory --scene "${S}" --out /root/inv ) > /root/isg_inv.log 2>&1; phase "inventory exit $?";;
    rv:*)  # eef reach volume shards: rv:<arm>:<translate,translate,...>  (all tilts of the grid)
      spec="${ISG_JOB#rv:}"; arm="${spec%%:*}"; trs="${spec##*:}"; phase "job reach volume arm=${arm} translates=${trs}"
      python -m ispatialgym.sampling.eef_reach_volume compute --arm "${arm}" --translates ${trs//,/ } --out /root/rv > /root/isg_rv.log 2>&1
      phase "reach volume exit $?"; grep -E "shard |REACH_VOLUME_DONE" /root/isg_rv.log | tail -25;;
    *) phase "unknown job ${ISG_JOB}";;
  esac
  touch /root/JOB_DONE
fi
phase "done"
