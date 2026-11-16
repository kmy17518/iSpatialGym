#!/bin/bash
# Build the serving venvs of the isg-train224 policies (pi05 / ACT / DP) in their checkouts under $Q
# (default /vision/u/minyeong/projects/isg-train224-eval, git worktrees at the branch tips), as the training hosts did:
#   openpi: scripts/b1k/setup_venv_jax.sh (.venv-jax, CPython 3.12, JAX 0.11.1 cuda13)
#   act:    scripts/b1k/setup_venv.sh (.venv, exact lock, CPython 3.10)
#   dp:     CPython 3.11, torch 2.10 cu130, requirements-b1k.txt, -e . (.venv)
# Run on a CPU node (sbatch -c 16); caches go to node-local /tmp.
set -uo pipefail
Q=${Q:-/vision/u/minyeong/projects/isg-train224-eval}
export PATH=$HOME/.local/bin:$PATH
T=/tmp/$USER/isgeval_venv_${SLURM_JOB_ID:-local}
mkdir -p $T
export CMAKE_POLICY_VERSION_MINIMUM=3.5 UV_CACHE_DIR=$T/uv_cache UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$Q/.uv_python} TMPDIR=$T UV_LINK_MODE=copy
which=${WHICH:-openpi act dp}
for w in $which; do
  t0=$SECONDS
  case $w in
    openpi) (cd $Q/openpi && bash scripts/b1k/setup_venv_jax.sh) > $Q/build_openpi.log 2>&1 ;;
    act) (cd $Q/act && PYTHON=3.10 bash scripts/b1k/setup_venv.sh) > $Q/build_act.log 2>&1 ;;
    dp) (cd $Q/diffusion_policy && uv venv --allow-existing --python 3.11 .venv \
          && uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision==0.25.0 \
          && uv pip install --python .venv/bin/python -r requirements-b1k.txt && uv pip install --python .venv/bin/python -e .) > $Q/build_dp.log 2>&1 ;;
  esac
  echo "BUILD $w rc=$? $((SECONDS - t0)) s"
done
rm -rf $T
