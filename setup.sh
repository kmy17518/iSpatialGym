#!/bin/bash
#
# iSpatialGym installation script (Linux).
#
# This is a wrapper around BEHAVIOR-1K's own setup.sh. For every flag it behaves
# like Behavior-1K/setup.sh, with two differences:
#
#   1. --primitives installs upstream NVlabs cuRobo v0.8.0 (cuRoboV2) instead of
#      the StanfordVL cuRobo fork (v1 API).
#   2. It repairs dependency gaps in the upstream install so it actually completes
#      end-to-end (see "core dependency remediation" below).
#
# Why a wrapper instead of editing Behavior-1K's sources:
#   * Keeps the Behavior-1K checkout pristine, so it stays in sync with upstream.
#   * cuRoboV2 installs very differently from the v1 fork: it does NOT compile
#     CUDA extensions at install time (it JIT-compiles at runtime via cuda.core),
#     so it does not need a system `nvcc`. We intercept `--primitives` so upstream
#     never installs the fork / runs its nvcc pre-check, then install cuRoboV2
#     ourselves with normal build isolation.
#
# Core dependency remediation:
#   Upstream OmniGibson imports `av` (PyAV) at import time (via omnigibson.eval /
#   lerobot.datasets), but `av` sits under lerobot's `av-dep` extra which nothing
#   requests, so a plain `--omnigibson` install cannot even `import omnigibson`
#   (which also aborts the dataset download). We therefore intercept `--dataset`,
#   install the missing core deps (with version pins that avoid breaking Isaac
#   Sim), then run the dataset download ourselves. (Since OmniGibson 3.9,
#   `lerobot[dataset]` is in install_requires, so `datasets`/`pandas` usually
#   arrive on their own — the pins below are kept as guards and are no-ops when
#   already satisfied.)
#
# IMPORTANT (scope): cuRoboV2 is a ground-up rewrite with a BREAKING public API.
# OmniGibson's motion-planning integration still targets the cuRobo v1 API, so
# this wrapper only guarantees that v0.8.0 *installs* cleanly. Running primitives
# (motion planning) at runtime requires a separate v1 -> v2 port and is NOT
# validated here.
#
set -euo pipefail

# --- cuRobo target (pinned to the v0.8.0 release) ---------------------------
CUROBO_REPO="https://github.com/NVlabs/curobo"
CUROBO_REF="v0.8.0"

# --- pinned remediation deps -------------------------------------------------
# These pins were validated against Isaac Sim 5.1 + OmniGibson 3.9 + cuRoboV2:
#   av      : >=15,<16   (matches lerobot's own av-dep range; av 16+ removed the
#                         `av.option` API that lerobot uses)
#   datasets: >=4,<5     (lerobot[dataset] range; normally already installed)
#   pillow  : ~=11.0.0   (OmniGibson 3.9's own pin; guard so nothing bumps it)
#   pandas  : <3         (lerobot[dataset] range; normally already installed)
#   numpy   : <2         (Isaac Sim / OmniGibson requirement)
#   packaging: <26       (cuRoboV2 pulls packaging 26, violating lerobot's <26)
#   warp-lang: >=0.9,<1.13 (member of OmniGibson's `primitives` extra, which we
#                         intercept, so we must install it ourselves)
AV_SPEC="av>=15.0.0,<16.0.0"
DATASETS_SPEC="datasets>=4.0.0,<5.0.0"
PILLOW_SPEC="pillow~=11.0.0"
PANDAS_SPEC="pandas<3"
NUMPY_SPEC="numpy<2"
PACKAGING_SPEC="packaging<26"
WARP_SPEC="warp-lang>=0.9.0,<1.13"

# --- locate (or clone) the bundled Behavior-1K checkout ----------------------
# Behavior-1K/ is a separate git checkout (gitignored here): our fork of
# StanfordVL/BEHAVIOR-1K on branch `ispatialgym` (LayoutReached/PoseReached/
# PoseSettled/Grasped states + the ispatialgym BDDL domain). If it is missing we
# clone it. Override with ISPATIALGYM_B1K_URL / ISPATIALGYM_B1K_BRANCH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B1K_DIR="$SCRIPT_DIR/Behavior-1K"
B1K_SETUP="$B1K_DIR/setup.sh"
B1K_URL="${ISPATIALGYM_B1K_URL:-git@github.com:kmy17518/BEHAVIOR-1K.git}"
B1K_URL_HTTPS="https://github.com/kmy17518/BEHAVIOR-1K.git"
B1K_BRANCH="${ISPATIALGYM_B1K_BRANCH:-ispatialgym}"

# Pre-converted custom assets (Kohs block `toy_dice/isgblk`) that OmniGibson
# loads as dataset `ispatialgym-assets` from gm.DATA_PATH.
ISG_ASSETS_SRC="$SCRIPT_DIR/ispatialgym/assets/datasets/ispatialgym-assets"

ensure_b1k_checkout() {
    if [ -f "$B1K_SETUP" ]; then
        return 0
    fi
    if [ -e "$B1K_DIR" ] && [ -n "$(ls -A "$B1K_DIR" 2>/dev/null)" ]; then
        echo "ERROR: $B1K_DIR exists but has no setup.sh (not a Behavior-1K checkout?)"
        exit 1
    fi
    command -v git >/dev/null || { echo "ERROR: git is required to clone Behavior-1K"; exit 1; }
    echo "==> Behavior-1K checkout not found; cloning $B1K_URL (branch $B1K_BRANCH) into $B1K_DIR"
    if ! git clone --branch "$B1K_BRANCH" "$B1K_URL" "$B1K_DIR"; then
        if [ "$B1K_URL" != "$B1K_URL_HTTPS" ]; then
            echo "WARNING: clone over $B1K_URL failed (no GitHub SSH key?); retrying over HTTPS: $B1K_URL_HTTPS"
            rm -rf "$B1K_DIR"
            git clone --branch "$B1K_BRANCH" "$B1K_URL_HTTPS" "$B1K_DIR"
            # keep pushes going to the SSH URL so `git push` works once a key is set up
            git -C "$B1K_DIR" remote set-url --push origin "$B1K_URL"
        else
            exit 1
        fi
    fi
    [ -f "$B1K_SETUP" ] || { echo "ERROR: clone succeeded but $B1K_SETUP is missing"; exit 1; }
}

# --- argument handling ------------------------------------------------------
# Forward every argument to Behavior-1K/setup.sh untouched EXCEPT the ones we
# handle ourselves: --primitives, --dataset, and --accept-dataset-tos. We also
# peek at a few flags (--new-env / --cuda-version / --omnigibson) needed by the
# post-install steps, while still forwarding them.
FORWARD_ARGS=()
PRIMITIVES=false
DATASET=false
OMNIGIBSON=false
NEW_ENV=false
NEW_ENV_NAME="behavior"
CUDA_VERSION="12.8"
ACCEPT_DATASET_TOS=false
SHOW_HELP=false

[ "$#" -eq 0 ] && SHOW_HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        --primitives)
            # Intercepted: handled after the upstream install (see below).
            PRIMITIVES=true
            shift
            ;;
        --dataset)
            # Intercepted: handled after core-dep remediation (see below).
            DATASET=true
            shift
            ;;
        --accept-dataset-tos)
            # Intercepted: only affects the (intercepted) dataset download.
            ACCEPT_DATASET_TOS=true
            shift
            ;;
        --omnigibson)
            OMNIGIBSON=true
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --new-env)
            NEW_ENV=true
            FORWARD_ARGS+=("$1")
            # Mirror upstream: `--new-env NAME` or bare `--new-env` (default name).
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                NEW_ENV_NAME="$2"
                FORWARD_ARGS+=("$2")
                shift 2
            else
                NEW_ENV_NAME="behavior"
                shift 1
            fi
            ;;
        --cuda-version)
            if [[ -n "${2:-}" ]]; then
                CUDA_VERSION="$2"
                FORWARD_ARGS+=("$1" "$2")
                shift 2
            else
                FORWARD_ARGS+=("$1")
                shift 1
            fi
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    cat << EOF
iSpatialGym Installation Script (Linux)
Usage: ./setup.sh [OPTIONS]

This wraps BEHAVIOR-1K's setup.sh and accepts all of its options. Differences:
  * Behavior-1K/ (our fork, branch '$B1K_BRANCH') is cloned from
    $B1K_URL if it is missing.
  * --primitives installs NVlabs cuRobo $CUROBO_REF (cuRoboV2) instead of the
    StanfordVL cuRobo fork.
  * Missing OmniGibson core deps (av, datasets, ...) are installed automatically
    so the install completes; --dataset is downloaded after that remediation.
  * --omnigibson also installs this repo (pip install -e .) and the custom
    'ispatialgym-assets' dataset (Kohs block) into OmniGibson's data path.

Recommended full install (the 'ispatialgym' conda env used for development):
  ./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --dev --primitives \\
             --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

NOTE: cuRoboV2 has a BREAKING API compared to the cuRobo v1 that OmniGibson's
primitives code targets; iSpatialGym's own planner (ispatialgym.oracle) uses the
v2 API directly, OmniGibson's built-in primitives do not work with it.

--- Upstream BEHAVIOR-1K options follow ---
EOF
    if [ -f "$B1K_SETUP" ]; then
        ( cd "$B1K_DIR" && bash setup.sh --help )
    else
        echo "(Behavior-1K not cloned yet; run without --help to clone it and see its options)"
    fi
    exit 0
fi

# Preserve upstream dependency rules for the flags we intercept.
if [ "$PRIMITIVES" = true ] && [ "$OMNIGIBSON" = false ]; then
    echo "ERROR: --primitives requires --omnigibson"
    exit 1
fi
if [ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ]; then
    echo "ERROR: --dataset requires --omnigibson"
    exit 1
fi

# Resolve the cuda.core runtime extra matching the requested CUDA major version.
CUDA_MAJOR="${CUDA_VERSION%%.*}"
case "$CUDA_MAJOR" in
    12) CUROBO_EXTRA="cu12" ;;
    13) CUROBO_EXTRA="cu13" ;;
    *)
        echo "WARNING: unrecognized CUDA major version '$CUDA_MAJOR'; defaulting cuRobo extra to cu12"
        CUROBO_EXTRA="cu12"
        ;;
esac

# Optional preview: set ISPATIALGYM_DRY_RUN=1 to print the resolved plan without
# running anything. Useful before kicking off a long, irreversible install.
if [ "${ISPATIALGYM_DRY_RUN:-0}" = "1" ]; then
    if [ ! -f "$B1K_SETUP" ]; then
        echo "[dry-run] would clone: git clone --branch $B1K_BRANCH $B1K_URL $B1K_DIR"
    fi
    echo "[dry-run] Forwarded to Behavior-1K/setup.sh: ${FORWARD_ARGS[*]:-(none)}"
    echo "[dry-run] omnigibson: $OMNIGIBSON | dataset intercepted: $DATASET | primitives intercepted: $PRIMITIVES"
    echo "[dry-run] new-env: $NEW_ENV (name: $NEW_ENV_NAME) | cuda: $CUDA_VERSION -> cuRobo extra: $CUROBO_EXTRA"
    if [ "$OMNIGIBSON" = true ]; then
        echo "[dry-run] would run: pip install \"$AV_SPEC\" \"$DATASETS_SPEC\" \"$PILLOW_SPEC\" \"$PANDAS_SPEC\" \"$NUMPY_SPEC\""
        echo "[dry-run] would run: pip install -e \"$SCRIPT_DIR\"  (ispatialgym package)"
        echo "[dry-run] would copy: $ISG_ASSETS_SRC -> \${OMNIGIBSON_DATA_PATH:-$B1K_DIR/datasets}/ispatialgym-assets"
    fi
    if [ "$DATASET" = true ]; then
        echo "[dry-run] would download: robot assets + behavior-1k assets (accept_license=$ACCEPT_DATASET_TOS) + 2026 challenge instances"
    fi
    if [ "$PRIMITIVES" = true ]; then
        echo "[dry-run] would run: pip install \"ninja~=1.13.0\" \"$WARP_SPEC\""
        echo "[dry-run] would run: SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=${CUROBO_REF#v} pip install \"nvidia-curobo[${CUROBO_EXTRA}] @ git+${CUROBO_REPO}@${CUROBO_REF}\" \"$NUMPY_SPEC\""
        echo "[dry-run] would run: pip install \"$PACKAGING_SPEC\""
    fi
    exit 0
fi

# Activate the same environment the upstream script installed into, so our pip /
# python calls land in the right place. (conda activate trips `set -u`.)
activate_target_env() {
    if [ "$NEW_ENV" = true ]; then
        command -v conda >/dev/null || { echo "ERROR: conda not found for post-install steps"; exit 1; }
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$NEW_ENV_NAME"
        set -u
    fi
}

# --- make sure the Behavior-1K checkout is there ----------------------------
ensure_b1k_checkout

# --- run the upstream installer (everything except dataset + primitives) -----
echo "==> Running BEHAVIOR-1K setup..."
( cd "$B1K_DIR" && bash setup.sh "${FORWARD_ARGS[@]}" )

# --- core dependency remediation --------------------------------------------
# OmniGibson cannot `import omnigibson` without these (they are missing from its
# install_requires). Pins avoid the broken latest versions (see header).
if [ "$OMNIGIBSON" = true ]; then
    echo ""
    echo "==> Repairing OmniGibson core dependencies (av/datasets/pillow/pandas)..."
    activate_target_env
    pip install "$AV_SPEC" "$DATASETS_SPEC" "$PILLOW_SPEC" "$PANDAS_SPEC" "$NUMPY_SPEC"
    python -c "import omnigibson" >/dev/null 2>&1 \
        && echo "✓ omnigibson imports cleanly" \
        || { echo "ERROR: omnigibson still fails to import after remediation"; exit 1; }

    # --- this repo: editable install of the `ispatialgym` package -----------
    echo ""
    echo "==> Installing iSpatialGym (pip install -e $SCRIPT_DIR)..."
    pip install -e "$SCRIPT_DIR"
    python -c "import ispatialgym, ispatialgym.sampling" >/dev/null 2>&1 \
        && echo "✓ ispatialgym imports cleanly" \
        || { echo "ERROR: ispatialgym fails to import"; exit 1; }

    # --- custom assets: dataset `ispatialgym-assets` (Kohs block) -----------
    # Pre-converted USD (ispatialgym/assets/objects/kohs_block/convert_to_omnigibson.py
    # output) so a fresh machine does not have to run the Isaac Sim conversion.
    DATA_PATH="${OMNIGIBSON_DATA_PATH:-$B1K_DIR/datasets}"
    if [ -d "$ISG_ASSETS_SRC" ]; then
        echo ""
        echo "==> Installing custom dataset ispatialgym-assets -> $DATA_PATH/ispatialgym-assets"
        mkdir -p "$DATA_PATH/ispatialgym-assets"
        cp -r "$ISG_ASSETS_SRC/." "$DATA_PATH/ispatialgym-assets/"
        echo "✓ ispatialgym-assets installed"
    else
        echo "WARNING: $ISG_ASSETS_SRC not found; run ispatialgym/assets/objects/kohs_block/convert_to_omnigibson.py later"
    fi
fi

# --- dataset download -------------------------------------------------------
if [ "$DATASET" = true ]; then
    echo ""
    echo "==> Downloading BEHAVIOR datasets..."
    activate_target_env
    export OMNI_KIT_ACCEPT_EULA=YES
    ACCEPT_FLAG=$([ "$ACCEPT_DATASET_TOS" = true ] && echo "True" || echo "False")
    python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
    python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=${ACCEPT_FLAG})"
    python -c "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances; download_2026_challenge_task_instances()"
fi

# --- install cuRoboV2 in place of the fork ----------------------------------
if [ "$PRIMITIVES" = true ]; then
    echo ""
    echo "==> Installing motion-planning backend: NVlabs cuRobo $CUROBO_REF (cuRoboV2)"
    activate_target_env

    # `ninja` and `warp-lang` are the other members of OmniGibson's `primitives`
    # extra (which upstream never installs because we intercept the flag).
    pip install "ninja~=1.13.0" "$WARP_SPEC"

    # cuRoboV2 derives its version via setuptools_scm. pip's VCS checkout can be
    # shallow and miss the tag, which breaks version detection, so pin it
    # explicitly. DEFAULT build isolation (no --no-build-isolation) lets pip
    # provision setuptools_scm/wheel for the build. numpy<2 guards the env.
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${CUROBO_REF#v}" \
        pip install "nvidia-curobo[${CUROBO_EXTRA}] @ git+${CUROBO_REPO}@${CUROBO_REF}" "$NUMPY_SPEC"

    # cuRoboV2's deps bump `packaging` to 26+, which violates lerobot's <26.
    # Restore a compatible version (curobo only needs packaging at build time).
    pip install "$PACKAGING_SPEC"

    echo "==> cuRobo $CUROBO_REF installed (extra: ${CUROBO_EXTRA})."
fi

# --- summary ----------------------------------------------------------------
echo ""
echo "=== iSpatialGym setup complete ==="
[ "$OMNIGIBSON" = true ] && echo "✓ OmniGibson core deps repaired (av/datasets/pillow/pandas/numpy)"
[ "$OMNIGIBSON" = true ] && echo "✓ ispatialgym installed (editable) + ispatialgym-assets dataset"
[ "$DATASET" = true ] && echo "✓ Downloaded BEHAVIOR datasets"
if [ "$PRIMITIVES" = true ]; then
    echo "✓ Installed NVlabs cuRobo $CUROBO_REF (cuRoboV2)"
    echo ""
    echo "Reminder: OmniGibson's primitives target the cuRobo v1 API. Runtime"
    echo "motion planning will NOT work until the integration is ported to v2."
fi
