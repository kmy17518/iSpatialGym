#!/bin/bash
#
# iSpatialGym installation (Linux). Wraps Behavior-1K/setup.sh (a git submodule) and takes all of its options, with two
# additions:
#   --omnigibson  also repairs the OmniGibson dependencies its install leaves out and installs this package (editable);
#   --dataset     also installs the iSpatialGym custom objects and downloads the eval instances from Hugging Face.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B1K_DIR="$SCRIPT_DIR/Behavior-1K"

# OmniGibson imports Warp (declared only in its "primitives" extra) and PyAV through lerobot (not declared at all); the
# other pins keep the stack on the NumPy 1 and Pillow 11 that Isaac Sim requires (pyarrow 26 needs NumPy 2).
REPAIR_SPECS=("warp-lang==1.12.0" "av>=15.0.0,<16.0.0" "datasets>=4.0.0,<5.0.0" "pyarrow<26" "pillow~=11.0.0"
              "pandas<3" "numpy<2")

FORWARD_ARGS=()
OMNIGIBSON=false
DATASET=false
ACCEPT_DATASET_TOS=false
NEW_ENV=false
NEW_ENV_NAME="behavior"
SHOW_HELP=false
[ "$#" -eq 0 ] && SHOW_HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --dataset) DATASET=true; shift ;;
        --accept-dataset-tos) ACCEPT_DATASET_TOS=true; shift ;;
        --omnigibson) OMNIGIBSON=true; FORWARD_ARGS+=("$1"); shift ;;
        --new-env)
            NEW_ENV=true
            FORWARD_ARGS+=("$1")
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then
                NEW_ENV_NAME="$2"
                FORWARD_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "$B1K_DIR/setup.sh" ]; then
    echo "==> Initializing the Behavior-1K submodule"
    git -C "$SCRIPT_DIR" submodule update --init Behavior-1K
fi

if [ "$SHOW_HELP" = true ]; then
    cat << EOF
iSpatialGym installation (Linux)
Usage: ./setup.sh [OPTIONS]

Takes every option of Behavior-1K/setup.sh (listed below). In addition:
  --omnigibson   also repairs OmniGibson's missing dependencies and installs iSpatialGym (pip install -e .)
  --dataset      also installs the iSpatialGym custom objects and downloads the eval instances (Hugging Face)

Recommended:
  ./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset \\
             --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

--- Behavior-1K options ---
EOF
    (cd "$B1K_DIR" && bash setup.sh --help)
    exit 0
fi

if [ "$DATASET" = true ] && [ "$OMNIGIBSON" = false ]; then
    echo "ERROR: --dataset requires --omnigibson"
    exit 1
fi

activate_env() {
    if [ "$NEW_ENV" = true ]; then
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$NEW_ENV_NAME"
        set -u
    fi
}

echo "==> Running Behavior-1K setup"
(cd "$B1K_DIR" && bash setup.sh "${FORWARD_ARGS[@]}")

if [ "$OMNIGIBSON" = true ]; then
    echo "==> Repairing OmniGibson dependencies and installing iSpatialGym"
    activate_env
    pip install "${REPAIR_SPECS[@]}"
    python -c "import omnigibson" || { echo "ERROR: omnigibson fails to import"; exit 1; }
    pip install -e "$SCRIPT_DIR"
    python -c "import ispatialgym.run_isg" || { echo "ERROR: ispatialgym fails to import"; exit 1; }
fi

if [ "$DATASET" = true ]; then
    echo "==> Downloading the datasets"
    activate_env
    export OMNI_KIT_ACCEPT_EULA=YES
    DATA_PATH="${OMNIGIBSON_DATA_PATH:-$B1K_DIR/datasets}"
    ACCEPT=$([ "$ACCEPT_DATASET_TOS" = true ] && echo True || echo False)
    python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
    python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=$ACCEPT)"
    mkdir -p "$DATA_PATH/ispatialgym-assets"
    cp -r "$SCRIPT_DIR/ispatialgym/assets/datasets/ispatialgym-assets/." "$DATA_PATH/ispatialgym-assets/"
    OMNIGIBSON_DATA_PATH="$DATA_PATH" python -m ispatialgym.eval.instances download
fi

echo "=== iSpatialGym setup complete ==="
