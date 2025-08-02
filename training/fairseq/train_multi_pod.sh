#!/bin/bash
set -e

log() {
    if [ "${RANK:-0}" -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    fi
}
error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR on RANK ${RANK:-N/A}: $1" >&2
    exit 1
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script prepares the environment and starts Hubert pre-training directly from storage (no RAM disk)."
    echo ""
    echo "Required Arguments:"
    echo "  --wheelhouse <path>      Path to the directory containing .whl and .deb files."
    echo "  --manifest-dir <path>    Path to the directory containing train.tsv and valid.tsv."
    echo "  --kmeans-dir <path>      Path to the directory containing the K-Means model and labels."
    echo "  --fairseq-repo <path>    Path to the cloned fairseq repository (for training configs)."
    echo "  --restore-file <path>    Path to the initial model checkpoint (.pt) to restore from."
    echo "  --old-base-path <path>   The old, incorrect base path in the manifest files to be replaced."
    echo "  --new-base-path <path>   The new, correct base path to replace the old one with."
    echo ""
    echo "Optional Arguments:"
    echo "  --python-exec <path>     Path to the python executable in the correct env. (Default: python3)"
    echo "  -h, --help               Display this help message and exit."
    exit 1
}

WHEELHOUSE_DIR=""
MANIFEST_DIR=""
KMEANS_DIR=""
FAIRSEQ_REPO_DIR=""
RESTORE_FILE=""
OLD_BASE_PATH=""
NEW_BASE_PATH=""
PYTHON_EXEC="python3"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wheelhouse) WHEELHOUSE_DIR="$2"; shift 2 ;;
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --kmeans-dir) KMEANS_DIR="$2"; shift 2 ;;
        --fairseq-repo) FAIRSEQ_REPO_DIR="$2"; shift 2 ;;
        --restore-file) RESTORE_FILE="$2"; shift 2 ;;
        --old-base-path) OLD_BASE_PATH="$2"; shift 2 ;;
        --new-base-path) NEW_BASE_PATH="$2"; shift 2 ;;
        --python-exec) PYTHON_EXEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

for var in WHEELHOUSE_DIR MANIFEST_DIR KMEANS_DIR FAIRSEQ_REPO_DIR RESTORE_FILE OLD_BASE_PATH NEW_BASE_PATH; do
    if [ -z "${!var}" ]; then
        error "Missing required argument: --${var//_/- | tr '[:upper:]' '[:lower:]'}. Use -h for help."
    fi
done

if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    error "Python executable not found at: ${PYTHON_EXEC}"
fi
if [ -z "${WORLD_SIZE}" ]; then
    error "Environment variable WORLD_SIZE is not set. This script expects to be run in a PyTorchJob environment."
fi

log "Configuration:"
log "  Wheelhouse Directory:   ${WHEELHOUSE_DIR}"
log "  Manifest Directory:     ${MANIFEST_DIR}"
log "  K-Means Model Dir:      ${KMEANS_DIR}"
log "  Fairseq Repo (configs): ${FAIRSEQ_REPO_DIR}"
log "  Restore Checkpoint:     ${RESTORE_FILE}"
log "  Old Path in Manifest:   ${OLD_BASE_PATH}"
log "  New Path in Manifest:   ${NEW_BASE_PATH}"
log "  Python Executable:      ${PYTHON_EXEC}"
log "  World Size (from env):  ${WORLD_SIZE}"


export OPENBLAS_NUM_THREADS=1

# 修复文件路径
if [ "${RANK:-0}" -eq 0 ]; then
    log "Step 1: Patching manifest file paths on RANK 0..."
    for MANIFEST_FILE in "${MANIFEST_DIR}/train.tsv" "${MANIFEST_DIR}/valid.tsv"; do
        if [ -f "$MANIFEST_FILE" ]; then
            if grep -qF "${OLD_BASE_PATH}" "$MANIFEST_FILE"; then
                log "Patching paths in ${MANIFEST_FILE}..."
                sed -i "s|${OLD_BASE_PATH}|${NEW_BASE_PATH}|g" "$MANIFEST_FILE"
                log "Path correction complete for ${MANIFEST_FILE}."
            else
                log "Paths in ${MANIFEST_FILE} seem correct (no old paths found), skipping."
            fi
        else
            warn "Manifest file not found: ${MANIFEST_FILE}, skipping path correction."
        fi
    done
fi
# 环境安装
log "Step 2: Setting up Python environment..."
# 安装 .deb 包
if [ "${RANK:-0}" -eq 0 ]; then
    if ls "${WHEELHOUSE_DIR}"/*.deb 1> /dev/null 2>&1; then
        log "Installing all .deb packages from local wheelhouse... (requires sudo)"
        sudo dpkg -i "${WHEELHOUSE_DIR}"/*.deb || sudo apt-get -f install -y
    else
        log "No .deb packages found, skipping."
    fi
fi
# Python 包安装
CORRECT_NETWORKX_WHL="${WHEELHOUSE_DIR}/networkx-2.8.8-py3-none-any.whl"
if [ ! -f "$CORRECT_NETWORKX_WHL" ]; then
    error "The required networkx-2.8.8-py3-none-any.whl is not found in your wheelhouse!"
fi
log "Installing/updating Python packages..."
"$PYTHON_EXEC" -m pip install --force-reinstall "$CORRECT_NETWORKX_WHL"
find "${WHEELHOUSE_DIR}" -name "*.whl" -not -name "networkx*" -exec "$PYTHON_EXEC" -m pip install --no-deps {} +

# fairseq 补丁
log "Step 3: Patching Fairseq for PyTorch compatibility..."
FAIRSEQ_LIB_PATH=$("$PYTHON_EXEC" -c "import fairseq, os; print(os.path.dirname(fairseq.__file__))")
if [ -z "$FAIRSEQ_LIB_PATH" ]; then
    error "Could not find the fairseq library path."
fi
log "Found fairseq library at: ${FAIRSEQ_LIB_PATH}"

CHECKPOINT_UTILS_FILE="${FAIRSEQ_LIB_PATH}/checkpoint_utils.py"
DIST_UTILS_FILE="${FAIRSEQ_LIB_PATH}/distributed/utils.py"

if [ -f "$CHECKPOINT_UTILS_FILE" ] && ! grep -q "weights_only=False" "$CHECKPOINT_UTILS_FILE"; then
    log "Patching ${CHECKPOINT_UTILS_FILE}"
    sed -i 's/torch.load(f, map_location=torch.device("cpu"))/torch.load(f, map_location=torch.device("cpu"), weights_only=False)/g' "${CHECKPOINT_UTILS_FILE}"
else
    log "${CHECKPOINT_UTILS_FILE} already patched or not found."
fi

# Patch distributed/utils.py
if [ -f "$DIST_UTILS_FILE" ] && ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then
    log "Patching ${DIST_UTILS_FILE}"
    sed -i 's/torch.load(buffer, map_location="cpu")/torch.load(buffer, map_location="cpu", weights_only=False)/g' "${DIST_UTILS_FILE}"
else
    log "${DIST_UTILS_FILE} already patched or not found."
fi

# 训练
log "Step 4: Starting Hubert pre-training with torchrun..."
TRAIN_SCRIPT_PATH=$(dirname $($PYTHON_EXEC -c "import sys; print(sys.executable)"))/fairseq-hydra-train
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    error "fairseq-hydra-train script not found."
fi
log "Using training script at: ${TRAIN_SCRIPT_PATH}"

torchrun                                                              \
  --nproc_per_node=8                                                  \
  "$TRAIN_SCRIPT_PATH"                                                \
    --config-dir "${FAIRSEQ_REPO_DIR}/examples/hubert/config/pretrain" \
    --config-name hubert_xlarge_librivox.yaml                         \
    model.label_rate=100                                              \
    distributed_training.distributed_world_size=${WORLD_SIZE}         \
    task.data="${MANIFEST_DIR}"                                       \
    task.label_dir="${KMEANS_DIR}"                                    \
    task.labels='["km"]'                                              \
    checkpoint.restore_file="${RESTORE_FILE}"                         \
    checkpoint.reset_dataloader=true                                  \
    checkpoint.reset_optimizer=true                                   \
    checkpoint.reset_lr_scheduler=true                                \
    checkpoint.reset_meters=true                                      \
    common.seed=42                                                    \
    common.amp=false                                                  \
    common.fp16=false                                                 \
    common.bf16=true                                                  \
    dataset.num_workers=64                                            \
    dataset.max_tokens=2800000                                        \
    optimization.max_update=1000                                      \
    optimization.update_freq='[8]'

log "Training script finished on RANK ${RANK}."