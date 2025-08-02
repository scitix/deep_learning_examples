#!/bin/bash
set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script sets up the environment, prepares data on a RAM disk, and starts Hubert pre-training."
    echo ""
    echo "Required Arguments:"
    echo "  --wheelhouse <path>      Path to the directory containing .whl and .deb files."
    echo "  --data-root <path>       Path to the original base data directory (contains 'data-process' subdir)."
    echo "  --fairseq-repo <path>    Path to the cloned fairseq repository (for training configs)."
    echo "  --restore-file <path>    Path to the initial model checkpoint (.pt) to restore from."
    echo ""
    echo "Optional Arguments:"
    echo "  --ramdisk-path <path>    Base path for the RAM disk. (Default: /dev/shm/ramdisk)"
    echo "  --python-exec <path>     Path to the python executable in the correct env. (Default: python3)"
    echo "  -h, --help               Display this help message and exit."
    exit 1
}

WHEELHOUSE_DIR=""
DATA_ROOT=""
FAIRSEQ_REPO_DIR=""
RESTORE_FILE=""
RAMDISK_BASE_DIR="/dev/shm/ramdisk"
PYTHON_EXEC="python3"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wheelhouse) WHEELHOUSE_DIR="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --fairseq-repo) FAIRSEQ_REPO_DIR="$2"; shift 2 ;;
        --restore-file) RESTORE_FILE="$2"; shift 2 ;;
        --ramdisk-path) RAMDISK_BASE_DIR="$2"; shift 2 ;;
        --python-exec) PYTHON_EXEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

if [ -z "$WHEELHOUSE_DIR" ] || [ -z "$DATA_ROOT" ] || [ -z "$FAIRSEQ_REPO_DIR" ] || [ -z "$RESTORE_FILE" ]; then
    error "Missing one or more required arguments. Use -h or --help for details."
fi
if ! command -v "$PYTHON_EXEC" &> /dev/null; then
    error "Python executable not found at: ${PYTHON_EXEC}"
fi
if [ -z "${WORLD_SIZE}" ]; then
    error "Environment variable WORLD_SIZE is not set. Please set it to the number of GPUs."
fi

log "Configuration:"
log "  Wheelhouse Directory:   ${WHEELHOUSE_DIR}"
log "  Original Data Root:     ${DATA_ROOT}"
log "  Fairseq Repo (configs): ${FAIRSEQ_REPO_DIR}"
log "  Restore Checkpoint:     ${RESTORE_FILE}"
log "  RAM Disk Path:          ${RAMDISK_BASE_DIR}"
log "  Python Executable:      ${PYTHON_EXEC}"
log "  World Size (from env):  ${WORLD_SIZE}"

OLD_BASE_PATH="${DATA_ROOT}/data-process/processed_output"
RAMDISK_DATA_TARGET_DIR="${RAMDISK_BASE_DIR}/data"
RAMDISK_MANIFEST_DIR="${RAMDISK_DATA_TARGET_DIR}/data-process/manifests" # Note: Original script had a typo 'mainfest'
RAMDISK_KMEANS_MODEL_DIR="${RAMDISK_DATA_TARGET_DIR}/data-process/kmeans_model"
NEW_BASE_PATH="${RAMDISK_DATA_TARGET_DIR}/data-process/processed_output"


export OPENBLAS_NUM_THREADS=1

# 数据准备
log "Step 1: Preparing data on RAM Disk..."
mkdir -p "${RAMDISK_DATA_TARGET_DIR}" || error "Failed to create RAM Disk target dir: ${RAMDISK_DATA_TARGET_DIR}"

log "Copying data from ${DATA_ROOT}/data-process to ${RAMDISK_DATA_TARGET_DIR}"
rsync -ah --info=progress2 "${DATA_ROOT}/data-process" "${RAMDISK_DATA_TARGET_DIR}/" || error "Failed to copy data to RAM Disk!"
log "Data copy complete."
du -sh "${RAMDISK_BASE_DIR}"
df -h "${RAMDISK_BASE_DIR}"

# 替换 train.tsv 和 valid.tsv 中的路径
for MANIFEST_FILE in ${RAMDISK_MANIFEST_DIR}/train.tsv ${RAMDISK_MANIFEST_DIR}/valid.tsv; do
    if [ -f "$MANIFEST_FILE" ]; then
        if grep -qF "${OLD_BASE_PATH}" "$MANIFEST_FILE"; then
            log "Patching paths in ${MANIFEST_FILE}..."
            # 使用 | 作为 sed 分隔符，避免路径中的 / 冲突
            sed -i "s|${OLD_BASE_PATH}|${NEW_BASE_PATH}|g" "$MANIFEST_FILE"
            log "Path correction complete for ${MANIFEST_FILE}."
        else
            log "Paths in ${MANIFEST_FILE} seem correct (no old paths found), skipping."
        fi
    else
        log "Manifest file not found on RAM Disk: ${MANIFEST_FILE}, skipping path correction."
    fi
done

# 环境安装
log "Step 2: Setting up Python environment..."
# 安装所有 .deb 包
if ls "${WHEELHOUSE_DIR}"/*.deb 1> /dev/null 2>&1; then
    log "Installing all .deb packages from local wheelhouse... (requires sudo)"
    sudo dpkg -i "${WHEELHOUSE_DIR}"/*.deb || sudo apt-get -f install -y
else
    log "No .deb packages found, skipping."
fi

# 确保 networkx 2.8.8 被安装
CORRECT_NETWORKX_WHL="${WHEELHOUSE_DIR}/networkx-2.8.8-py3-none-any.whl"
if [ ! -f "$CORRECT_NETWORKX_WHL" ]; then
    error "The required networkx-2.8.8-py3-none-any.whl is not found in your wheelhouse!"
fi
log "Force-reinstalling networkx 2.8.8..."
"$PYTHON_EXEC" -m pip install --force-reinstall "$CORRECT_NETWORKX_WHL"

# 安装所有 .whl 包 (排除 networkx*)
log "Installing all other .whl packages..."
find "${WHEELHOUSE_DIR}" -name "*.whl" -not -name "networkx*" -exec "$PYTHON_EXEC" -m pip install --no-deps {} +

# --- 3. 修复 Fairseq 代码 (针对新版 PyTorch) ---
log "Step 3: Patching Fairseq for PyTorch compatibility..."
# 动态查找 fairseq 包的位置
FAIRSEQ_LIB_PATH=$("$PYTHON_EXEC" -c "import fairseq, os; print(os.path.dirname(fairseq.__file__))")
if [ -z "$FAIRSEQ_LIB_PATH" ]; then
    error "Could not find the fairseq library path. Is it installed in the ${PYTHON_EXEC} environment?"
fi
log "Found fairseq library at: ${FAIRSEQ_LIB_PATH}"

CHECKPOINT_UTILS_FILE="${FAIRSEQ_LIB_PATH}/checkpoint_utils.py"
DIST_UTILS_FILE="${FAIRSEQ_LIB_PATH}/distributed/utils.py"

# Patch checkpoint_utils.py
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
# 找到 fairseq-hydra-train 脚本的位置
TRAIN_SCRIPT_PATH=$(dirname $($PYTHON_EXEC -c "import sys; print(sys.executable)"))/fairseq-hydra-train

if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    error "fairseq-hydra-train script not found at ${TRAIN_SCRIPT_PATH}. Ensure fairseq is properly installed."
fi
log "Using training script at: ${TRAIN_SCRIPT_PATH}"

torchrun                                                              \
  --nproc_per_node=8                                                  \
  "$TRAIN_SCRIPT_PATH"                                                \
    --config-dir "${FAIRSEQ_REPO_DIR}/examples/hubert/config/pretrain" \
    --config-name hubert_large_librivox.yaml                          \
    model.label_rate=100                                              \
    distributed_training.distributed_world_size=${WORLD_SIZE}         \
    task.data="${RAMDISK_MANIFEST_DIR}"                               \
    task.label_dir="${RAMDISK_KMEANS_MODEL_DIR}"                      \
    task.labels='["km"]'                                              \
    checkpoint.restore_file="${RESTORE_FILE}"                         \
    checkpoint.reset_dataloader=true                                  \
    checkpoint.reset_optimizer=true                                   \
    checkpoint.reset_lr_scheduler=true                                \
    checkpoint.reset_meters=true                                      \
    common.seed=42                                                    \
    common.bf16=true                                                  \
    common.fp16=false                                                 \
    dataset.num_workers=64                                            \
    dataset.max_tokens=2800000                                        \
    optimization.max_update=1000                                      \
    optimization.update_freq='[8]'

log "Training script finished."