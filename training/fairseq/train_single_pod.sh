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
    echo "This script runs single-node, multi-GPU Hubert pre-training using torchrun."
    echo ""
    echo "Required Arguments:"
    echo "  --fairseq-repo <path>    Path to the cloned Fairseq repository (for training configs)."
    echo "  --manifest-dir <path>    Path to the directory containing train.tsv and valid.tsv."
    echo "  --kmeans-dir <path>      Path to the directory containing the K-Means model and labels."
    echo "  --restore-file <path>    Path to the initial model checkpoint (.pt) to restore from."
    echo ""
    echo "Optional Training Arguments:"
    echo "  --config-name <name>     Name of the config file in config-dir. (Default: hubert_xlarge_librivox.yaml)"
    echo "  --nproc-per-node <int>   Number of GPUs/processes to use. (Default: 8)"
    echo "  --max-tokens <int>       Max tokens per batch per GPU. (Default: 1400000)"
    echo "  --max-update <int>       Total number of training updates. (Default: 10000)"
    echo "  --master-port <int>      Port for the master process. (Default: 29500)"
    echo ""
    echo "Optional Environment Arguments:"
    echo "  --python-exec <path>     Path to the python executable in the correct env. (Default: python3)"
    echo "  -h, --help               Display this help message and exit."
    exit 1
}

FAIRSEQ_REPO_DIR=""
MANIFEST_DIR=""
KMEANS_DIR=""
RESTORE_FILE=""
CONFIG_NAME="hubert_xlarge_librivox.yaml"
NPROC_PER_NODE=8
MAX_TOKENS=1400000
MAX_UPDATE=10000
MASTER_PORT=29500
PYTHON_EXEC="python3"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fairseq-repo) FAIRSEQ_REPO_DIR="$2"; shift 2 ;;
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --kmeans-dir) KMEANS_DIR="$2"; shift 2 ;;
        --restore-file) RESTORE_FILE="$2"; shift 2 ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --max-update) MAX_UPDATE="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --python-exec) PYTHON_EXEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

for var in FAIRSEQ_REPO_DIR MANIFEST_DIR KMEANS_DIR RESTORE_FILE; do
    if [ -z "${!var}" ]; then
        error "Missing required argument: --${var//_/- | tr '[:upper:]' '[:lower:]'}. Use -h for help."
    fi
done

log "Configuration:"
log "  Fairseq Repo:     ${FAIRSEQ_REPO_DIR}"
log "  Manifest Dir:     ${MANIFEST_DIR}"
log "  K-Means Dir:      ${KMEANS_DIR}"
log "  Restore File:     ${RESTORE_FILE}"
log "  Config Name:      ${CONFIG_NAME}"
log "  Processes/GPUs:   ${NPROC_PER_NODE}"
log "  Max Tokens:       ${MAX_TOKENS}"
log "  Max Updates:      ${MAX_UPDATE}"

export OPENBLAS_NUM_THREADS=1

# 修复 Fairseq 代码
log "Step 1: Patching Fairseq for PyTorch compatibility..."
FAIRSEQ_LIB_PATH=$("$PYTHON_EXEC" -c "import fairseq, os; print(os.path.dirname(fairseq.__file__))" 2>/dev/null)
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
fi
# Patch distributed/utils.py
if [ -f "$DIST_UTILS_FILE" ] && ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then
    log "Patching ${DIST_UTILS_FILE}"
    sed -i 's/torch.load(buffer, map_location="cpu")/torch.load(buffer, map_location="cpu", weights_only=False)/g' "${DIST_UTILS_FILE}"
fi


# 训练
log "Step 2: Starting single-node training with torchrun..."
PYTHON_BIN_DIR=$(dirname $($PYTHON_EXEC -c "import sys; print(sys.executable)"))
TORCHRUN_PATH="${PYTHON_BIN_DIR}/torchrun"
TRAIN_SCRIPT_PATH="${PYTHON_BIN_DIR}/fairseq-hydra-train"

if [ ! -f "$TORCHRUN_PATH" ] || [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    error "torchrun or fairseq-hydra-train not found in ${PYTHON_BIN_DIR}. Ensure torch and fairseq are installed."
fi

"$TORCHRUN_PATH"                                                      \
  --nproc_per_node=${NPROC_PER_NODE}                                  \
  --nnodes=1                                                          \
  --node_rank=0                                                       \
  --master_addr=localhost                                             \
  --master_port=${MASTER_PORT}                                        \
  "$TRAIN_SCRIPT_PATH"                                                \
    --config-dir "${FAIRSEQ_REPO_DIR}/examples/hubert/config/pretrain" \
    --config-name "${CONFIG_NAME}"                                    \
    task.data="${MANIFEST_DIR}"                                       \
    task.label_dir="${KMEANS_DIR}"                                    \
    checkpoint.restore_file="${RESTORE_FILE}"                         \
    model.label_rate=100                                              \
    task.labels='["km"]'                                              \
    common.seed=42                                                    \
    common.amp=false                                                  \
    common.fp16=false                                                 \
    common.bf16=true                                                  \
    dataset.num_workers=64                                            \
    dataset.max_tokens=${MAX_TOKENS}                                  \
    optimization.max_update=${MAX_UPDATE}

log "Training finished."