#!/bin/bash
set -e
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
}
warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1" >&2
}
error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}
run_with_retry() {
    local stage_name="$1"
    local target_file="$2"
    shift 2
    local cmd=("$@")

    if [ -f "${target_file}" ]; then
        log "Skipping [${stage_name}]. Target file '${target_file}' already exists."
    else
        log "Running [${stage_name}]..."
        log "COMMAND: ${cmd[*]}"
        "${cmd[@]}"
        if [ ! -f "${target_file}" ]; then
            error "Stage [${stage_name}] failed. Target file '${target_file}' was not created."
        fi
        log "Finished [${stage_name}] successfully."
    fi
}

usage() {
    echo "Usage: $0 --wheelhouse <path> --workdir <path> --fairseq <path>"
    echo ""
    echo "This script prepares data for Hubert pre-training."
    echo ""
    echo "Required Arguments:"
    echo "  --wheelhouse <path>   Path to the directory containing .whl and .deb files."
    echo "  --workdir <path>      Path to the main working directory for data processing."
    echo "  --fairseq <path>      Path to the cloned fairseq repository."
    echo ""
    echo "Options:"
    echo "  -h, --help            Display this help message and exit."
    exit 1
}

WHEELHOUSE_DIR=""
WORK_DIR=""
FAIRSEQ_REPO_DIR=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wheelhouse) WHEELHOUSE_DIR="$2"; shift 2 ;;
        --workdir) WORK_DIR="$2"; shift 2 ;;
        --fairseq) FAIRSEQ_REPO_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

if [ -z "$WHEELHOUSE_DIR" ] || [ -z "$WORK_DIR" ] || [ -z "$FAIRSEQ_REPO_DIR" ]; then
    echo "Error: All required arguments (--wheelhouse, --workdir, --fairseq) must be provided."
    echo ""
    usage
fi

log "Configuration:"
log "  Wheelhouse Directory: ${WHEELHOUSE_DIR}"
log "  Working Directory:    ${WORK_DIR}"
log "  Fairseq Repo:         ${FAIRSEQ_REPO_DIR}"

UNLAB_AUDIO_TAR_DIR="${WORK_DIR}/tar-data"
UNLAB_AUDIO_EXTRACTED_DIR="${WORK_DIR}/extracted_audio"
PROCESSED_AUDIO_SEGMENTED_DIR="${WORK_DIR}/processed_output"
MANIFESTS_DIR="${WORK_DIR}/manifests"
FEATURE_DIR="${WORK_DIR}/mfcc_features"
KMEANS_MODEL_DIR="${WORK_DIR}/kmeans_model"


# 安装所有 .deb 包
if ls "${WHEELHOUSE_DIR}"/*.deb 1> /dev/null 2>&1; then
    log "Installing all .deb packages from local wheelhouse..."
    sudo dpkg -i "${WHEELHOUSE_DIR}"/*.deb || sudo apt-get -f install -y # 如果dpkg失败，尝试修复依赖
else
    log "No .deb packages found, skipping."
fi

# 确保 networkx 2.8.8 被安装
CORRECT_NETWORKX_WHL="${WHEELHOUSE_DIR}/networkx-2.8.8-py3-none-any.whl"
if [ ! -f "$CORRECT_NETWORKX_WHL" ]; then
    error "FATAL ERROR: The required networkx-2.8.8-py3-none-any.whl is not found in your wheelhouse!"
fi
log "Force-reinstalling networkx 2.8.8..."
pip install --force-reinstall "$CORRECT_NETWORKX_WHL"

# 安装所有 .whl 包 (排除 networkx*)
log "Installing all other .whl packages..."
find "${WHEELHOUSE_DIR}" -name "*.whl" -not -name "networkx*" -exec pip install --no-deps {} +

# 确保所有目录存在
mkdir -p "${UNLAB_AUDIO_EXTRACTED_DIR}" "${PROCESSED_AUDIO_SEGMENTED_DIR}" \
         "${MANIFESTS_DIR}" "${FEATURE_DIR}" "${KMEANS_MODEL_DIR}"

cd "${WORK_DIR}" || error "Failed to change directory to ${WORK_DIR}"

export OPENBLAS_NUM_THREADS=32

# 解压原始无标签音频
if [ ! -f "${UNLAB_AUDIO_TAR_DIR}/large.tar" ]; then
    error "Raw audio file large.tar not found in ${UNLAB_AUDIO_TAR_DIR}. Please place it there."
fi
run_with_retry "Stage 0a: Untarring large.tar" \
  "${UNLAB_AUDIO_EXTRACTED_DIR}/large" \
  tar -xvf "${UNLAB_AUDIO_TAR_DIR}/large.tar" -C "${UNLAB_AUDIO_EXTRACTED_DIR}"

# VAD 切分长音频
if [ ! -f "${WORK_DIR}/cut_by_vad.py" ]; then
    error "cut_by_vad.py not found at ${WORK_DIR}/."
fi
run_with_retry "Stage 0.5: Segmenting audio with VAD" \
  "${PROCESSED_AUDIO_SEGMENTED_DIR}/.vad_done" \
  python "${WORK_DIR}/cut_by_vad.py" \
    --input_dir "${UNLAB_AUDIO_EXTRACTED_DIR}/large" \
    --output_dir "${PROCESSED_AUDIO_SEGMENTED_DIR}" && touch "${PROCESSED_AUDIO_SEGMENTED_DIR}/.vad_done"

# 生成 Manifest Files
run_with_retry "Stage 1: Generating Manifest Files (.tsv)" \
  "${MANIFESTS_DIR}/train.tsv" \
  python "./create_manifest.py" --data-root "${PROCESSED_AUDIO_SEGMENTED_DIR}" --dest-dir "${MANIFESTS_DIR}" --valid-percent 0.01

# 检测并清理损坏文件
if [ ! -f "./find_corrupted_audio.py" ]; then
    error "find_corrupted_audio.py not found in current directory (${WORK_DIR}). Please copy it."
fi
if [ ! -f "./create_manifest.py" ]; then
    error "create_manifest.py not found in current directory (${WORK_DIR}). Please copy it."
fi
if [ ! -f "./create_len.py" ]; then
    error "create_len.py not found in current directory (${WORK_DIR}). Please copy it."
fi

python "./find_corrupted_audio.py" --manifest "${MANIFESTS_DIR}/train.tsv" --log-file "${WORK_DIR}/bad_files_train.txt"
python "./find_corrupted_audio.py" --manifest "${MANIFESTS_DIR}/valid.tsv" --log-file "${WORK_DIR}/bad_files_valid.txt"

# 生成 .len 文件
run_with_retry "Stage 1.5: Generating .len files (.len)" \
  "${MANIFESTS_DIR}/train.len" \
  python "./create_len.py" --manifest-dir "${MANIFESTS_DIR}"

# 提取 MFCC Features
run_with_retry "Stage 2a: Extracting MFCC Features for train" \
  "${FEATURE_DIR}/train_0_1.npy" \
  python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_mfcc_feature.py" \
    "${MANIFESTS_DIR}" train 1 0 "${FEATURE_DIR}" --sample_rate 16000

run_with_retry "Stage 2b: Extracting MFCC Features for valid" \
  "${FEATURE_DIR}/valid_0_1.npy" \
  python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_mfcc_feature.py" \
    "${MANIFESTS_DIR}" valid 1 0 "${FEATURE_DIR}" --sample_rate 16000

# 训练 K-Means 模型
KM_MODEL_500_PATH="${KMEANS_MODEL_DIR}/km_500.bin"
run_with_retry "Stage 3a: Training K-Means model" \
  "${KM_MODEL_500_PATH}" \
  python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/learn_kmeans.py" \
    "${FEATURE_DIR}" train 1 0 "${KM_MODEL_500_PATH}" 500 --percent 0.1

# 导出 K-Means 伪标签
run_with_retry "Stage 4a: Dumping K-Means labels for train" \
  "${KMEANS_MODEL_DIR}/train_0_1.km" \
  python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_km_label.py" \
    "${FEATURE_DIR}" train "${KM_MODEL_500_PATH}" 1 0 "${KMEANS_MODEL_DIR}"

run_with_retry "Stage 4b: Dumping K-Means labels for valid" \
  "${KMEANS_MODEL_DIR}/valid_0_1.km" \
  python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_km_label.py" \
    "${FEATURE_DIR}" valid "${KM_MODEL_500_PATH}" 1 0 "${KMEANS_MODEL_DIR}"

# 重命名 .km 文件
log "Renaming .km files..."
mv "${KMEANS_MODEL_DIR}/train_0_1.km" "${KMEANS_MODEL_DIR}/train.km"
mv "${KMEANS_MODEL_DIR}/valid_0_1.km" "${KMEANS_MODEL_DIR}/valid.km"

# 生成 dict.km.txt
run_with_retry "Stage 5: Creating dict.km.txt" \
  "${KMEANS_MODEL_DIR}/dict.km.txt" \
  python "./create_dict.py" --input-file "${KMEANS_MODEL_DIR}/train.km" --output-dir "${KMEANS_MODEL_DIR}" --num-clusters 500

log "All stages completed successfully!"