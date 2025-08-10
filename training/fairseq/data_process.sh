#!/bin/bash
set -e

# 定义路径
WORK_DIR="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process"
FAIRSEQ_REPO_DIR="/tmp/fairseq"
UNLAB_AUDIO_EXTRACTED_DIR="${WORK_DIR}/extracted_audio"
PROCESSED_AUDIO_SEGMENTED_DIR="${WORK_DIR}/processed_output"
MANIFESTS_DIR="${WORK_DIR}/manifests"
FEATURE_DIR="${WORK_DIR}/mfcc_features"
KMEANS_MODEL_DIR="${WORK_DIR}/kmeans_model"

# 确保需要的目录存在
mkdir -p "${UNLAB_AUDIO_EXTRACTED_DIR}" "${PROCESSED_AUDIO_SEGMENTED_DIR}"        \
         "${MANIFESTS_DIR}" "${FEATURE_DIR}" "${KMEANS_MODEL_DIR}"

export OPENBLAS_NUM_THREADS=32
cd "${WORK_DIR}"

# 解压原始音频
tar -xvf "${WORK_DIR}/small.tar" -C "${UNLAB_AUDIO_EXTRACTED_DIR}"

# 使用 VAD 切分音频
python "cut_by_vad.py" --input_dir "${UNLAB_AUDIO_EXTRACTED_DIR}/small"           \
                                    --output_dir "${PROCESSED_AUDIO_SEGMENTED_DIR}"

# 生成 Manifest 和 .len 文件
python "create_mainfest.py" --data-root "${PROCESSED_AUDIO_SEGMENTED_DIR}" --dest-dir "${MANIFESTS_DIR}" --valid-percent 0.01
python "create_len.py" --manifest-dir "${MANIFESTS_DIR}"

# 提取 MFCC 特征
python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_mfcc_feature.py"   \
  "${MANIFESTS_DIR}" train 1 0 "${FEATURE_DIR}" --sample_rate 16000
python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_mfcc_feature.py"   \
  "${MANIFESTS_DIR}" valid 1 0 "${FEATURE_DIR}" --sample_rate 16000

# 训练 K-Means 模型
python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/learn_kmeans.py"        \
  "${FEATURE_DIR}" train 1 "${KMEANS_MODEL_DIR}/km_500.bin" 500 --percent 0.1

# 导出 K-Means 标签
python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_km_label.py"       \
  "${FEATURE_DIR}" train "${KMEANS_MODEL_DIR}/km_500.bin" 1 0 "${KMEANS_MODEL_DIR}"
python "${FAIRSEQ_REPO_DIR}/examples/hubert/simple_kmeans/dump_km_label.py"       \
  "${FEATURE_DIR}" valid "${KMEANS_MODEL_DIR}/km_500.bin" 1 0 "${KMEANS_MODEL_DIR}"

# 文件覆盖
mv -f "${KMEANS_MODEL_DIR}/train_0_1.km" "${KMEANS_MODEL_DIR}/train.km"
mv -f "${KMEANS_MODEL_DIR}/valid_0_1.km" "${KMEANS_MODEL_DIR}/valid.km"

# 生成字典文件
python "create_dict.py" --input-file "${KMEANS_MODEL_DIR}/train.km" --output-dir "${KMEANS_MODEL_DIR}" --num-clusters 500