#!/bin/bash
set -e


MANIFEST_DIR="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/manifests"
ORIGINAL_DATA_DIR="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty"
# RAM 挂载点
RAMDISK_BASE_DIR="/dev/shm/ramdisk" 
# 旧清单文件目录
OLD_BASE_PATH="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/processed_output"
# 复制数据到内存
RAMDISK_MANIFEST_DIR="${RAMDISK_BASE_DIR}/data/data-process/manifests"
RAMDISK_KMEANS_MODEL_DIR="${RAMDISK_BASE_DIR}/data/data-process/kmeans_model"
NEW_BASE_PATH="${RAMDISK_BASE_DIR}/data/data-process/processed_output"

export OPENBLAS_NUM_THREADS=1

mkdir -p "${RAMDISK_BASE_DIR}/data/data-process" || { echo "Failed to create RAM Disk target dir!"; exit 1; }
# 复制数据
cp -r "${ORIGINAL_DATA_DIR}/data-process" "${RAMDISK_BASE_DIR}/data/" || { echo "Failed to copy data to RAM Disk!"; exit 1; }
du -sh "${RAMDISK_BASE_DIR}"
df -h "${RAMDISK_BASE_DIR}"

# 替换 train.tsv 和 valid.tsv 中的路径
for MANIFEST_FILE in ${RAMDISK_MANIFEST_DIR}/train.tsv ${RAMDISK_MANIFEST_DIR}/valid.tsv; do
    if [ -f "$MANIFEST_FILE" ]; then
        if grep -q "${OLD_BASE_PATH}" "$MANIFEST_FILE"; then
            echo "Patching paths in ${MANIFEST_FILE}..."
            sed -i "s|${OLD_BASE_PATH}|${NEW_BASE_PATH}|g" "$MANIFEST_FILE"
            echo "Path correction complete for ${MANIFEST_FILE}."
        else
            echo "Paths in ${MANIFEST_FILE} already correct (no old paths found), skipping."
        fi
    else
        echo "Manifest file not found on RAM Disk: ${MANIFEST_FILE}, skipping path correction."
    fi
done

# torch >=2.6 只支持 tensor，设置 weights_only = false
CHECKPOINT_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/checkpoint_utils.py"
if ! grep -q "weights_only=False" "$CHECKPOINT_UTILS_FILE"; then sed -i "s/torch.load(f, map_location=torch.device(\"cpu\"))/torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)/g" ${CHECKPOINT_UTILS_FILE}; fi
DIST_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/distributed/utils.py"
if ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then sed -i "s/torch.load(buffer, map_location=\"cpu\")/torch.load(buffer, map_location=\"cpu\", weights_only=False)/g" ${DIST_UTILS_FILE}; fi

# 执行
/opt/conda/bin/torchrun                                               \
  --nproc_per_node=8                                                  \
  /opt/conda/lib/python3.10/site-packages/fairseq_cli/hydra_train.py  \
    --config-dir /tmp/fairseq/examples/hubert/config/pretrain         \
    --config-name hubert_large_librivox.yaml                          \
    model.label_rate=100                                              \
    distributed_training.distributed_world_size=${WORLD_SIZE}         \
    task.data=${RAMDISK_MANIFEST_DIR}                                 \
    task.label_dir=${RAMDISK_KMEANS_MODEL_DIR}                        \
    task.labels="[\"km\"]"                  \
    checkpoint.restore_file="/everything/qcheng_workspace/train/benchmark/models/hubert_large_ll60k.pt" \
    checkpoint.reset_dataloader=true        \
    checkpoint.reset_optimizer=true         \
    checkpoint.reset_lr_scheduler=true      \
    checkpoint.reset_meters=true            \
    common.seed=42                          \
    common.amp=false                        \
    common.fp16=false                       \
    dataset.num_workers=32                  \
    dataset.max_tokens=1400000              \
    optimization.max_update=1000