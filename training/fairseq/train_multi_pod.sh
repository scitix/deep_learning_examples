#!/bin/bash
set -e

MANIFEST_DIR="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/mainfest_dir"
KMEANMS_MODLE="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/kmeans_model"
OLD_BASE_PATH="/datasets/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/processed_output"
NEW_BASE_PATH="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/processed_output"

export OPENBLAS_NUM_THREADS=1

# 替换 train.tsv 和 valid.tsv 中的路径
for MANIFEST_FILE in ${MANIFEST_DIR}/train.tsv ${MANIFEST_DIR}/valid.tsv; do
    if [ -f "$MANIFEST_FILE" ]; then
        if grep -q "${OLD_BASE_PATH}" "$MANIFEST_FILE"; then
            echo "Patching paths in ${MANIFEST_FILE}..."
            sed -i "s|${OLD_BASE_PATH}|${NEW_BASE_PATH}|g" "$MANIFEST_FILE"
            echo "Path correction complete for ${MANIFEST_FILE}."
        else
            echo "Paths in ${MANIFEST_FILE} already correct (no old paths found), skipping."
        fi
    else
        echo "Manifest file not found: ${MANIFEST_FILE}, skipping path correction."
    fi
done

# torch >=2.6 只支持 tensor，设置 weights_only = false
CHECKPOINT_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/checkpoint_utils.py"
if ! grep -q "weights_only=False" "$CHECKPOINT_UTILS_FILE"; then sed -i "s/torch.load(f, map_location=torch.device(\"cpu\"))/torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)/g" ${CHECKPOINT_UTILS_FILE}; fi
DIST_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/distributed/utils.py"
if ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then sed -i "s/torch.load(buffer, map_location=\"cpu\")/torch.load(buffer, map_location=\"cpu\", weights_only=False)/g" ${DIST_UTILS_FILE}; fi

# 执行
/opt/conda/bin/torchrun                                                 \
  --nproc_per_node=8                                                    \
  /opt/conda/lib/python3.10/site-packages/fairseq_cli/hydra_train.py    \
    --config-dir /tmp/fairseq/examples/hubert/config/pretrain           \
    --config-name hubert_xlarge_librivox.yaml                           \
    model.label_rate=100                                                \
    distributed_training.distributed_world_size=${WORLD_SIZE}           \
    task.data=${MANIFEST_DIR}                                           \
    task.label_dir=${KMEANMS_MODLE}                                     \
    task.labels="[\"km\"]"                  \
    checkpoint.restore_file="/everything/qcheng_workspace/train/benchmark/models/hubert_xlarge_ll60k.pt" \
    checkpoint.reset_dataloader=true        \
    checkpoint.reset_optimizer=true         \
    checkpoint.reset_lr_scheduler=true      \
    checkpoint.reset_meters=true            \
    common.seed=42                          \
    common.amp=false                        \
    common.fp16=false                       \
    common.bf16=true                        \
    dataset.num_workers=64                  \
    dataset.max_tokens=1400000              \
    optimization.max_update=1000            \
    optimization.update_freq=[4]            