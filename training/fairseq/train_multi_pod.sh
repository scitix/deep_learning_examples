#!/bin/bash
set -e

WHEELHOUSE_DIR="/everything/qcheng_workspace/train/benchmark/environment/wheelhouse"
MANIFEST_DIR="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/manifests"
KMEANMS_MODLE="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/kmeans_model"
OLD_BASE_PATH="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/processed_output"
NEW_BASE_PATH="/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/processed_output"

export export OPENBLAS_NUM_THREADS=1

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

# 安装所有 .deb 包
if ls ${WHEELHOUSE_DIR}/*.deb 1> /dev/null 2>&1; then
    echo "Installing all .deb packages from local wheelhouse..."
    dpkg -i ${WHEELHOUSE_DIR}/*.deb
else
    echo "No .deb packages found, skipping."
fi

# networkx 3.2.1 => networkx 2.8.8
CORRECT_NETWORKX_WHL="${WHEELHOUSE_DIR}/networkx-2.8.8-py3-none-any.whl"
if [ ! -f "$CORRECT_NETWORKX_WHL" ]; then
    echo "FATAL ERROR: The required networkx-2.8.8-py3-none-any.whl is not found in your wheelhouse!"
    echo "Please download it and place it in ${WHEELHOUSE_DIR}"
    exit 1
fi
pip install --force-reinstall "$CORRECT_NETWORKX_WHL"
# 安装所有 .whl 包
find "${WHEELHOUSE_DIR}" -name "*.whl" -not -name "networkx*" -exec pip install {} +

# torch >=2.6 只支持 tensor，设置 weights_only = false
CHECKPOINT_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/checkpoint_utils.py"
if ! grep -q "weights_only=False" "$CHECKPOINT_UTILS_FILE"; then sed -i "s/torch.load(f, map_location=torch.device(\"cpu\"))/torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)/g" ${CHECKPOINT_UTILS_FILE}; fi
DIST_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/distributed/utils.py"
if ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then sed -i "s/torch.load(buffer, map_location=\"cpu\")/torch.load(buffer, map_location=\"cpu\", weights_only=False)/g" ${DIST_UTILS_FILE}; fi

# 执行
/opt/conda/bin/torchrun                                                 \
  --nproc_per_node=8                                                    \
  /opt/conda/lib/python3.10/site-packages/fairseq_cli/hydra_train.py    \
    --config-dir /workspace/fairseq/examples/hubert/config/pretrain     \
    --config-name hubert_xlarge_librivox.yaml                           \
    model.label_rate=100                                                \
    distributed_training.distributed_world_size=${WORLD_SIZE}           \
    task.data=${MANIFEST_DIR}                                           \
    task.label_dir=${KMEANMS_MODLE}                                     \
    task.labels="[\"km\"]"                  \
    checkpoint.restore_file="/everything/qcheng_workspace/train/benchmark/models/hubert_xtralarge_ll60k.pt" \
    checkpoint.reset_dataloader=true        \
    checkpoint.reset_optimizer=true         \
    checkpoint.reset_lr_scheduler=true      \
    checkpoint.reset_meters=true            \
    common.seed=42                          \
    common.amp=false                        \
    common.fp16=false                       \
    common.bf16=true                        \
    dataset.num_workers=64                  \
    dataset.max_tokens=2800000              \
    optimization.max_update=1000            \
    optimization.update_freq=[8]