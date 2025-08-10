#!/bin/bash

export OPENBLAS_NUM_THREADS=1
RESTORE_FILE="/everything/qcheng_workspace/train/benchmark/models/hubert_xlarge_ll60k.pt"
# torch >= 2.6 只加载 tensor，pt 文件中包含自定义对象
# torch.serialization.add_safe_globals 加载 fairseq.data.dictionary.Dictionary 实例
CHECKPOINT_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/checkpoint_utils.py"
if ! grep -q "weights_only=False" "$CHECKPOINT_UTILS_FILE"; then sed -i "s/torch.load(f, map_location=torch.device(\"cpu\"))/torch.load(f, map_location=torch.device(\"cpu\"), weights_only=False)/g" ${CHECKPOINT_UTILS_FILE}; fi
DIST_UTILS_FILE="/opt/conda/lib/python3.10/site-packages/fairseq/distributed/utils.py"
if ! grep -q "weights_only=False" "$DIST_UTILS_FILE"; then sed -i "s/torch.load(buffer, map_location=\"cpu\")/torch.load(buffer, map_location=\"cpu\", weights_only=False)/g" ${DIST_UTILS_FILE}; fi

/opt/conda/bin/torchrun                       \
  --nproc_per_node=8                          \
  --nnodes=1                                  \
  --node_rank=0                               \
  --master_addr=localhost                     \
  --master_port=29500                         \
  /opt/conda/bin/fairseq-hydra-train          \
    --config-dir /tmp/fairseq/examples/hubert/config/pretrain                                         \
    --config-name hubert_xlarge_librivox.yaml \
    task.data=/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/manifests   \
    checkpoint.restore_file='$RESTORE_FILE'   \
    task.label_dir=/everything/qcheng_workspace/train/deep_learning_examples/thirdparty/data_process/kmeans_model              \
    model.label_rate=100                      \
    task.labels="[\"km\"]"                    \
    common.seed=42                            \
    common.amp=false                          \
    common.fp16=false                         \
    dataset.num_workers=64                    \
    dataset.max_tokens=1400000                \
    optimization.max_update=10000

