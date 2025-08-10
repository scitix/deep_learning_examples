# 使用说明
路径写死了，主要脚本位于/deep_learning_examples/training/fairseq下 
## 环境
hubert模型依赖fairseq框架

镜像地址：registry-cn-shanghai.siflow.cn/hisys/fairseq-hpc:v2.0

模型下载地址：https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

fairseq源码地址：https://github.com/facebookresearch/fairseq.git

## 数据集
hubert模型位于：https://github.com/facebookresearch/fairseq/tree/main/examples/hubert，主要使用libri-light数据集。

数据集源码位于：https://github.com/facebookresearch/libri-light.git

下载脚本为 libri-light/data_preparation

## 数据预处理
执行脚本 /deep_learning_examples/launcher_scripts/k8s/training/fairseq/data_process.yaml

默认处理libri-light的small.tar，如果需要使用其他类型的数据集，修改代码
```bash
# 解压原始音频
tar -xvf "${WORK_DIR}/small.tar" -C "${UNLAB_AUDIO_EXTRACTED_DIR}"

# 使用 VAD 切分音频
python "cut_by_vad.py" --input_dir "${UNLAB_AUDIO_EXTRACTED_DIR}/small" \
                                    --output_dir "${PROCESSED_AUDIO_SEGMENTED_DIR}"
```
将small.tar换成large.tar或者其他tar文件。

## 训练
执行脚本 /deep_learning_examples/launcher_scripts/k8s/training/fairseq/train_multi_pod.yaml

默认 huber-large模型，使用其他模型需要
```bash
--config-name hubert_xlarge_librivox.yaml                           \
    model.label_rate=100                                                \
    distributed_training.distributed_world_size=${WORLD_SIZE}           \
    task.data=${MANIFEST_DIR}                                           \
    task.label_dir=${KMEANMS_MODLE}                                     \
    task.labels="[\"km\"]"                  \
    checkpoint.restore_file="/everything/qcheng_workspace/train/benchmark/models/hubert_xlarge_ll60k.pt"
```
调整config-name和checkpoint文件，train_multi_pod_ram.yaml 会将数据放入到内存再开始训练。