# User Guide
This document provides instructions for training the Hubert speech model, including environment setup, data preparation, and the use of provided training scripts and third-party libraries.
## 1. Data Preparation
### 1.1. Environment Setup
First, pull the required Docker image
```bash
registry-cn-shanghai.siflow.cn/hisys/fairseq-hpc:v1.0.
```
Additional dependencies are required for the environment. Download the packages listed below and place them into a single directory, which we will refer to as the "wheelhouse".
- **WHL Packages**
```bash
cffi-1.17.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
joblib-1.5.1-py3-none-any.whl
networkx-2.8.8-py3-none-any.whl
npy_append_array-0.9.19-py3-none-any.whl
numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pycparser-2.22-py3-none-any.whl
scikit_learn-1.7.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
soundfile-0.13.1-py2.py3-none-any.whl
threadpoolctl-3.6.0-py3-none-any.whl
tqdm-4.67.1-py3-none-any.whl
```
- **DEB Packages**
```bash
gcc-12-base_12.3.0-1ubuntu1~22.04_amd64.deb
libc6_2.35-0ubuntu3.10_amd64.deb
libcrypt1_1%3a4.4.27-1_amd64.deb
libflac8_1.3.3-2ubuntu0.2_amd64.deb
libgcc-s1_12.3.0-1ubuntu1~22.04_amd64.deb
libogg0_1.3.5-0ubuntu3_amd64.deb
libopus0_1.3.1-0.1build2_amd64.deb
libsndfile1_1.0.31-2ubuntu0.2_amd64.deb
libvorbis0a_1.3.7-1build2_amd64.deb
libvorbisenc2_1.3.7-1build2_amd64.deb
```
### 1.2. Dataset Preparation
The Hubert model is primarily trained on the Libri-Light dataset. You can find download links and information here: Libri-Light Data Preparation. To automate the data preprocessing, execute the launcher script

```bash
/deep_learning_examples/launcher_scripts/k8s/training/fairseq/data_process.yaml. 
```
This will start a container and run the processing script deep_learning_examples/training/fairseq/data_process.sh, which handles environment configuration and data preprocessing automatically. The script requires the following directory paths to be configured:
```bash
WHEELHOUSE_DIR          # The path to the directory containing the .whl and .deb dependency packages.
FAIRSEQ_REPO_DIR        # The path to the cloned Fairseq source code repository. Scripts from this repository are used during data processing.
WORK_DIR                deep_learning_examples/thirdparty/data_process      # Path to data preprocessing scripts and The main working directory where most operations will take place and where intermediate and final artifacts will be stored.
```
## 2. Running the Training
Two primary training modes are provided: training with the dataset stored on disk, and training with the dataset loaded into memory (RAM disk). Multi-node training is managed through Kubernetes YAML files
```bash
/deep_learning_examples/launcher_scripts/k8s/training/fairseq/train_multi_pod_ram.yaml
/deep_learning_examples/launcher_scripts/k8s/training/fairseq/train_multi_pod.yaml 
```
Single-node training is handled by the run_single_node_train.sh script.
- **Multi-Node Training**
For multi-node training, modify the command-line arguments within the corresponding YAML file. The key arguments are:
```bash
--wheelhouse        # Path to the directory with .whl and .deb dependencies.
--data-root         # Path to the original raw dataset.
--fairseq-repo      # Path to the Fairseq source code repository.
--restore-file      # Path to the model checkpoint file to restore from.
--ramdisk-path      # (For RAM training) The target path on the RAM disk where data will be copied.
--python-exec       # The Python executable to use (e.g., /opt/conda/bin/python3).
```
- **Single-Node Training**
For single-node training, execute the run_single_node_train.sh script with the following arguments:
```bash
./run_single_node_train.sh \
    --fairseq-repo /workspace/fairseq \
    --manifest-dir /datasets/qmanifests \
    --kmeans-dir /datasets/kmeans_model \
    --restore-file /datasets/models/hubert_xlarge_ll60k.pt \
    --nproc-per-node 8 \
    --max-update 10000
```
Argument Descriptions
```bash
--fairseq-repo          # Path to the Fairseq source code repository.
--manifest-dir          # Path to the directory containing manifest files (train.tsv, valid.tsv).
--kmeans-dir            # Path to the directory with K-Means model files.
--restore-file          # Path to the Hubert checkpoint to restore from.
--nproc-per-node        # Number of processes per node (typically equals the number of GPUs).
--max-update            # Total number of training steps (updates).
```

## 3. Troubleshooting and Lessons Learned
### 3.1 raining ends after a single step when using a pre-trained checkpoint.
- **Root Cause**

The checkpoint was pre-trained for 500,000 steps. If the new training session's max_update is set to a value less than or equal to this, Fairseq considers the training complete and exits immediately.
- **Solution**

Set max_update to a value significantly greater than the checkpoint's step count (e.g., max_update > 500000).
Alternatively, to restart training from step 0 with the pre-trained weights, set checkpoint.reset_dataloader=true. This resets the step counter and re-initializes the dataset iterator.

### 3.2 Encountering gradient explosion or NaN (Not-a-Number) values during training.
- **Root Cause**

Numerical instability, often related to mixed-precision training settings.
- **Solution** 

Enable Brain Floating-Point 16 (common.bf16=true) while explicitly disabling both automatic mixed precision (common.amp=false) and standard half-precision (common.fp16=false). The Hubert architecture contains specific layers (e.g., LayerNorm, embeddings) that are hardcoded to operate in FP32, causing data type conversion failures when standard AMP is used. BF16 provides a more stable alternative for mixed-precision training in this context.
### 3.3 Model architecture mismatch errors when loading a checkpoint.
- **Root Cause**

The model architecture defined in the YAML configuration file does not match the architecture saved within the checkpoint file (.pt).
- **Solution**

Ensure that the training configuration YAML (e.g., hubert_large_librivox.yaml) corresponds exactly to the model size and structure of the checkpoint being loaded. For example, use a "large" config for a "large" model.

### 3.4 Optimizing data loading and CPU utilization.
- **Root Cause**

 The num_workers parameter, which controls the number of parallel data-loading processes, internally utilizes libraries like OpenBLAS.
- **Solution**

To prevent CPU oversubscription and resource contention, adhere to the constraint: num_workers * OPENBLAS_NUM_THREADS <= Total CPU Cores. The best practice is often to use single-threaded workers by setting export OPENBLAS_NUM_THREADS=1 and increasing dataset.num_workers for parallelism.

### 3.5 ILow GPU utilization (low MFU - Model FLOPs Utilization).
- **Root Cause** 

The model or batch size is too small to saturate the GPU's computational capacity, leading to significant time spent on communication overhead (e.g., all_reduce for gradient synchronization, all_gather for data distribution).
- **Solution**

 Increase the effective batch size by using gradient accumulation (optimization.update_freq=[N], where N > 1). This allows the model to process more data before performing a single all_reduce operation, thereby reducing communication frequency and improving overall MFU.

### 3.6 To be continued... 