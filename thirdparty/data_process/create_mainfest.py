import os
import soundfile as sf
import argparse
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file_info(full_path, data_root):
    # 在单独的线程中处理单个音频文件的信息
    try:
        info = sf.info(full_path)
        frames = info.frames
        relative_path = os.path.relpath(full_path, data_root)
        return (relative_path, frames)
    except sf.LibsndfileError as e:
        tqdm.write(f"警告: 发现可能损坏的音频文件 (soundfile error): {full_path} | 错误: {e}")
        return None
    except Exception as e:
        tqdm.write(f"警告: 无法处理文件: {full_path} | 错误: {e}")
        return None

def write_tsv(root_path, file_path, data):
    """将数据写入 .tsv 文件的函数"""
    with open(file_path, 'w') as f:
        f.write(root_path + '\n')
        for rel_path, frames in tqdm(data, desc=f"写入 {os.path.basename(file_path)}"):
            f.write(f"{rel_path}\t{frames}\n")

def main(data_root, dest_dir, valid_percent, num_threads):
    """
    遍历音频根目录，创建 fairseq 所需的 train.tsv 和 valid.tsv 清单文件。
    使用多线程并行处理文件信息。
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    data_root = os.path.abspath(data_root)

    all_file_paths = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".flac"):
                all_file_paths.append(os.path.join(root, file))

    print(f"成功找到 {len(all_file_paths)} 个音频文件。将使用 {num_threads} 线程处理信息。")
    processed_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_path = {executor.submit(process_file_info, path, data_root): path for path in all_file_paths}
        for future in tqdm(as_completed(future_to_path), total=len(all_file_paths), desc="处理音频文件信息"):
            result = future.result()
            if result:
                processed_results.append(result)

    print(f"成功处理了 {len(processed_results)} 个音频文件的信息。")
    random.shuffle(processed_results)
    split_idx = int(len(processed_results) * (1 - valid_percent))
    train_files = processed_results[:split_idx]
    valid_files = processed_results[split_idx:]

    print(f"切分数据集：{len(train_files)} 个用于训练，{len(valid_files)} 个用于验证。")
    write_tsv(data_root, os.path.join(dest_dir, "train.tsv"), train_files)
    write_tsv(data_root, os.path.join(dest_dir, "valid.tsv"), valid_files)
    print("清单文件 train.tsv 和 valid.tsv 已成功创建！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 Fairseq 创建音频清单文件。")
    parser.add_argument("--data-root", required=True, type=str, help="存放所有切分后音频的根目录的绝对路径。")
    parser.add_argument("--dest-dir", required=True, type=str, help="用于保存生成的 train.tsv 和 valid.tsv 文件的目录。")
    parser.add_argument("--valid-percent", default=0.01, type=float, help="用作验证集的数据百分比（例如 0.01 表示 1%）。")
    parser.add_argument("--num-threads", default=1, type=int,
                        help="用于并行处理文件信息的线程数量。")
    
    args = parser.parse_args()
    main(args.data_root, args.dest_dir, args.valid_percent, args.num_threads)