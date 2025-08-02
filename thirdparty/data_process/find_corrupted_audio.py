import os
import argparse
import soundfile as sf
from tqdm import tqdm

def find_corrupted_files(manifest_path, bad_files_log):
    if not os.path.exists(manifest_path):
        print(f"错误: 清单文件不存在 at {manifest_path}")
        return

    print(f"检测清单文件: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        root_path = lines[0].strip()
        relative_paths = [line.strip().split('\t')[0] for line in lines[1:]]
    
    corrupted_count = 0
    with open(bad_files_log, 'w') as f_bad:
        for rel_path in tqdm(relative_paths, desc="正在检测"):
            full_path = os.path.join(root_path, rel_path)
            try:
                sf.read(full_path)
            except (sf.LibsndfileError, RuntimeError) as e:
                print(f"\n检测到损坏文件: {rel_path} | 错误: {e}")
                f_bad.write(rel_path + '\n')
                corrupted_count += 1

    print(f"\n检测完成！共发现 {corrupted_count} 个损坏文件。")
    if corrupted_count > 0:
        print(f"所有损坏文件的路径已记录在: {bad_files_log}")
    else:
        print("未发现损坏文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检测 Fairseq 音频清单中的损坏文件。")
    parser.add_argument("--manifest", required=True, type=str, help="要检测的 .tsv 清单文件路径 (例如 train.tsv)。")
    parser.add_argument("--log-file", default="bad_files.txt", type=str, help="用于记录损坏文件列表的日志文件名。")
    
    args = parser.parse_args()
    find_corrupted_files(args.manifest, args.log_file)