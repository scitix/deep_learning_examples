import os
import argparse
from tqdm import tqdm

def main(manifest_dir):
    # 根据 train.tsv 和 valid.tsv 文件生成 fairseq K-Means 所需的 .len 文件。
    manifest_dir = os.path.abspath(manifest_dir)
    print(f"正在为清单文件目录 {manifest_dir} 生成 .len 文件...")
    # 处理 train 和 valid 
    for split in ["train", "valid"]:
        tsv_path = os.path.join(manifest_dir, f"{split}.tsv")
        len_path = os.path.join(manifest_dir, f"{split}_0_1.len")

        if not os.path.exists(tsv_path):
            print(f"警告: 找不到 {tsv_path}。跳过生成 {split}_0_1.len。")
            continue

        print(f"正在从 {tsv_path} 读取并生成 {len_path}...")
        # 读取 TSV 文件内容
        lengths = []
        with open(tsv_path, 'r') as f_tsv:
            lines = f_tsv.readlines()
            if len(lines) < 2: 
                print(f"警告: {tsv_path} 文件内容不足（可能只有标题行）。跳过。")
                continue
            
            for line in tqdm(lines[1:], desc=f"处理 {os.path.basename(tsv_path)}"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        lengths.append(str(int(parts[1])))
                    except ValueError:
                        print(f"警告: 无法解析行中的帧数，跳过: {line.strip()}")
                else:
                    print(f"警告: 跳过格式不正确的行: {line.strip()}")

        with open(len_path, 'w') as f_len:
            for length in lengths:
                f_len.write(length + '\n')
        
        print(f"{len_path} 已成功创建，包含 {len(lengths)} 条记录。")
    
    print("所有 .len 文件生成完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 .tsv 文件生成 fairseq K-Means 所需的 .len 文件。")
    parser.add_argument("--manifest-dir", required=True, type=str, 
                        help="存放 train.tsv 和 valid.tsv 文件的目录的绝对路径。")
    
    args = parser.parse_args()
    main(args.manifest_dir)