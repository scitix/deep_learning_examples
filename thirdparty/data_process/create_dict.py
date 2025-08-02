import argparse
from collections import Counter
from tqdm import tqdm
import os

def main(input_path, output_dir, num_clusters):
    """
    读取 .km 标签文件，统计每个聚类标签的频率，并生成 fairseq 格式的字典文件。
    """
    print(f"正在从 {input_path} 读取标签并统计频率...")
    counter = Counter()
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="处理标签文件"):
            labels = line.strip().split()
            counter.update(labels)

    print("频率统计完成！")

    dict_path = os.path.join(output_dir, "dict.km.txt")
    print(f"正在将字典写入到: {dict_path}")

    with open(dict_path, 'w') as f:
        for i in range(num_clusters):
            label_str = str(i)
            count = counter.get(label_str, 0)
            f.write(f"{label_str} {count}\n")
            
    print("字典文件 dict.km.txt 已成功创建！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为 K-Means 标签创建 Fairseq 字典文件。")
    parser.add_argument("--input-file", required=True, type=str, help="输入的 .km 标签文件路径 (例如 train.km)。")
    parser.add_argument("--output-dir", required=True, type=str, help="用于保存生成的 dict.km.txt 文件的目录。")
    parser.add_argument("--num-clusters", default=100, type=int, help="K-Means 使用的聚类中心数量。")
    
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.num_clusters)