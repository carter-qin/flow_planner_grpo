import os
import json
import random

# 你的 Pittsburgh 数据路径
data_path = os.path.expanduser("~/data/nuplan/processed_data_train_pittsburgh")
output_json = "train_pittsburgh.json"

if not os.path.exists(data_path):
    print(f"Error: 还没开始跑数据处理吗？找不到 {data_path}")
else:
    print("正在扫描文件列表 (这可能需要几分钟)...")
    # 既然是百万级文件，listdir 可能会慢，耐心等待
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".npz")])

    with open(output_json, "w") as f:
        json.dump(all_files, f, indent=4)

    print(f"文件数: {len(all_files)}")
    print(f"训练列表已保存至: {output_json}")

    # 算账
    batch_size = 224
    steps = len(all_files) / batch_size
    time_per_epoch = steps * 0.4 / 60  # 假设 0.4s 一步
    print(f"\n[训练耗时预估] 单卡4090 (BS={batch_size}):")
    print(f" - 数据量: {len(all_files)}")
    print(f" - 每 Epoch: ~{time_per_epoch:.1f} 分钟")
    print(f" - 200 Epochs: ~{time_per_epoch * 200 / 60:.1f} 小时")
