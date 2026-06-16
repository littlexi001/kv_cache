import os
import re
import argparse

def clean_checkpoints(directory: str, dry_run: bool = False):
    """
    根据指定规则清理模型检查点文件。

    规则：
    - 文件名格式为 <数字>.pth，数字代表 batch 数。
    - 20000 步以内：每 2000 步保留一个。
    - 20000 步及以后：每 5000 步保留一个。
    - 其余文件将被删除。

    Args:
        directory (str): 包含检查点文件的目录路径。
        dry_run (bool): 如果为 True，则只打印要删除的文件而不实际删除。
                        默认为 False。
    """
    if not os.path.isdir(directory):
        print(f"错误: 指定的路径 '{directory}' 不是一个有效的目录。")
        return

    # 用于存储所有检查点文件信息 (batch数, 文件路径)
    checkpoints = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 使用正则表达式匹配 "<数字>.pth" 格式的文件名
        match = re.match(r'^(\d+)\.pth$', filename)
        if match:
            batch_num = int(match.group(1))
            file_path = os.path.join(directory, filename)
            checkpoints.append((batch_num, file_path))
    
    # 如果没有找到任何检查点文件
    if not checkpoints:
        print(f"在目录 '{directory}' 中未找到匹配 '<数字>.pth' 格式的文件。")
        return

    # 按 batch 数对检查点进行排序
    checkpoints.sort(key=lambda x: x[0])
    print(f"在目录 '{directory}' 中找到 {len(checkpoints)} 个检查点文件。")

    # 存储需要保留的 batch 数
    to_keep = set()

    # --- 应用保留规则 ---
    for batch_num, _ in checkpoints:
        if batch_num < 20000:
            # 20000 步以内：每 2000 步保留一个 (包括第 0 步)
            if batch_num % 2000 == 0:
                to_keep.add(batch_num)
        elif batch_num < 200000:
            # 20000 步 - 200000 步：每 5000 步保留一个
            if batch_num % 5000 == 0:
                to_keep.add(batch_num)
        else:
            # 200000 步及以后：每 10000 步保留一个
            if batch_num % 50000 == 0:
                to_keep.add(batch_num)
                
    # --- 处理最后一个检查点 ---
    # 无论规则如何，总是保留最后一个检查点（通常包含最终模型）
    last_checkpoint_batch = checkpoints[-1][0]
    print(f"始终保留最后一个检查点: {last_checkpoint_batch}.pth")
    to_keep.add(last_checkpoint_batch)

    # --- 删除不符合规则的文件 ---
    deleted_count = 0
    for batch_num, file_path in checkpoints:
        if batch_num not in to_keep:
            try:
                if dry_run:
                    print(f"[Dry Run] 将要删除: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                deleted_count += 1
            except OSError as e:
                print(f"删除文件 '{file_path}' 时出错: {e}")

    print(f"\n清理完成。总共删除了 {deleted_count} 个文件。")
    print(f"保留了 {len(to_keep)} 个文件。")

if __name__ == "__main__":
    for directory in os.listdir("../checkpoints"):
        directory = os.path.join("../checkpoints", directory)
        if os.path.isdir(directory):
            print(f"正在处理目录: {directory}")
            clean_checkpoints(directory, dry_run=False)