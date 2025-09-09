#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random


def save_random_paths(folder_path, output_file, label, max_num=100):
    """
    从 folder_path 中随机选取 max_num 个文件（如不足 max_num 则全部使用）并写入到 output_file。
    写入格式：绝对路径 + 制表符 + 标签 + 换行。
    """
    # 1. 收集所有文件的绝对路径
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            absolute_path = os.path.join(root, file)
            all_files.append(absolute_path)

    # 2. 如果文件数大于 max_num，就随机选取 max_num 个，否则全部使用
    if len(all_files) <= max_num:
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, max_num)

    # 3. 将选取到的文件写入 output_file
    #    这里以追加模式打开 ('a')；如果你想每次都覆盖，可以改成 'w'
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for fpath in selected_files:
            out_file.write(f"{fpath}\t{label}\n")


def main():
    # 指定输出文件存放的文件夹
    output_file_base = r"/data/ssd2/tangshuai/DFIL299/mulu"
    # 包含 (文件夹路径 out_dir label) 信息的 txt 文件
    txt_file = "/data/ssd2/tangshuai/DFIL299/mulu.txt"

    # 若想每次运行都获得相同的随机结果，可以设置随机种子
    # random.seed(42)

    with open(txt_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # line.strip() 去除前后空白，split() 默认按空白分隔
            parts = line.strip().split()
            # 假设 txt 中每行包含：文件夹路径 out_dir label
            if len(parts) < 3:
                # 如果行信息不足，可以选择打印警告或跳过
                print(f"Line is not valid: {line}")
                continue

            folder_path = parts[0]
            out_dir = parts[1]
            label = parts[2]

            # 构建输出文件的完整路径
            output_file = os.path.join(output_file_base, f'rd_{out_dir}.txt')

            # 调用函数保存随机 100 个文件的绝对路径和标签
            save_random_paths(folder_path, output_file, label, max_num=300)

    print("处理完成！")


if __name__ == '__main__':
    main()
