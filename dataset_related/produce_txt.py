import os

def save_absolute_paths(folder1, output_file,label):
    # 打开 output_file 以写入模式
    with open(output_file, 'a') as out_file:
        # 遍历 folder1 中的所有文件
        for root, dirs, files in os.walk(folder1):
            for file in files:
                # 计算文件的绝对路径
                absolute_path = os.path.join(root, file)
                # 写入文件
                out_file.write(absolute_path + "\t" + label + '\n')

# 使用示例
output_file_base = r"/data/ssd2/tangshuai/DFIL299"  # 保存绝对路径的文本文件名
txt_file="/data/ssd2/tangshuai/DFIL299/mulu.txt"

with open(txt_file,'r') as infile:
    for line in infile:
        parts=line.strip().split()
        dir=parts[0]
        out_dir=parts[1]
        label=parts[2]
        output_file=os.path.join(output_file_base,f'split_val_{out_dir}.txt')
        save_absolute_paths(dir, output_file, label)

