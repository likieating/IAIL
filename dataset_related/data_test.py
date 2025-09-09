import random

# 文件路径（替换为你的实际文件路径）
input_file = "C:\Users\11409\Desktop\real.txt"  # 原始文件
output_file = "C:\Users\11409\Desktop\out_real.txt"  # 保存200行的文件

# 读取原始文件中的所有行
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 随机选择200行
sampled_lines = random.sample(lines, 200)

# 将随机选择的200行写入新的文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(sampled_lines)

print(f"成功从 {input_file} 中随机挑选出 200 行，并保存到 {output_file} 中。")
