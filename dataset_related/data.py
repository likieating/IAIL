import numpy as np

# 加载 .npy 文件
data = np.load('/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/identity/0224-19-50-33-744_OSMA_split1_xception_1993_B0_Inc1/task1/closed_set_pred.npy')

# 打印文件内容
print(data)

# 可选：查看数组的形状、类型等信息
print("数组的形状:", data.shape)
print("数组的数据类型:", data.dtype)
np.savetxt('/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/identity/0224-19-50-33-744_OSMA_split1_xception_1993_B0_Inc1/task1/pred_file.txt', data)