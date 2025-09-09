import matplotlib.pyplot as plt
import numpy as np

# 生成 x 轴数据 (Session ID)
sessions = np.array([0, 1, 2, 3])
session_labels = ["Stage1(FF++)", "Stage2(DFDC-P)", "Stage3(DFD)", "Stage4(CDF2)"]

# 生成不同方法的 mAA 数据
dfil = np.array([95.65, 86.15, 81.51, 81.43])
icarl=np.array([94.76,80.21,77.16,75.83])
lwf=np.array([94.83,77.90,77.59,69.60])
dmp=np.array([95.59,86.73,82.39,78.99])
ours=np.array([95.40,88.28,83.84,84.56])
# 颜色和标记
colors = ["blue", "green", "purple", "black","red"]
markers = ["o", "d", "s", "x", "+"]

# 绘制折线图
plt.figure(figsize=(6, 4))

plt.plot(sessions, lwf, marker=markers[0], color=colors[0], label="LWF", linestyle="-")
plt.plot(sessions, icarl, marker=markers[1], color=colors[1], label="iCaRL", linestyle="-")
plt.plot(sessions, dfil, marker=markers[2], color=colors[2], label="DFIL", linestyle="-")
plt.plot(sessions, dmp, marker=markers[3], color=colors[3], label="DMP", linestyle="-")
plt.plot(sessions, ours, marker=markers[4], color=colors[4], label="Ours", linestyle="-")


# # 添加水平虚线
# plt.axhline(y=95, color="black", linestyle="dashed")

# 设置 x 轴和 y 轴
plt.xticks(sessions, session_labels)
plt.xlabel("Stage")
plt.ylabel("AA(%)")

# 添加标题和图例
# plt.title("AA on original and modified datasets")
plt.legend()

# 添加 (c) 标注
plt.figtext(0.5, -0.15, "(c) Comparison results with new generative models", ha="center", fontsize=10)

# 显示图表
plt.tight_layout()
plt.savefig("output.png", format="png", dpi=600)
plt.show()
