import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 GPU，避免 NCCL 报错

import numpy as np
import matplotlib.pyplot as plt
import torch
from GaussianProcess import GaussianProcessCBF  # 确保路径和类名一致

# === 参数设置 ===
MODEL_PATH = "gp_model_sigma.pt"   # 模型文件路径
Z_FIXED = 1.0                      # 固定高度
VEL_FIXED = np.zeros(3)           # 固定速度向量
GRID_RES = 60                     # 网格精度
XY_RANGE = (0.0, 1.5)             # x/y 范围

# === 加载模型 ===
print(f"Loading GP model from {MODEL_PATH}")
gp_model = GaussianProcessCBF.load(MODEL_PATH, input_dim=6)

# === 构造查询点 ===
x_vals = np.linspace(*XY_RANGE, GRID_RES)
y_vals = np.linspace(*XY_RANGE, GRID_RES)
xx, yy = np.meshgrid(x_vals, y_vals)
grid_xy = np.c_[xx.ravel(), yy.ravel()]
X_query = np.zeros((len(grid_xy), 6))  # 6维输入：x, y, z, vx, vy, vz
X_query[:, 0:2] = grid_xy
X_query[:, 2] = Z_FIXED
X_query[:, 3:] = VEL_FIXED

# === 预测 σ²(x) ===
print("Predicting σ²(x)...")
with torch.no_grad():
    _, cov = gp_model.predict(torch.tensor(X_query, dtype=torch.float32))
sigma2 = torch.diagonal(cov).unsqueeze(-1)  # 提取对角线 σ²
sigma2 = sigma2.detach().numpy().reshape(xx.shape)

# === 可视化 ===
plt.figure(figsize=(8, 6))
cs = plt.contourf(xx, yy, sigma2, levels=50, cmap='viridis')
cbar = plt.colorbar(cs)
cbar.set_label("Predicted Variance $\sigma^2(x)$")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title(f"GP Predicted Variance at z={Z_FIXED}, v={VEL_FIXED.tolist()}")
plt.tight_layout()
plt.savefig("gp_variance_map.png", dpi=300)
plt.show()

print("✅ 可视化完成，图像已保存为 gp_variance_map.png")
