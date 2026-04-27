import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

num_samples = 200
# NumPy 生成数据
x1_np = np.random.randn(num_samples, 1)
x2_np = np.random.randn(num_samples, 1)
x_np = np.hstack([x1_np, x2_np]) # 列方向拼接, 参考下面举例
# x1:
# [[a1],
#  [a2],
#  ...
#  [a200]]
# x2:
# [[b1],
#  [b2],
#  ...
#  [b200]]
# x:
# [[a1, b1],
#  [a2, b2],
#  ...
#  [a200, b200]] 
noise = np.random.randn(num_samples, 1) * 0.5 #从 N(0, 0.25) 采样的噪声，用来模拟真实世界的不确定性 loss ≈ 0.25
y_np = 2 * x1_np + 3 * x2_np + 5 + noise

# 转成 torch
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float() # .unsqueeze(1)语法上不必须，因为已经是N(200,1)

model = nn.Linear(in_features=2, out_features=1)

#损失函数 优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(200):
    # 前向传播
    y_pred = model(x)

    # 损失
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"{epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    w = model.weight
    b = model.bias

print("Learned weights:", w.weight)
print("Learned bias:", b.item())