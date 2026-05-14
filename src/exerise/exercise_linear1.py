import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 实例化模型、损失函数和优化器            
x_np = np.random.randn(100, 1) # [100, 1]  batch size=100 表示“每100条样本更新一次模型参数”; 1 表示特征数为1
y_np = 2 * x_np + 3 # [100, 1]
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

model = nn.Linear(1, 1) # nn.Linear(2, 1)=输入2个特征，输出1个值的线性模型y=w1x1+w2x2+b, Linear(1, 1)是输入1个特征y=wx+b
criterion = nn.MSELoss(reduction='mean') # MSE = Mean Squared Error（均方误差）Math.sqrt(outputs-y).mean()   
optimizer = optim.SGD(model.parameters(), lr=0.1) # SGD = Stochastic Gradient Descent（随机梯度下降）

# --- 进入训练循环 ---
for _ in range(100):
    # 2. 前向传播：将数据输入模型，得到预测结果
    outputs = model(x)

    # 3. 计算损失：使用 criterion 比较预测值和真实值
    # 这是 MSELoss 对象的核心用途
    # outputs 和 y 的形状必须一致
    loss = criterion(outputs, y)

    # 4. 清空过往梯度
    # 每次反向传播前必须执行，否则梯度会累积
    optimizer.zero_grad()

    # 5. 反向传播：计算损失相对于模型所有参数的梯度
    # PyTorch 的自动求导引擎会从 loss 开始，
    # 沿着计算图反向计算，填充所有参数的 .grad 属性
    # .grad是 PyTorch 中保存梯度值的属性: w.grad(∂L/∂w) b.grad(∂L/∂b)
    loss.backward()

    # 6. 更新参数：优化器根据计算出的梯度来更新模型的权重
    # optimizer 会查看每个参数的 .grad 值，并根据其学习率更新参数值
    # w -= lr * w.grad
    # b -= lr * b.grad
    optimizer.step()

print(f"loss: {loss.item():.4f}") # 目标loss ≈ 0
print(model.weight.item(), model.bias.item())


with torch.no_grad():
    x_test = torch.tensor([[10.0], [5.0], [6.0]])
    y_pred = model(x_test)
 
print(y_pred.numpy())

 
# shape = [batch_size, in_features]
 

# 2. 自定义正态分布 (均值=80, 标准差=20)
# x = torch.normal(80.0, 20.0, size=(10,))
# x2 = torch.clamp(x, min=0, max=100)