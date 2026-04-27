import torch
import torch.nn as nn
import torch.optim as optim
from common.fifa_helper import get_train_data,features,get_data_custom,format_value
import numpy as np

# ========= 1. 加载真实数据 =========
x_train_origin, x_test_origin, y_train_origin, y_test_origin = get_train_data(1500)
# ========= 2. 特征标准化（非常重要） =========
# 均值和标准差
# ---- x ----
mean = x_train_origin.mean(axis=0) # 计算每一个特征的均值 mean = [age_mean, rating_mean, wage_mean, ...]
std = x_train_origin.std(axis=0)  # 计算每一个特征的标准差 std = x_train.std(axis=0)
std[std == 0] = 1.0   # 防止除零

x_train_scaled = (x_train_origin - mean) / std #除std 让所有特征 ≈ 同一量级
x_test_scaled  = (x_test_origin  - mean) / std

# ---- y ----
y_mean = y_train_origin.mean()
y_std = y_train_origin.std()
if y_std == 0:
    y_std = 1.0

y_train_scaled = (y_train_origin - y_mean) / y_std
y_test_scaled  = (y_test_origin  - y_mean) / y_std

# ========= 3. 转成 torch =========
# 全部是标准化以后的数据
x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)

x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)
 
# ========= 4. 定义模型 =========
# model = nn.Sequential(
#     nn.Linear(len(features), 64), # 把len(features)维特征映射到64维隐藏空间
#     nn.ReLU(), #引入非线性 缓解梯度消失（相比 sigmoid）没有ReLU相当于 h1=w1x+b1;h2=w2h1+b2;h3=w3h2+b3, 永远是线性
#     nn.Dropout(0.2),#训练时随机关掉 20% 神经元，用来防止 FIFA19 模型过拟合。
#     nn.Linear(64, 1) #输入特性64，输出特性1; 解释上面Dropout [ h1, h2, h3, ..., h64 ], 随机选13个h=0
# )

model = nn.Sequential(
    nn.Linear(len(features), 128), # W初始是很小默认生成的随机数; h1 = [2.1, -0.3, 0.8, -1.5, 0.4, ...]  (128维)
    nn.ReLU(), #Rectified Linear Unit；  ReLU(x)=max(0,x) 负信号：丢弃， 正信号：保留
    nn.Linear(128, 64), # h2_relu = [0, 1.5, 0, 0.9, ...]   (64维)
    nn.ReLU(),
    nn.Linear(64, 1) # h3 = [3.47]  # 标量（在标准化空间）
)

# ========= 5. 损失函数 & 优化器 =========
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # lr: learning rate

model.train()
for epoch in range(500):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # h = W @ x + b ；每一次训练，W 都在微调 W = W - learning_rate × ∂loss/∂W
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1:5d}/100] | "
            f"Loss: {loss.item():.6f} | "
        )


model.eval() # 不做梯度计算​ 保证推理结果稳定
with torch.no_grad(): # 测试时反标准化 
    y_test_pred_scaled = model(x_test)
    test_loss = criterion(y_test_pred_scaled, y_test)

    # 反标准化
    y_test_pred = y_test_pred_scaled * y_std + y_mean
    y_test_real = y_test * y_std + y_mean
    

torch.set_printoptions(sci_mode=False)

print("Test Loss (scaled):", test_loss.item())
print("Predicted:", y_test_pred[:5])
print("Actual:   ", y_test_real[:5])


# =========================================================
# 7. 自定义数据推理
# =========================================================
x_inference_origin = get_data_custom()
print(x_inference_origin)
# 特征对齐
x_inference_scaled = (x_inference_origin - mean) / std
x_inference = torch.tensor(x_inference_scaled, dtype=torch.float32)
 
# 推理
model.eval()
with torch.no_grad():
    y_inference_scaled = model(x_inference)
    y_inference = y_inference_scaled * y_std + y_mean

# =========================================================
# 8. 输出结果
# =========================================================

for i in range(len(x_inference)):
    print(
        # f"{df.loc[i,'Nationality']} "
        # f"{df.loc[i,'Position']} | "
        f"Predicted Value: {format_value(y_inference[i].item())}"
    )