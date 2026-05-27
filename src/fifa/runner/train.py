import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import DEVICE, EPOCHS, LR
from models.checkpoint import save_model


def train(model, data, ckpt_path=None, start_epoch=1):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        y_pred = model(data["x_cat"], data["x_num"])
        loss = criterion(y_pred, data["y_num"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # h = W @ x + b ；每一次训练，W 都在微调 W = W - learning_rate × ∂loss/∂W
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1:5d}/{EPOCHS}] | "
                f"Loss: {loss.item():.6f} | "
            )

    # ✅ 保存 checkpoint
    if ckpt_path:
        save_model(ckpt_path, model, data, epoch)

def validate(model, data, ckpt):
    model.eval() # 不做梯度计算​ 保证推理结果稳定
    with torch.no_grad(): # 测试时反标准化 
        y_pred = model(data["x_cat"], data["x_num"])
        loss = nn.MSELoss()(y_pred, data["y_num"])

        y_pred_real = y_pred * ckpt["y_std"] + ckpt["y_mean"]
        y_real = data["y_num"] * ckpt["y_std"] + ckpt["y_mean"]
    
    torch.set_printoptions(sci_mode=False)
    print("Test Loss:", loss.item())
    print("Predicted:", y_pred_real[:10])
    print("Actual:", y_real[:10])
    