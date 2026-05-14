import torch
import torch.nn as nn
import torch.optim as optim
from helper import (
    HybridPlayerModel,
    load_model,
    save_model,
    format_value,
    )
import numpy as np
import argparse
from loader import (
    prepare_data,
    get_data_custom,
)

# python src/fifa/main.py --mode train

def train(data):
    model = HybridPlayerModel(
        data["cat_vocab_sizes"],
        num_feat_dim=data["x_num"].shape[1]
    )

    # ========= 4. 损失函数 & 优化器 =========
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr: learning rate

    model.train()
    for epoch in range(500):
        y_pred = model(data["x_cat"], data["x_num"])
        loss = criterion(y_pred, data["y_num"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # h = W @ x + b ；每一次训练，W 都在微调 W = W - learning_rate × ∂loss/∂W
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1:5d}/100] | "
                f"Loss: {loss.item():.6f} | "
            )
    return model


def validate(model, ckpt, data):
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

 

# =========================================================
# 自定义数据推理
# =========================================================
def predict(model, ckpt, data):
    cat_x_inf, num_x_inf = data
    with torch.no_grad():
        y = model(cat_x_inf, num_x_inf)
        y = y * ckpt["y_std"] + ckpt["y_mean"]

    print("Predictions:")
    for v in y:
        print(format_value(v.item()))
     

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "validate", "predict"], required=True)
    args = parser.parse_args()
   
    
    if args.mode == "train":
        data = prepare_data(5000)
        model = train(data)
        save_model("dist/fifa_model.pt", model, data)
        ### 同步验证
        model_v, ckpt = load_model("dist/fifa_model.pt")
        data_v = prepare_data(50, ckpt)
        validate(model_v, ckpt, data_v)

    elif args.mode == "validate":
        model, ckpt = load_model("dist/fifa_model.pt")
        data = prepare_data(50, ckpt)
        validate(model, ckpt, data)

    elif args.mode == "predict":
        model, ckpt = load_model("dist/fifa_model.pt")
        data = get_data_custom(ckpt)
        predict(model, ckpt, data)

 