import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import os
from .hybrid_model import HybridPlayerModel

 

def save_model(path, model, data, epoch):
    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        # ✅ 模型结构
        "config": {
            "vocab_idx": data["cat_vocab_idx"], 
            "vocab_sizes": data["cat_vocab_sizes"],
            "num_feat_dim": data["x_num"].shape[1],
        },

        # ✅ 推理必须
        "stats": {
            "x_mean": data["x_mean"],
            "x_std": data["x_std"],
            "y_mean": data["y_mean"],
            "y_std": data["y_std"],
        },
    }

    torch.save(checkpoint, path)


def load_model(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HybridPlayerModel(
        cat_vocab_sizes=ckpt["config"]["vocab_sizes"],
        num_feat_dim=ckpt["config"]["num_feat_dim"]
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt
