import torch
from data.inference import get_data_custom
from utils import format_value

def predict(model, data, ckpt):
    cat_x, num_x = data
    with torch.no_grad():
        y = model(cat_x, num_x)
        y = y * ckpt["y_std"] + ckpt["y_mean"]

    print("Predictions:")
    for v in y:
        print(format_value(v.item()))