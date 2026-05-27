import numpy as np
import torch
from config import FEATURES_CAT


def get_data_custom(ckpt):
    cat_vocab_idx = ckpt["config"]["vocab_idx"]
    
     origin_data = [
        [
            78., 85., 21., 170.18, 72.12,
            cat_vocab_idx["Nationality"]["Argentina"],
            cat_vocab_idx["Position"]["RF"],
            cat_vocab_idx["Club"]["FC Barcelona"],
        ],
        [
            85., 89., 28., 185.42, 75.29,
            cat_vocab_idx["Nationality"]["Brazil"],
            cat_vocab_idx["Position"]["RCB"],
            cat_vocab_idx["Club"]["Juventus"],
        ],
        [
            92., 93., 31., 200.66, 83.01,
            cat_vocab_idx["Nationality"]["China PR"],
            cat_vocab_idx["Position"]["GK"],
            cat_vocab_idx["Club"]["FC Barcelona"],
        ],
        [
            92., 93., 31., 200.66, 83.01,
            cat_vocab_idx["Nationality"]["China PR"],
            cat_vocab_idx["Position"]["GK"],
            cat_vocab_idx["Club"]["Fulham"],
        ],
        [
            92., 93., 31., 200.66, 83.01,
            cat_vocab_idx["Nationality"]["Argentina"],
            cat_vocab_idx["Position"]["GK"],
            cat_vocab_idx["Club"]["Fulham"],
        ],
        [
            92., 93., 31., 200.66, 83.01,
            cat_vocab_idx["Nationality"]["Argentina"],
            cat_vocab_idx["Position"]["ST"],
            cat_vocab_idx["Club"]["FC Barcelona"],
        ],
    ]

    x_raw = np.array(origin_data)

    num_idx = [0, 1, 2, 3, 4]
    cat_idx = {
        "Nationality": 5,
        "Position": 6,
        "Club": 7,
    }

    # ---- 数值特征 ----
    num_x = x_raw[:, num_idx].astype(np.float32)
    num_x = (num_x - ckpt["stats"]["x_mean"][num_idx]) / ckpt["stats"]["x_std"][num_idx]
    num_x = torch.tensor(num_x, dtype=torch.float32)

    # ---- 类别特征 ----
    cat_x = {
        col: torch.tensor(x_raw[:, cat_idx[col]], dtype=torch.long)
        for col in FEATURES_CAT
    }

    return cat_x, num_x