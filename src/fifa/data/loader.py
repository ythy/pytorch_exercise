import pandas as pd
import torch
from .preprocess import value_transform
from config import FEATURES_NUM, FEATURES_CAT, DATA_PATH


# ========= 1. 读原始 CSV =========
def read_train_data(count: int):
    df = pd.read_csv(DATA_PATH, index_col=0)
    cat_vocab_idx, cat_vocab_sizes = build_category_mappings(df)
    df = value_transform(df.iloc[:count], cat_vocab_idx)
    return df, cat_vocab_idx, cat_vocab_sizes


# ========= 2. 构造训练用 tensor dict =========
def prepare_raw_data(count: int, pre_data_stats: dict | None = None):
    df, cat_vocab_idx, cat_vocab_sizes = read_train_data(count)

    # ---- 数值特征 ----
    x_num = df[FEATURES_NUM].fillna(0).values
    x_mean = pre_data_stats["x_mean"] if pre_data_stats else x_num.mean(axis=0)
    x_std = pre_data_stats["x_std"] if pre_data_stats else x_num.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_num = (x_num - x_mean) / x_std

    # ---- 类别特征 ----
    x_cat = {
        col: torch.tensor(df[f"{col}_enc"].values, dtype=torch.long)
        for col in FEATURES_CAT
    }

    # ---- 目标值 ----
    y_num = df["value_num"].fillna(0).values
    y_mean = pre_data_stats["y_mean"] if pre_data_stats else y_num.mean()
    y_std = pre_data_stats["y_std"] if pre_data_stats else y_num.std()
    if y_std == 0:
        y_std = 1.0
    y_num = (y_num - y_mean) / y_std

    return {
        "x_cat": x_cat,
        "x_num": torch.tensor(x_num, dtype=torch.float32),
        "y_num": torch.tensor(y_num, dtype=torch.float32).unsqueeze(1),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "cat_vocab_idx": cat_vocab_idx,
        "cat_vocab_sizes": cat_vocab_sizes,
    }



def build_category_mappings(df: pd.DataFrame):
    cat_vocab_idx = {}
    cat_vocab_sizes = {}

    for col in FEATURES_CAT:
        vals = df[col].unique()
        cat_vocab_idx[col] = {v: i for i, v in enumerate(vals)}
        cat_vocab_sizes[col] = len(vals)

    return cat_vocab_idx, cat_vocab_sizes