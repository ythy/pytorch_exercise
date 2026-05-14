import torch
import numpy as np
import pandas as pd


FEATURES_NUM = [
    "Overall", "Potential", "Age", "height_cm", "weight_kg",
]
FEATURES_CAT = ["Nationality", "Position", "Club"]

def get_data_custom(ckpt):
    cat_vocab_idx = ckpt["vocab_idx"]
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
    num_idx = [0,1,2,3,4]
    cat_idx = {
        "Nationality": 5,
        "Position": 6,
        "Club": 7,
    }
    print("Raw inference data:\n", x_raw)
    # 特征对齐
    # ---- 数值部分 ----
    num_x = x_raw[:, num_idx].astype(np.float32)
    num_x = (num_x -  ckpt["x_mean"][num_idx]) / ckpt["x_std"][num_idx]
    num_x_inf = torch.tensor(num_x, dtype=torch.float32)

    # ---- 类别部分（token id）----
    cat_x_inf = {
        col: torch.tensor(x_raw[:, cat_idx[col]], dtype=torch.long)
        for col in FEATURES_CAT
    }

    return cat_x_inf, num_x_inf

def read_train_data(count: int) -> tuple[pd.DataFrame, dict, dict]:
    df_origin = pd.read_csv("./data/FIFA19.csv", index_col=0)
    cat_vocab_idx, cat_vocab_sizes = build_category_mappings(df_origin)
    df_final = df_origin.iloc[:count]
    return value_transform(df_final, cat_vocab_idx), cat_vocab_idx, cat_vocab_sizes

def split_data(dp:pd.DataFrame, size: int = 20, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = dp.shape[0] #x是 (n_samples, n_features)的NumPy数组
    indices =  rng.permutation(n_samples) #打乱顺序
    return dp.iloc[indices[:size]]

def prepare_data(count: int, pre_data: dict | None = None):
    # ========= 1. 加载真实数据 =========
    all_data, cat_vocab_idx, cat_vocab_sizes = read_train_data(count)
    # ========= 2. 特征标准化 & 转成torch =========
    # 注意数值特征需要标准化，类别特征不需要标准化
    # ---- x ----
    x_num = all_data[FEATURES_NUM].fillna(0).values
    x_mean = x_num.mean(axis=0) # 计算每一个特征的均值 mean = [age_mean, rating_mean, wage_mean, ...]
    x_std = x_num.std(axis=0)  # 计算每一个特征的标准差 std = x_train.std(axis=0)
    x_std[x_std == 0] = 1.0   # 防止除零

    # ---- 数值部分（标准化）----
    if not pre_data:
        x_num = torch.tensor((x_num - x_mean) / x_std, dtype=torch.float32)
    else:
        x_num = torch.tensor((x_num - pre_data["x_mean"]) /  pre_data["x_std"], dtype=torch.float32)
    # ---- 类别部分（token id）----
    x_cat = {
        col: torch.tensor(all_data[f"{col}_enc"].values, dtype=torch.long)
        for col in FEATURES_CAT
    }
    # ---- y ----
    y_num = all_data["value_num"].fillna(0).values
    y_mean = y_num.mean()
    y_std = y_num.std()
    if y_std == 0:
        y_std = 1.0
    if not pre_data:
        y_num = torch.tensor((y_num - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
    else:
        y_num = torch.tensor((y_num - pre_data["y_mean"]) / pre_data["y_std"], dtype=torch.float32).unsqueeze(1)
    
    return {
        "x_num": x_num,
        "x_cat": x_cat,
        "y_num": y_num,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "cat_vocab_idx": cat_vocab_idx, 
        "cat_vocab_sizes": cat_vocab_sizes
    }


def build_category_mappings(
    df: pd.DataFrame,
) -> tuple[dict, dict]:
    cat_vocab_idx: dict = {}
    cat_vocab_sizes: dict = {}

    for col in FEATURES_CAT:
        unique_vals = df[col].unique()
        cat_vocab_idx[col] = {
            val: idx for idx, val in enumerate(unique_vals)
        }
        cat_vocab_sizes[col] = len(unique_vals)

    return cat_vocab_idx, cat_vocab_sizes

def value_transform(data_frame:pd.DataFrame, cat_vocab_idx:dict):
    df = data_frame.copy()
    df["value_num"] = df["Value"].apply(parse_value)
    df["value_log"] = np.log1p(df["value_num"])
    df["height_cm"] = df["Height"].apply(parse_height)
    df["weight_kg"] = (
        df["Weight"]
        .str.replace("lbs", "", regex=False)
        .astype(float)
        * 0.453592
    )

    for col in FEATURES_CAT:
        mapping = cat_vocab_idx[col]
        df[f"{col}_enc"] = df[col].map(mapping)

    return df

#处理目标值
def parse_value(v):
    if pd.isna(v):
        return np.nan

    v = str(v).replace("€", "").strip()

    if "M" in v:
        return float(v.replace("M", "")) * 1e6
    elif "K" in v:
        return float(v.replace("K", "")) * 1e3
    else:
        return float(v)
    
def parse_height(h):
    try:
        ft, inch = h.split("'")
        return int(ft)*30.48 + int(inch)*2.54
    except:
        return np.nan