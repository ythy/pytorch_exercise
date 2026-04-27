import numpy as np
import pandas as pd

features = [
    "Overall", "Potential", "Age", "height_cm", "weight_kg",
    "Nationality_enc", "Position_enc"
]

global_encode_map_nationality = None
global_encode_map_position = None

def label_encode(series):
    series = series.astype(str)
    uniques = sorted(series.unique())   # 排序，保证稳定
    mapping = {v: i for i, v in enumerate(uniques)}
    encoded = series.map(mapping).values
    return encoded, mapping


def get_data_custom():
    return [
    [
        78., 85., 21., 170.18, 72.12,
        global_encode_map_nationality["Argentina"],
        global_encode_map_position["RF"]
    ],
    [
        85., 89., 28., 185.42, 75.29,
        global_encode_map_nationality["Brazil"],
        global_encode_map_position["RCB"]
    ],
    [
        92., 93., 31., 200.66, 83.01,
        global_encode_map_nationality["China PR"],
        global_encode_map_position["GK"]
    ],
    [
        92., 93., 31., 200.66, 83.01,
        global_encode_map_nationality["Argentina"],
        global_encode_map_position["ST"]
    ]
]
 

 

def get_train_data(count = 1000, test_count = 20)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_origin = pd.read_csv("./data/FIFA19.csv", index_col=0)
    df_slice = df_origin.iloc[:count] 
    df = value_transform(df_slice)
    x = df[features].fillna(0).values
    y = df["value_num"].fillna(0).values
    return train_test_split(x, y, test_count)

def train_test_split(x: np.ndarray, y: np.ndarray, 
                     test_size: int = 20, seed: int = 42
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0] #x是 (n_samples, n_features)的NumPy数组
    indices =  rng.permutation(n_samples) #打乱顺序
    test_idx = indices[:test_size] #取前test_size个
    return x, x[test_idx], y, y[test_idx]

def value_transform(data_frame:pd.DataFrame):
    global global_encode_map_nationality, global_encode_map_position
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
    #处理日期
    # df["Joined"] = pd.to_datetime(df["Joined"], errors="coerce")
    # df["joined_days"] = (df["Joined"] - df["Joined"].min()).dt.days

    cat_cols = ["Nationality", "Position"]
    df[cat_cols[0] + "_enc"], global_encode_map_nationality = label_encode(df[cat_cols[0]])
    df[cat_cols[1] + "_enc"], global_encode_map_position = label_encode(df[cat_cols[1] ])
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

def format_value(num):
    if pd.isna(num):
        return np.nan

    num = float(num)

    if num >= 1e6:
        return f"€{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"€{num / 1e3:.1f}K"
    else:
        return f"{int(num)}"
    
def parse_height(h):
    try:
        ft, inch = h.split("'")
        return int(ft)*30.48 + int(inch)*2.54
    except:
        return np.nan