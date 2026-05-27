import numpy as np
import pandas as pd
from config import FEATURES_CAT

def parse_value(v):
    if pd.isna(v):
        return np.nan
    v = str(v).replace("€", "").strip()
    if "M" in v:
        return float(v.replace("M", "")) * 1e6
    elif "K" in v:
        return float(v.replace("K", "")) * 1e3
    return float(v)


def parse_height(h):
    try:
        ft, inch = h.split("'")
        return int(ft)*30.48 + int(inch)*2.54
    except:
        return np.nan

def value_transform(df: pd.DataFrame, cat_vocab_idx):
    df = df.copy()
    df["value_num"] = df["Value"].apply(parse_value)
    df["height_cm"] = df["Height"].apply(parse_height)
    df["weight_kg"] = (
        df["Weight"].str.replace("lbs", "", regex=False).astype(float) * 0.453592
    )

    for col in FEATURES_CAT:
        df[f"{col}_enc"] = df[col].map(cat_vocab_idx[col])

    return df