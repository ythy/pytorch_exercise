import numpy as np
import pandas as pd
import torch

 

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
    

 