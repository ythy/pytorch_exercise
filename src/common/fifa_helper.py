import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
 
features = [
    "Overall", "Potential", "Age", "height_cm", "weight_kg",
    "Nationality_enc", "Position_enc"
]
cat_origin_features = ["Nationality", "Position"]

num_idx = [0,1,2,3,4]
cat_idx = {
   "Nationality": 5,
   "Position": 6 
}

global_cat2idx = {}
cat_vocab_sizes = {}

def get_data_custom():
    return [
    [
        78., 85., 21., 170.18, 72.12,
        global_cat2idx["Nationality"]["Argentina"],
        global_cat2idx["Position"]["RF"],
    ],
    [
        85., 89., 28., 185.42, 75.29,
        global_cat2idx["Nationality"]["Brazil"],
        global_cat2idx["Position"]["RCB"],
    ],
    [
        92., 93., 31., 200.66, 83.01,
        global_cat2idx["Nationality"]["China PR"],
        global_cat2idx["Position"]["GK"],
    ],
    [
        92., 93., 31., 200.66, 83.01,
        global_cat2idx["Nationality"]["Argentina"],
        global_cat2idx["Position"]["ST"],
    ]
]
 

def get_train_data(count = 1000, test_count = 20)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global global_cat2idx
    df_origin = pd.read_csv("./data/FIFA19.csv", index_col=0)
    df_slice = df_origin.iloc[:count] 
    for col in cat_origin_features:
        unique_vals = df_slice[col].unique()
        global_cat2idx[col] = {
            v: i for i, v in enumerate(unique_vals)
        }
        # 记录 vocab size
        cat_vocab_sizes[col] = len(unique_vals)
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
    global global_cat2idx
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

    for col in cat_origin_features:
        mapping = global_cat2idx[col]
        df[col + "_enc"] = df[col].map(mapping)

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


class FIFA_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, input_dim)
        h1 = self.fc1(x)
        # h1 = [2.1, -0.3, 0.8, -1.5, 0.4, ...]  (128维)

        h1_relu = self.relu1(h1)
        # ReLU(x)=max(0,x)

        h2 = self.fc2(h1_relu)
        # h2 = [..., 1.5, ..., 0.9, ...]  (64维)

        h2_relu = self.relu2(h2)

        out = self.fc3(h2_relu)
        # out = [3.47]  标量（在标准化空间）

        return out



class HybridPlayerModel(nn.Module):
    def __init__(
        self,
        num_feat_dim,
        d_model=32, #  the size of each embedding vector，维度
        hidden_dim=64,
        n_heads=4,
        num_layers=2
    ):
        super().__init__()
        
        # 示例 nn.Embedding(5, 3) 5个token，3个维度
        # tensor([[ 0.1296,  0.4550,  0.1673],  # token 0
        # [ 0.0925,  0.3475,  0.0279],  # token 1
        # [-0.1826,  0.2176,  0.0312],  # token 2
        # [ 0.5434, -0.1234,  0.8765],  # token 3
        # [ 0.2345,  0.6789, -0.4567]]) # token 4
        # ---- 模拟推理输入 ----
        # cat_x = torch.tensor([
        #     [3, 4],  # 阿根廷(3), GK(4)
        #     [1, 2]   # 巴西(1), ST(2)
        # ])
        # shape: (2, 2)
        # ---- 模拟推理输出 ----
        # tensor([[[ 0.5434, -0.1234,  0.8765],  # token 3 (阿根廷)
        #  [ 0.2345,  0.6789, -0.4567]],  # token 4 (GK)
        #
        # [[ 0.0925,  0.3475,  0.0279],  # token 1 (巴西)
        #  [-0.1826,  0.2176,  0.0312]]]) # token 2 (ST)
        # torch.Size([2, 2, 3])
        # ---- Transformer 分支 ----
        # embedding: 一个从离散集合到连续向量空间的映射，把一个单词 嵌入​ 到向量空间
        # 离散集合 {阿根廷, 巴西, 中国, …}
        # 向量空间 R^32 → 举例 0.1296, 0.4550, 0.1673, 0.5434, ..., 0.2345]  # 32 个实数
        # embedding 函数 f(阿根廷) = [0.54, -0.12, ..., 0.88]
        # 输入: 单词"Argentina" → token id 3
        # 输出: 向量 [0.54, -0.12, ..., 0.88]
        # 每个特征有自己的 embedding

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, d_model)
            for name, vocab_size in cat_vocab_sizes.items()
        })
 
        # torch.zeros创建了一个“全 0 的三维张量” 为每一个 token 位置准备一个“可学习的空向量”
        # 用来作为 可学习的位置编码（positional encoding）的初始值
        # torch.zeros(
        #     1,               # 第 0 维
        #     max_len,         # 第 1 维
        #     d_model          # 第 2 维
        # )  1, 2, 4
        # [
        #     [
        #         [0, 0, 0, 0],   # pos 0
        #         [0, 0, 0, 0],   # pos 1

        #     ]
        # ]
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, len(cat_origin_features), d_model)
        )

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # the number of expected features  向量维度 32 每个 token 的向量长度，阿根廷向量 = [x₁, x₂, ..., x₃₂]
            # 4个并行注意力头，32 维向量被切成： 每个头都看到了完整的 2 个特征
            # 头1: 阿根廷[0:8]  ↔  GK[0:8]   → 技术交互
            # 头2: 阿根廷[8:16] ↔  GK[8:16]  → 身体交互
            # 头3: 阿根廷[16:24] ↔  GK[16:24] → 名气交互
            # 头4: 阿根廷[24:32] ↔  GK[24:32] → 年龄交互
            nhead=n_heads, # the number of heads in the multiheadattention models，
            dim_feedforward=hidden_dim, # FFN 的“中间层”维度 32 → Linear(64) → ReLU → Linear(32)，  2 × d_model  # 常见
            batch_first=True #输入形状的顺序  true: (batch, seq, dim) False: (seq, batch, dim)  # 老版本
        )

        #是把多个 TransformerEncoderLayer堆叠起来
        self.transformer = nn.TransformerEncoder(
            encoder_layer, # 单层
            num_layers=num_layers # 堆叠层数 1层基础特征交互； 2层 基于第1层结果的深层交互
        )

        # ---- MLP 分支 ----
        self.mlp = nn.Sequential(
            nn.Linear(num_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        # ---- 输出层 ----
        self.fc = nn.Linear(d_model * 2, 1)

    def forward(self, cat_x_dict, num_x):
        """
        Args:
            cat_x_dict: {
                "Nationality": (batch_size,)  # token ids
                "Position": (batch_size,)     # token ids
            }
            num_x: (batch_size, num_feat_dim)
        """
        # ---- Transformer 分支 (独立 embedding) ----
        emb_list = []
        # 1. 分别对每个类别特征做 embedding
        for feat_name, feat_tensor in cat_x_dict.items():
            # unsqueeze(1) 前: [x₁, x₂, x₃, ..., x₁₅₀₀]        ← 1500 个数字
            # unsqueeze(1) 后: [[x₁], [x₂], [x₃], ..., [x₁₅₀₀]]  ← 1500 个单元素列表
            if feat_tensor.dim() == 1:
                feat_tensor = feat_tensor.unsqueeze(1)  # (B,1)
            
            # embedding 查询: (B,1) → (B,1,d_model)
            # 以 d_model = 3，B = 4 举例
            # 国籍 embedding 输出
            # emb_nation = torch.tensor([
            #     [[0.54, -0.12, 0.88]],  # 样本1: 阿根廷
            #     [[0.09, 0.35, 0.03]],   # 样本2: 巴西
            #     [[0.13, 0.46, 0.17]],   # 样本3: 中国
            #     [[-0.18, 0.22, 0.03]]   # 样本4: 西班牙
            # ])  # 形状: (4, 1, 3)
            # 位置 embedding 输出
            # emb_pos = torch.tensor([
            #     [[0.23, 0.68, -0.46]],  # 样本1: GK
            #     [[-0.18, 0.22, 0.03]],  # 样本2: ST
            #     [[0.09, 0.35, 0.03]],   # 样本3: CB
            #     [[0.54, -0.12, 0.88]]   # 样本4: CM
            # ])  # 形状: (4, 1, 3)
            emb = self.embeddings[feat_name](feat_tensor)
            emb_list.append(emb)
        
        # 2. 拼接成序列: (B,1,d_model)+(B,1,d_model) → (B,2,d_model)
        # 拼接后形状: torch.Size([4, 2, 3])
        # 拼接后值:
        # tensor([[[ 0.5400, -0.1200,  0.8800],  # 样本1: 国籍
        #         [ 0.2300,  0.6800, -0.4600]], # 样本1: 位置

        #         [[ 0.0900,  0.3500,  0.0300],  # 样本2: 国籍
        #         [-0.1800,  0.2200,  0.0300]], # 样本2: 位置

        #         [[ 0.1300,  0.4600,  0.1700],  # 样本3: 国籍
        #         [ 0.0900,  0.3500,  0.0300]], # 样本3: 位置

        #         [[-0.1800,  0.2200,  0.0300],  # 样本4: 国籍
        #         [ 0.5400, -0.1200,  0.8800]]]) # 样本4: 位置    
        cat_emb = torch.cat(emb_list, dim=1)  # (B, cat_seq_len, d_model)
        
        # 3. 加位置编码
        # 
        cat_emb = cat_emb + self.positional_encoding[:, :cat_emb.shape[1], :]
        
        # 4. Transformer
        cat_out = self.transformer(cat_emb)  # (B, cat_seq_len, d_model)
        
        # 5. 池化 (沿序列维度平均)
        # .mean(dim=1)是把每个样本的多个特征向量“平均”成一个向量
        # cat_out = torch.tensor([
        #     # 样本1
        #     [[0.6, 0.1, 1.2],   # 特征1: 国籍+位置0编码+transformer处理
        #     [0.6, 1.2, 0.1]],   # 特征2: 位置+位置1编码+transformer处理
            
        #     # 样本2
        #     [[0.2, 0.6, 0.3],
        #     [0.2, 0.7, 0.6]],
            
        #     # 样本3
        #     [[0.2, 0.7, 0.5],
        #     [0.5, 0.9, 0.6]],
            
        #     # 样本4
        #     [[-0.1, 0.4, 0.3],
        #     [0.9, 0.4, 1.5]]
        # ])
        # 输出形状: torch.Size([4, 3])
        # 输出值:
        # tensor([[0.6000, 0.6500, 0.6500],
        #         [0.2000, 0.6500, 0.4500],
        #         [0.3500, 0.8000, 0.5500],
        #         [0.4000, 0.4000, 0.9000]])
        cat_out = cat_out.mean(dim=1)  # (B, d_model)
        
        # ---- MLP 分支 (不变) ----
        num_out = self.mlp(num_x)  # (B, d_model)
        
        # ---- 融合 ----
        fused = torch.cat([cat_out, num_out], dim=1)  # (B, 2*d_model)
        
        return self.fc(fused)  # (B, 1)

