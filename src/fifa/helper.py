import numpy as np
import pandas as pd
import torch.nn as nn
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
    

class HybridPlayerModel(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes,
        num_feat_dim,
        d_model=32, #  the size of each embedding vector，维度
        hidden_dim=64,
        n_heads=4,
        num_layers=2
    ):
        super().__init__()
        
        # 生成特征字典
        # 示例 nn.Embedding(5, 32) 5个token 32个维度. 
        # token是特征字典中的一个key值, 比如3代表阿根廷(3), 4代表 GK(4), 不同特征最好生成独立的字典.
        # tensor([
        # [ 0.1296, ....,  0.4572],  # token 0, ∈ ℝ³²
        # [ 0.0925, ....,  0.0279],  # token 1, ∈ ℝ³²
        # [-0.1826, ....,  0.0312],  # token 2, ∈ ℝ³²
        # [ 0.5434, ....,  0.8765],  # token 3, ∈ ℝ³²
        # [ 0.2345, ...., -0.4567],  # token 4, ∈ ℝ³²
        # ])
        # ---- Transformer 分支 ----
        # embedding: 一个从离散集合到连续向量空间的映射，把一个单词 嵌入​ 到向量空间
        # 离散集合 {阿根廷, 巴西, 中国, …}
        # 向量空间 R^32 → 举例 0.1296, 0.4550, 0.1673, 0.5434, ..., 0.2345]  # 32 个实数  
        # embedding 函数 f(阿根廷) = [0.54, -0.12, ..., 0.88] ∈ ℝ³²
        # 输入: 单词"Argentina" → token id 3
        # 输出: 向量 [0.54, -0.12, ..., 0.88] ∈ ℝ³²
        # 每个特征有自己的 embedding
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(
                vocab_size,  # 词表大小
                d_model      # 每个 token 映射成 d_model 维向量 
            )
            for name, vocab_size in cat_vocab_sizes.items()
        })
 
        # torch.zeros创建了一个“全 0 的三维张量” 为每一个 token 位置准备一个“可学习的空向量”
        # 用来作为 可学习的位置编码（positional encoding）的初始值
        # torch.zeros(
        #     1,               # 第 0 维
        #     max_len,         # 第 1 维
        #     d_model          # 第 2 维
        # )  1, 3, 32
        # [
        #     [
        #         [0, ..., 0],   # pos 0 ∈ ℝ³²
        #         [0, ..., 0],   # pos 1 ∈ ℝ³²
        #         [0, ..., 0],   # pos 2 ∈ ℝ³²
        #     ]
        # ]
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, len(list(cat_vocab_sizes.keys())), d_model)
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

        # ---- Transformer 分支 (独立 embedding) ----
        emb_list = []
        # 1. 分别对每个类别特征做 embedding
        for feat_name, feat_tensor in cat_x_dict.items():
            # unsqueeze(1) 前: feat_tensor: tensor([ 1,  3, 26,  ...,  3, 11, 21])      ← 样本数量
            # unsqueeze(1) 后: feat_tensor: tensor([[ 1],[ 3],[26],...,[ 3],[11],[21]]) ← 样本数量个单元素列表
            if feat_tensor.dim() == 1:
                feat_tensor = feat_tensor.unsqueeze(1)  # (B,1)
            # embedding 转换: (B,1) → (B,1,d_model)
            # 以 d_model = 32，B = 1500  举例
            # 国籍 embedding 输出 特征相同,对应相同一组向量,例如下面的巴西
            # emb_nation = torch.tensor([
            #     [[0.54, ..., 0.88]],  # 样本1: 阿根廷  ∈ ℝ³²
            #     [[0.09,..., 0.03]],   # 样本2: 巴西  ∈ ℝ³²
            #     .....,
            #     [[0.09,..., 0.03]],   # 样本1500 巴西 ∈ ℝ³²
            # ])  # 形状: (1500, 1, 32)
            # 位置 embedding 输出
            # emb_pos = torch.tensor([
            #     [[0.23, ..., -0.46]], # 样本1: GK ∈ ℝ³²
            #     [[-0.18,..., 0.03]],  # 样本2: ST ∈ ℝ³²
            #     .....,
            #     [[0.23,..., 0.13]],   # 样本1500 ∈ ℝ³²
            # ])  # 形状: (1500, 1, 32)
            emb = self.embeddings[feat_name](feat_tensor)
            emb_list.append(emb)
        
        # 2. 拼接成序列: (B,1,d_model)+(B,1,d_model) → (B,2,d_model)
        # 拼接后形状: torch.Size([1500, 2, 32])
        # 拼接后值:
        # tensor([[[0.54, ..., 0.88],  # 样本1: 国籍 ∈ ℝ³²
        #         [0.23, ..., -0.46]],  # 样本1: 位置 ∈ ℝ³²
        #         [[0.09,..., 0.03],  # 样本2: 国籍 ∈ ℝ³²
        #         [-0.18,..., 0.03]],  # 样本2: 位置 ∈ ℝ³²
        #         .....,
        #         [[0.09,..., 0.03],  # 样本1500: 国籍 ∈ ℝ³²
        #         [0.23,..., 0.13]]]) # 样本1500: 位置 ∈ ℝ³²    
        cat_emb = torch.cat(emb_list, dim=1)  # (B, cat_seq_len, d_model)
        
        # 3. 加位置编码
        # Transformer 本身是完全并行的，必须显式地告诉模型：每个 token 在序列中的位置 
        # cat_emb.shape             (1500, 2, 32)
        # positional_encoding.shape (1, 2, 32)
        cat_emb = cat_emb + self.positional_encoding[:, :cat_emb.shape[1], :]
        # 4. Transformer
        cat_out = self.transformer(cat_emb)  # (B, cat_seq_len, d_model)
        # 5. 池化 (沿序列维度平均)
        # .mean(dim=1)是把每个样本的多个特征向量“平均”成一个向量
        # 相当于[(Nationality + Position) / 2]
        # cat_out = tensor([
        #           [[0.54, ..., 0.88],   # 样本1: 国籍+位置0编码+transformer处理 ∈ ℝ³²
        #           [0.23, ..., -0.46]],  # 样本1: 位置+位置1编码+transformer处理 ∈ ℝ³²
        #           [[0.09,..., 0.03],   # 样本2 ∈ ℝ³²
        #           [-0.18,..., 0.03]],  # 样本2 ∈ ℝ³²
        #           .....,
        #           [[0.39,..., 0.13],  # 样本1500 ∈ ℝ³²
        #           [0.23,..., 0.13]]]) # 样本1500 ∈ ℝ³²
        # 输出形状: torch.Size([1500, 32])
        # 输出值:
        # tensor([[0.6000, ..., 0.6500], ∈ ℝ³²
        #         [0.2000, ..., 0.4500], ∈ ℝ³²
        #         .....,
        #         [0.4000, ..., 0.9000]]) ∈ ℝ³²
        cat_out = cat_out.mean(dim=1)  # (B, d_model)
        
        # ---- MLP 分支 (不变) ----
        num_out = self.mlp(num_x)  # (B, d_model)
        
        # ---- 融合 ----
        fused = torch.cat([cat_out, num_out], dim=1)  # (B, 2*d_model)
        
        return self.fc(fused)  # (B, 1)

def load_model(path):
    ckpt = torch.load(path, weights_only=False)
    model = HybridPlayerModel(
        ckpt["vocab_sizes"],
        num_feat_dim=ckpt["x_num"].shape[1]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

def save_model(path, model, data):
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_sizes": data["cat_vocab_sizes"],
        "vocab_idx": data["cat_vocab_idx"],
        "x_num": data["x_num"],
        "x_mean": data["x_mean"],
        "x_std": data["x_std"],
        "y_mean": data["y_mean"],
        "y_std": data["y_std"],
    }
    torch.save(checkpoint, path)