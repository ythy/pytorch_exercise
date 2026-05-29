import torch.nn as nn
import torch

# ===============================
# 模型定义
# ===============================
class TinyChatModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, 
            hidden_dim, 
            num_layers = 2,
            batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)                  # [B, T, E] 1, sequence length, emb_dim
        out, hidden = self.lstm(x)              # [B, T, H] 1, sequence length, hidden_dim
        logits = self.fc(out)              # [B, T, V] 1, sequence length, vocab_size
        return logits, hidden


def expand_model(model, word2idx):
    old_vocab_size, emb_dim = model.embed.weight.shape
    new_vocab_size = len(word2idx)

    # ===== 新建 embedding =====
    new_embed = nn.Embedding(new_vocab_size, emb_dim)

    with torch.no_grad():
        # 从旧模型拷贝旧词
        new_embed.weight[:old_vocab_size].copy_(
            model.embed.weight
        )
        # 新词随机初始化
        new_embed.weight[old_vocab_size:].normal_(0, 1)

    # ===== 输出层 =====
    hidden_dim = model.fc.in_features
    new_fc = nn.Linear(hidden_dim, new_vocab_size)

    # 把旧模型输出层里，已经学好的“旧 token 的打分能力”完整搬过来, 新token靠训练重新学
    with torch.no_grad():
        new_fc.weight[:old_vocab_size].copy_(
            model.fc.weight
        )
        new_fc.bias[:old_vocab_size].copy_(
            model.fc.bias
        )

    model.embed = new_embed
    model.fc = new_fc

    return model