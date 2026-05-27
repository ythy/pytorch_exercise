import torch
import torch.nn as nn

# ===============================
# 模型定义
# ===============================
class TinyChatModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)                  # [B, T, E] 1, sequence length, emb_dim
        out, _ = self.lstm(x)              # [B, T, H] 1, sequence length, hidden_dim
        logits = self.fc(out)              # [B, T, V] 1, sequence length, vocab_size
        return logits



def save_model(model, word2idx, idx2word, path):
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(word2idx),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "emb_dim": model.embed.embedding_dim,
        "hidden_dim": model.fc.in_features
    }, path)

def load_model(path):
    ckpt = torch.load(path)
    model = TinyChatModel(
        vocab_size=ckpt["vocab_size"],
        emb_dim=ckpt["emb_dim"],
        hidden_dim=ckpt["hidden_dim"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    word2idx = ckpt["word2idx"]
    idx2word = ckpt["idx2word"]
    return model, word2idx, idx2word

