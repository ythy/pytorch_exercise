# model.py
import torch.nn as nn
import torch

class PretrainModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.d_model = 256
        self.max_len = 512

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_len, self.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(self.d_model, vocab_size)

    def forward(self, x, mask=None):
        B, T = x.shape

        if T > self.max_len:
            raise ValueError("seq too long")

        pos = torch.arange(T, device=x.device).clamp(max=self.max_len - 1)
        x = self.embedding(x) + self.pos_embedding(pos)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0
            if key_padding_mask.all():
                print("💥 mask all True")
                return torch.zeros_like(x)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.fc(x)