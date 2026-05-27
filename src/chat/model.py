import torch.nn as nn
import torch

class TinyChatModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)              # [B, T, D]
        out, _ = self.lstm(x)              # [B, T, H]
        logits = self.fc(out)              # [B, T, vocab]
        return logits



class TinyTransformerChat(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        tok = self.embedding(x)
        pos = self.pos_embedding(pos)
        x = tok + pos

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        memory = torch.zeros(B, 1, 256, device=x.device)
        out = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)