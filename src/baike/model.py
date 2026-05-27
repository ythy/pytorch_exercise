# model.py
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class SmallLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=256, # D
            n_head=4,
            n_layer=2,
            n_positions=128,# T 不够需要补位到128
            bos_token_id=1,
            eos_token_id=2,
            attn_implementation="eager"
        )
        self.transformer = GPT2Model(cfg)
        self.lm_head = nn.Linear(256, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        self.transformer.eval()  # ✅ 关 dropout
        out = self.transformer(input_ids=input_ids)
        logits = self.lm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return {"loss": loss, "logits": logits}