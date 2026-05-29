# model.py
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model


class SmallLM(GPT2LMHeadModel):
    def __init__(self, vocab_size):
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=256,
            n_head=4,
            n_layer=2,
            n_positions=128,
            bos_token_id=1,
            eos_token_id=2,
            attn_implementation="eager",
            use_cache=False,
            loss_type="for_causal_lm",
        )
        super().__init__(cfg)



class SmallLM_bak(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=256,
            n_head=4,
            n_layer=2,
            n_positions=128,
            bos_token_id=1,
            eos_token_id=2,
            attn_implementation="eager",
            use_cache=False,      # ✅ 推理安全
        )

        self.transformer = GPT2Model(cfg)
        self.lm_head = nn.Linear(256, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        outputs = self.transformer(input_ids=input_ids)
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # ✅ 忽略 pad（-100）
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
        }