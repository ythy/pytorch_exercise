# dataset.py
import torch
import json
from torch.utils.data import Dataset
 
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_id
        self.samples = self._load(jsonl_path)

    def _load(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            conv = []
            for line in f:
                obj = json.loads(line)
                conv.append(obj)
                if obj["role"] == "assistant":
                    data.append(conv)
                    conv = []
        return data

    def __getitem__(self, idx):
        conv = self.samples[idx]
        ids = build_sft_text(conv, self.tokenizer)

        first_eos = ids.index(self.tokenizer.eos_id)

        x = ids[:first_eos + 1]
        y = ids[first_eos + 1:]

        # 截断
        x = x[:self.max_len]
        y = y[:self.max_len]

        # padding
        while len(x) < self.max_len:
            x.append(self.pad_id)
        while len(y) < self.max_len:
            y.append(self.pad_id)

        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        return len(self.samples)

     


def build_sft_text(conv, tokenizer):
    """
    conv = [
      {"role":"user","content":"你是谁？"},
      {"role":"assistant","content":"我是你的对话助手。"}
    ]
    """
    bos = tokenizer.bos_id
    eos = tokenizer.eos_id

    ids = [bos]
    ids += tokenizer.sp.encode(conv[0]["content"], out_type=int)
    ids.append(eos)
    ids += tokenizer.sp.encode(conv[1]["content"], out_type=int)
    ids.append(eos)

    return ids