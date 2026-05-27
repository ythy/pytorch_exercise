# dataset.py
import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class PretrainDataset(Dataset):

    # max_len： 一条样本最多 32 个 token，(学习是一件很重要的事情: len=10)
    def __init__(self, txt_path, tokenizer_path, max_len=32):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)

        self.pad_id = 0
        self.max_len = max_len
    
        with open(txt_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        self.samples = []
        for text in texts:
            ids = self.tokenizer.encode(text, out_type=int)
            if len(ids) < 2:
                continue
            self.samples.append(ids[:max_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len

        x = ids[:-1]
        y = ids[1:]
        mask = [1] * (self.max_len - 1)

        if pad_len > 0:
            mask[-pad_len:] = [0] * pad_len

        return (
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(mask)
        )