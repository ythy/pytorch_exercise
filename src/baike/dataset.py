# data/dataset.py
import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class QADataset(Dataset):
    def __init__(self, path, tokenizer_path, max_len=64):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.max_len = max_len
        self.file = path
        self.offsets = []

        with open(path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            q, a = line.strip().split("\t", 1)

        text = f"<s>{q}</s>{a}"
        ids = self.sp.Encode(text, out_type=int)[:self.max_len]
        return ids[:-1], ids[1:]

# 补位用 collate_fn= 把一个 batch 里的变长样本，变成“等长张量”的函数
def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    def pad(seq, val):
        return [val] * (max_len - len(seq)) + seq

    return (
        torch.tensor([pad(x, 0) for x in xs]),
        torch.tensor([pad(y, 0) for y in ys]),
    )