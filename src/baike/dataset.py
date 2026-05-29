# data/dataset.py
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


class QADataset(Dataset):
    def __init__(self, txt_path, sp_model, max_len=128):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)

        self.max_len = max_len
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        self.sep_id = self.sp.piece_to_id("</s>")
        with open(txt_path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ids = self.sp.encode(self.lines[idx], out_type=int)

        # ✅ 硬截断（含 BOS/EOS）
        if len(ids) > self.max_len - 2:
            ids = ids[:self.max_len - 2]

        if ids[0] != self.bos_id:
            ids = [self.bos_id] + ids
        if ids[-1] != self.eos_id:
            ids.append(self.eos_id)

        # ✅ 强制 padding 到 max_len
        pad_len = self.max_len - len(ids)
        ids += [self.pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()

        try:
            sep_pos = ids.index(self.sep_id)
        except ValueError:
            sep_pos = len(ids) - 1

        labels[:sep_pos + 1] = -100

        return {
            "input_ids": input_ids, # [<s>, 问, ：, 香, 港, …, 哪, 边, 行, 使, ？, </s>, 答, ：, 右, </s>]
            "labels": labels, # [-100, -100, -100, …, -100, -100, 答, :, 右, </s>]
        }