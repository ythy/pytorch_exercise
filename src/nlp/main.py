import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import PretrainDataset
from model import PretrainModel
from train_tokenizer import spm_train
import argparse
import sentencepiece as spm
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DISTRIBUTED_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

def collate_fn(batch):
    xs, ys, masks = zip(*batch)

    xs = [x for x in xs if x.sum() > 0]
    ys = [y for y in ys if y.sum() > 0]
    masks = [m for m in masks if m.sum() > 0]

    if len(xs) == 0:
        raise RuntimeError("batch 全为空")

    return torch.stack(xs), torch.stack(ys), torch.stack(masks)

def pre_train(txt_path, tokenizer_path, save_path):
    dataset = PretrainDataset(txt_path, tokenizer_path, max_len=32)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,          # ✅ 防止 batch 太小
    )

    vocab_size = dataset.tokenizer.GetPieceSize()
    model = PretrainModel(vocab_size)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,           
        weight_decay=1e-2
    )

    best_loss = float("inf")
    patience = 3
    no_improve = 0

    for epoch in range(50):
        model.train()
        total_loss = 0
        steps = 0

        for x, y, mask in loader:
            # ✅ 最后一道防线
            if x.sum() == 0 or y.sum() == 0 or mask.sum() == 0:
                continue

            if y.max() >= vocab_size:
                print("💥 label 越界")
                return

            optimizer.zero_grad(set_to_none=True)

            logits = model(x, mask)

            if torch.isnan(logits).any():
                print("💥 logits NaN")
                return

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            if torch.isnan(loss):
                print("💥 loss NaN")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.05   # ✅ 非常保守
            )

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if steps == 0:
            print("💥 没有有效 batch")
            break

        avg_loss = total_loss / steps
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("✅ 保存预训练模型")
        else:
            no_improve += 1

        if no_improve >= patience or avg_loss < 1.5:
            print("🛑 早停触发")
            break



def greedy_generate(model, tokenizer, prompt, max_len=32):
    model.eval()

    ids = tokenizer.encode(prompt, out_type=int)
    x = torch.tensor([ids])  # [1, T]

    for _ in range(max_len - len(ids)):
        with torch.no_grad():
            logits = model(x)

        next_id = logits[0, -1].argmax()

        # ✅ 必须是 [1, 1]
        next_id = next_id.view(1, 1)

        x = torch.cat([x, next_id], dim=1)

        if next_id.item() == tokenizer.eos_id():
            break

    return tokenizer.decode(x[0].tolist())

  

def load_pretrained_model_and_tokenizer(model_path, tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    model = PretrainModel(tokenizer.GetPieceSize())
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, tokenizer

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "pretrain", "valid_pretrain", "predict", "tokenizer", "test"], required=True)
    args = parser.parse_args()
    
    file_size = "10k"
    
    if args.mode == "tokenizer":
        # 1. 构建 corpus
        spm_train(f"corpus/pretrain/{file_size}.txt", f"data/tokenizer/pretrain_{file_size}", 2100)

    elif args.mode == "pretrain":
        pre_train(f"corpus/pretrain/{file_size}.txt", f"data/tokenizer/pretrain_{file_size}.model", f"dist/pretrain_{file_size}.pt")

        model, tokenizer = load_pretrained_model_and_tokenizer(f"dist/pretrain_{file_size}.pt", f"data/tokenizer/pretrain_{file_size}.model")
        prompts = [
            "今天",
            "人工智能",
            "学习",
            "猫",
        ]
        for p in prompts:
            print(p, "->", greedy_generate(model, tokenizer, p))
      

    elif args.mode == "valid_pretrain":
        model, tokenizer = load_pretrained_model_and_tokenizer()
        prompts = [
            "今天",
            "人工智能",
            "学习",
            "猫",
            "老师",
        ]
        for p in prompts:
            print(p, "->", greedy_generate(model, tokenizer, p))
 









