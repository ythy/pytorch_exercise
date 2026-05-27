import torch
import torch.optim as optim
from config import *
from model import SmallLM
from dataset import QADataset, collate_fn
import argparse
from train_tokenizer import spm_train
import sentencepiece as spm
import os
torch._dynamo.disable()

def train(resume=True):
    ds = QADataset(TRAIN_FILE, TOKENIZER_PATH, MAX_LEN)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    model = SmallLM(VOCAB_SIZE)

    # ========== ✅ 断点续训 ==========
    start_epoch = 0
    if resume and os.path.exists(MODEL_SAVE_PATH):
        print("✅ 加载已有模型，继续训练")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("🆕 从头开始训练")

    model.train()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0

        for step, (x, y) in enumerate(dl):
            optimizer.zero_grad()
            loss = model(x, labels=y)["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        avg = total_loss / len(dl)
        print(f"Epoch {epoch} | Avg Loss: {avg:.4f}")

        # ========== ✅ 保存（覆盖式，但安全） ==========
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # ✅ 强烈建议：顺手留一个副本
        torch.save(model.state_dict(), f"model_epoch{epoch}.pth")

    print("✅ 训练完成")
 

def ask2(
    question,
    max_new_tokens=8, #  最多只生成 ? 个 token
    temperature=0.5,
    top_k=50,       # ✅ 必须给一个合理值
    top_p=0.9       # ✅ 收紧
):
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    model = SmallLM(VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    BOS_ID = 1
    EOS_ID = 2

    prompt = f"<s>{question}</s>"
    prompt_ids = sp.Encode(prompt, out_type=int)

    ids = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([ids]))["logits"]
            next_logit = logits[0, -1]

            # ✅ 禁止再生 <s>
            next_logit[BOS_ID] = float("-inf") 

            # ✅ temperature
            next_logit = next_logit / max(temperature, 1e-5)

            # ✅ top-k
            if top_k > 0:
                k = min(top_k, next_logit.size(-1))
                values, _ = torch.topk(next_logit, k)
                next_logit[next_logit < values[-1]] = float("-inf")

            # ✅ top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_logits[cumulative_probs > top_p] = float("-inf")
                next_logit.scatter_(0, sorted_indices, sorted_logits)

            # ✅✅✅ 防炸兜底（核心）
            if torch.all(next_logit == float("-inf")):
                next_logit = torch.zeros_like(next_logit)

            probs = torch.softmax(next_logit, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == EOS_ID:
                break

            ids.append(next_token)

    gen_ids = ids[len(prompt_ids):]
    return sp.Decode(gen_ids)

def ask(
    question,
    max_new_tokens=6,   # QA 一般很短
):
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    model = SmallLM(VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    BOS_ID = 1
    EOS_ID = 2

    prompt = f"<s>{question}</s>"
    prompt_ids = sp.Encode(prompt, out_type=int)
    ids = prompt_ids[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([ids]))["logits"]
            next_logit = logits[0, -1]

            # ✅ 禁止 <s>
            next_logit[BOS_ID] = float("-inf")

            # ✅ 禁止重复上一个 token
            if len(ids) > 0:
                next_logit[ids[-1]] = float("-inf")

            # ✅ 防炸
            if torch.all(next_logit == float("-inf")):
                break

            # ✅✅✅ 关键：用 greedy，不用 sampling
            next_token = torch.argmax(next_logit).item()

            if next_token == EOS_ID:
                break

            ids.append(next_token)

    gen_ids = ids[len(prompt_ids):]
    return sp.Decode(gen_ids)

def check_loss():
    # ========== 1. 加载模型 ==========
    model = SmallLM(VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    # ========== 2. 加载一小部分训练数据 ==========
    ds = QADataset(TRAIN_FILE, TOKENIZER_PATH, MAX_LEN)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ========== 3. 计算 loss ==========
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for x, y in dl:
            loss = model(x, labels=y)["loss"]
            total_loss += loss.item()
            steps += 1

            if steps >= 200:   # 只看一部分，快
                break

    print(f"\n✅ 当前模型在训练集上的平均 Loss: {total_loss / steps:.4f}\n")
 


val_qa = [
    ("“0531”是我国山东省哪个城市的电话区号？", "济南"),
    ("“0571”是我国浙江省哪个城市的电话区号？", "杭州"),
    ("“1、2、3、4、5、6、7”七个唱名的发明者来自？", "法国"),
    ("“20世纪最伟大运动员”之一贝利是什么运动员？", "足球"),
    ("“433”“352”“442”阵型是哪种运动的术语？", "足球"),
    ("“586电脑”和“奔腾电脑”是一回事吗？", "不是"),
    ("“606”是杀菌剂，请问“606”代表什么？", "试验编号"),
    ("“7UP”是我们常喝的哪种碳酸饮料？", "七喜"),
    ("“863”计划名称的由来是？", "1986年3月制定"),
    ("+、-、*、/、=”符号是由一个人发明的吗？", "不是"),
    ("“√”是英文“right”（正确）一词的第一个字母“r”的简化？", "对"),
]


if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "pretrain", "valid_pretrain", "predict", "tokenizer", "check"], required=True)
    args = parser.parse_args()
    
    if args.mode == "tokenizer":
        spm_train("data/chinese_baike_corpus.txt", "data/tokenizer/chinese_baike", 4400)

    elif args.mode == "train":
        train()

    elif args.mode == "check":
        check_loss()

    elif args.mode == "predict":
        for q in val_qa:
            print("Q:", q)
            print("A:", ask(q))
            print("-" * 40)