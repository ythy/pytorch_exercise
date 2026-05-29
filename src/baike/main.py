import torch
from config import *
from model import SmallLM
from dataset import QADataset
import argparse
from train_tokenizer import spm_train
import sentencepiece as spm
from tqdm import tqdm
torch._dynamo.disable()
from transformers import GenerationConfig

BOS_ID = 1
EOS_ID = 2

generation_config = GenerationConfig(
    max_new_tokens=32,
    min_new_tokens=1,

    num_beams=4,
    early_stopping=True,

    do_sample=False,

    eos_token_id=EOS_ID,
    pad_token_id=EOS_ID,
    bos_token_id=BOS_ID,
)

def train():
    dataset = QADataset(TRAIN_FILE, TOKENIZER_PATH, MAX_LEN)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = SmallLM(VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    low_loss_steps = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_tokens = 0

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()

            if step % 10 == 0:
                print(
                    f"[Epoch {epoch}] "
                    f"Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Tokens: {total_tokens} | "
                    f"Loss steps: {low_loss_steps}"
                )
                if loss.item() < MIN_LOSS:
                    low_loss_steps += 1
                    if low_loss_steps >= PATIENCE_STEPS:
                        print(
                            f"\n🛑 Stopped early: "
                            f"loss < {MIN_LOSS} for {PATIENCE_STEPS} steps."
                        )
                        torch.save(model.state_dict(), MODEL_SAVE_PATH)
                        print("✅ Model saved.")
                        return
                else:
                   low_loss_steps = 0 
               
        avg_loss = total_loss / len(loader)
        print(f"\n✅ Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Model saved.")


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
 

def load_model():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    model = SmallLM(VOCAB_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, sp


def generate_answer(model, sp, question):
    prompt = f"<s>问：{question}</s>答："
    assert prompt.endswith("答："), "prompt corrupted"

    input_ids = sp.encode(prompt, out_type=int)
    assert input_ids[-1] != EOS_ID, "EOS leaked into prompt!"

    input_ids = torch.tensor([input_ids], device=DEVICE)

    # prompt = f"<s>问：{question}</s>答："
    # input_ids = torch.tensor(
    #     [sp.encode(prompt, out_type=int)],
    #     device=DEVICE
    # )

    output_ids = model.generate(
        input_ids,
        generation_config=generation_config
    )

    # 只取生成部分
    gen_ids = output_ids[0][input_ids.size(1):]
    answer = sp.decode(gen_ids.tolist())

    return answer.strip()


val_qa = [
    ("0531 是我国哪个城市的电话区号？", "济南"),
    ("世界上最大的蝴蝶是哪一种？", "南美凤蝶"),
    ("香港特别行政区规定汽车在道路的哪边行驶？", "右"),
    ("“AK47”是NBA哪位俄罗斯籍球星外号？", "安德烈·基里连科"),
    ("“BRA”是哪个美洲国家的英文缩写？", "巴西"),
    ("“avi”是什么类型文件的格式？", "视频"),
    ("诗句“每逢佳节倍思亲”中的“佳节”指的是哪一个节日？", "重阳节"),
]

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "pretrain", "valid_pretrain", "predict", "tokenizer", "check"], required=True)
    args = parser.parse_args()
    
    if args.mode == "tokenizer":
        spm_train("data/chinese_baike_qa.txt", "data/tokenizer/chinese_baike", 4400)

    elif args.mode == "train":
        train()

    elif args.mode == "check":
        check_loss()

    elif args.mode == "predict":
        model, sp = load_model()

        for q, gt in val_qa:
            pred = generate_answer(model, sp, q)
            status = "✅" if pred.strip() == gt.strip() else "❌"

            print(f"{status} Q: {q}")
            print(f"   GT : {gt}")
            print(f"   Pred: {pred}\n")