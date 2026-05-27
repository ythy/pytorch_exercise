import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from data_builder import jsonl_to_corpus
from tokenizer import (
    ChatTokenizer,
    spm_train
    )
from utils import (
    save_model,
    load_model
    )
from dataset import (
    SFTDataset,
    build_sft_text,
)
from model import  (
    TinyChatModel,
    TinyTransformerChat,
)


def train():
    # 1. 加载 tokenizer
    tokenizer = ChatTokenizer("data/tokenizer/chat.model")

    # 2. Dataset / DataLoader
    dataset = SFTDataset("corpus/sft_chat.jsonl", tokenizer, max_len=64)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. 模型
    model = TinyTransformerChat(tokenizer.vocab_size)
    model.train()

    # 4. 损失 & 优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 30

    for epoch in range(EPOCHS):
        for x, y in loader :
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1)
            )
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

 
    save_model(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        save_dir="dist",
        filename="chat_tokenizer_model.pt"
    )

def generate(
    model,
    tokenizer,
    prompt,
    max_len=64,
    max_answer_len=24,      # ✅ 限制回答长度
    top_k=4,
    temperature=1.1,
    repeat_penalty=1.2     # ✅ 抑制重复
):
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids])

    for _ in range(min(max_len - len(ids), max_answer_len)):
        with torch.no_grad():
            logits = model(input_ids)

        next_logits = logits[0, -1]

        # ✅ 重复惩罚（防止 不不不 / 。。。。。）
        for token_id in set(input_ids[0].tolist()):
            next_logits[token_id] /= repeat_penalty

        next_logits = next_logits / temperature

        # ✅ Top-K 采样（禁止 argmax）
        topk_logits, topk_indices = torch.topk(
            next_logits, min(top_k, next_logits.size(-1))
        )
        probs = torch.softmax(topk_logits, dim=-1)
        next_id = topk_indices[torch.multinomial(probs, 1)]

        if next_id == tokenizer.eos_id:
            break

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    # ✅ 安全取出生成部分
    gen_ids = input_ids[0].tolist()[len(ids):]

    if tokenizer.eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_id)]

    # ✅ 解码 + 清洗
    text = tokenizer.decode(gen_ids)
    text = clean_output(text)

    return text
 

def clean_output(text: str) -> str:
    import re

    # 1. 合并重复标点
    text = re.sub(r"\.{2,}", "。", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)

    # 2. 去掉句尾无意义拖尾
    text = re.sub(r"[。！？]\s*$", "。", text)

    # 3. 去掉空格
    text = text.replace(" ", "")

    return text

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "retrain", "predict", "tokenizer"], required=True)
    args = parser.parse_args()
   
    
    if args.mode == "tokenizer":
        # 1. 构建 corpus
        jsonl_to_corpus("corpus/sft_chat.jsonl", "data/chat_corpus.txt")
        # 2. SentencePiece 训练
        spm_train()   

    elif args.mode == "train":
        train()
        model, tokenizer = load_model(
            "dist/chat_tokenizer_model.pt",
            "data/tokenizer/chat.model"
        )
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="推荐一首歌",
            max_len=64,
            max_answer_len=24,
            top_k=4,
            temperature=1.1,
        )
        print(output)
 
    elif args.mode == "predict":
        model, tokenizer = load_model(
            "dist/chat_tokenizer_model.pt",
            "data/tokenizer/chat.model"
        )
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="推荐一首歌",
            max_len=64,
            max_answer_len=24,
            top_k=4,
            temperature=1.1,
        )
        print(output)