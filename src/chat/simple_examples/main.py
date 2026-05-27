import torch
import torch.nn as nn
import torch.optim as optim
from helper import (
    TinyChatModel,
    save_model,
    load_model,
)
from word import(
    build_batch,
    initial_data,
    generate,
    extend_vocab,
)
import argparse


def train(sentences):
    X, Y, word2idx, idx2word = initial_data(sentences)
    vocab_size = len(word2idx)
    # ===============================
    # 初始化模型、损失、优化器
    # ===============================
    model = TinyChatModel(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    # ===============================
    # 训练模型
    # ===============================
    for epoch in range(300):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(
            logits.view(-1, vocab_size),
            Y.view(-1)
        )
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model, word2idx, idx2word

def predict(sentences, model, word2idx, idx2word):
    outputs = generate(sentences, model, word2idx, idx2word)
    for o in outputs:
        print(o)


def re_train(new_sentences, model, word2idx_old, idx2word_old):
    word2idx, idx2word = extend_vocab(
        word2idx_old, idx2word_old, new_sentences
    ) 

    old_vocab_size, emb_dim = model.embed.weight.shape
    new_vocab_size = len(word2idx)
    #新建embedding（保留旧权重）
    new_embed = nn.Embedding(new_vocab_size, emb_dim)
    with torch.no_grad():
        new_embed.weight[:old_vocab_size] = model.embed.weight
        # 新词随机初始化（或你指定）
        new_embed.weight[old_vocab_size:] = torch.randn(
            new_vocab_size - old_vocab_size, emb_dim
        )
    #新建输出层（保留旧权重）
    hidden_dim = model.fc.in_features
    new_fc = nn.Linear(hidden_dim, new_vocab_size)
    with torch.no_grad():
        new_fc.weight[:old_vocab_size] = model.fc.weight
        new_fc.bias[:old_vocab_size] = model.fc.bias

    model.embed = new_embed
    model.fc = new_fc

    X_new, Y_new = build_batch(new_sentences, word2idx)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])

    for epoch in range(200):
        optimizer.zero_grad()
        logits = model(X_new)
        loss = loss_fn(
            logits.view(-1, new_vocab_size),
            Y_new.view(-1)
        )
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Fine-tune Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    
    return word2idx, idx2word

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "retrain", "predict"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        sentences = [
            "我 喜欢 足球",
            "我 喜欢 篮球",
            "我 不喜欢 下雨"
        ]
        model, word2idx, idx2word = train(sentences)
        save_model(model, word2idx, idx2word, "dist/chat_model.pt")

    #持续训练 = 新数据 + 旧知识保护
    elif args.mode == "retrain":
        new_sentences = [
            "我 喜欢 音乐",
            "我 讨厌 噪音"
        ]
        model, word2idx_old, idx2word_old, = load_model("dist/chat_model.pt")
        word2idx, idx2word = re_train(new_sentences, model, word2idx_old, idx2word_old)
        save_model(model, word2idx, idx2word, "dist/chat_model.pt")

 
    elif args.mode == "predict":
        test_sentences = [
            "我 喜欢",
            "我 不喜欢"
        ]
        model_v, word2idx_v, idx2word_v = load_model("dist/chat_model.pt")
        predict(test_sentences, model_v, word2idx_v, idx2word_v)
