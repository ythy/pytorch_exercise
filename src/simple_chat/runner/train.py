import torch
import torch.nn as nn
import torch.optim as optim
from models.chat_model import (
    TinyChatModel,
    expand_model,
)
from data.word import(
    build_batch,
    initial_data,
    extend_vocab,
)
from config import *

def train(sentences):
    X, Y, word2idx = initial_data(sentences)
    model = TinyChatModel(len(word2idx))
    _train(model, word2idx, X, Y)
    return model, word2idx


def re_train(new_sentences, model, word2idx_old):
    word2idx = extend_vocab(
        word2idx_old, new_sentences
    ) 
    new_model = expand_model(model, word2idx)
    x, y = build_batch(new_sentences, word2idx)
    _train(new_model, word2idx, x, y)
    return word2idx

def _train(model, word2idx, x, y):
    vocab_size = len(word2idx)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    # ===============================
    # 训练模型
    # ===============================
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(
            logits.view(-1, vocab_size),
            y.view(-1)
        )
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
 