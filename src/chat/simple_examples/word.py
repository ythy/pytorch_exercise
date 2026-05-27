import torch
import torch.nn as nn

DEFAULT_WORD_IDX = {
    # 我 喜欢 足球  <PAD>
    # 我 喜欢 <PAD> <PAD>
    "<PAD>": 0,   # 填充符 把不同长度的句子补成相同长度. batch 训练、长度对齐
    # <BOS> 我 喜欢 足球
    "<BOS>": 1,   # 句子开始. 生成起点、序列边界
    # 我 喜欢 足球 <EOS>
    "<EOS>": 2    # 句子结束. 生成终点、防止无限生成
}

    
def initial_data(sentences):

    word2idx = DEFAULT_WORD_IDX.copy()

    for s in sentences:
        for w in s.split():
            if w not in word2idx:
                word2idx[w] = len(word2idx)

    idx2word = {
        v: k 
        for k, v in word2idx.items()
    }
   
    # 构造输入输出序列（自回归）
    X, Y = build_batch(sentences, word2idx)
    return X, Y, word2idx, idx2word


def extend_vocab(old_word2idx, old_idx2word, new_sentences):
    word2idx = old_word2idx.copy()
    idx2word = old_idx2word.copy()

    for s in new_sentences:
        for w in s.split():
            if w not in word2idx:
                idx = len(word2idx)
                word2idx[w] = idx
                idx2word[idx] = w

    return word2idx, idx2word


def build_batch(sentences, word2idx, max_len=6):
    batch_input, batch_target = [], []

    for s in sentences:
        words = s.split()

        # 输入：<BOS> + 句子
        inp = [word2idx["<BOS>"]] + [word2idx[w] for w in words]
        # 目标：句子 + <EOS>
        tgt = [word2idx[w] for w in words] + [word2idx["<EOS>"]]

        # padding
        while len(inp) < max_len:
            inp.append(word2idx["<PAD>"])
            tgt.append(word2idx["<PAD>"])

        batch_input.append(inp)
        batch_target.append(tgt)

    return (
        torch.tensor(batch_input),  # [B, T]
        torch.tensor(batch_target)  # [B, T]
    )


def generate(sentences, model, word2idx, idx2word, max_len=6):
    results = []
    for s in sentences:
        words = s.split()
        ids = [word2idx["<BOS>"]] + [word2idx[w] for w in words]
        input_ids = torch.tensor([ids])

        for _ in range(max_len - len(ids)):
            logits = model(input_ids)
            next_id = logits[0, -1].argmax().item()
            if next_id == word2idx["<EOS>"]:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]])],
                dim=1
            )

        ids = input_ids[0].tolist()
        text = " ".join(idx2word[i] for i in ids)
        results.append(text)

    return results