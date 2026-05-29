import torch

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

    # 构造输入输出序列（自回归）
    X, Y = build_batch(sentences, word2idx)
    return X, Y, word2idx


def extend_vocab(old_word2idx, new_sentences):
    word2idx = old_word2idx.copy()

    for s in new_sentences:
        for w in s.split():
            if w not in word2idx:
                idx = len(word2idx)
                word2idx[w] = idx

    return word2idx


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