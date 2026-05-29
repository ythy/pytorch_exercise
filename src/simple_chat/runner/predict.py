import torch

def predict(sentences, model, word2idx):
    outputs = generate(sentences, model, word2idx)
    for o in outputs:
        print(o)


def generate(sentences, model, word2idx, top_k = 5, max_len=6):
    results = []
    for s in sentences:
        words = s.split()
        ids = [word2idx["<BOS>"]] + [word2idx[w] for w in words]
        input_ids = torch.tensor([ids])
        logits = None
        for _ in range(max_len - len(ids)):
            logits, _ = model(input_ids)
            next_id = logits[0, -1].argmax().item() # 	argmax 贪心解码 并且  argmax = Top‑K(k=1)
            if next_id == word2idx["<EOS>"]:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]])],
                dim=1
            )
           # all_logits.append(logits[:, -1, :])

        ids = input_ids[0].tolist()
        idx2word =  {
            v: k 
            for k, v in word2idx.items()
        }
        text = " ".join(idx2word[i] for i in ids)
        results.append(text)

        if isinstance(top_k, int) and logits is not None:
            print(f"Top-K 预测：{s}")
            with torch.no_grad():
                for i in range(logits.size(1)):
                    last_logits = logits[0, i]
                    probs = torch.softmax(last_logits, dim=-1)
                    topk = torch.topk(probs, k=top_k) # argmax = Top‑K(k=1)
                    print(f"Step {i}:")
                    for p, idx in zip(topk.values, topk.indices):
                        print(f"  {idx2word[idx.item()]:10s} -> {p.item():.4f}") # :10s 补空格

 

    return results

 


# Top-K 预测：我
# Step 0:
#   我          -> 1.0000
#   音乐         -> 0.0000
#   篮球         -> 0.0000
#   足球         -> 0.0000
#   喜欢         -> 0.0000
# Step 1:
#   喜欢         -> 0.6000
#   不喜欢        -> 0.4000
#   噪音         -> 0.0000
#   音乐         -> 0.0000
#   篮球         -> 0.0000
# Step 2:
#   音乐         -> 0.3333
#   足球         -> 0.3333
#   篮球         -> 0.3333
#   噪音         -> 0.0000
#   喜欢         -> 0.0000
# Step 3:
#   <EOS>      -> 1.0000
#   篮球         -> 0.0000
#   音乐         -> 0.0000
#   足球         -> 0.0000
#   我          -> 0.0000
# <BOS> 我 喜欢 音乐