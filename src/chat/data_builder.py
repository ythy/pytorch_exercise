import json

def jsonl_to_corpus(jsonl_path, corpus_path):
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        conv = []
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            conv.append(obj["content"])
            conv.append("<EOS>")

            if obj["role"] == "assistant":
                texts.append("<BOS> " + "".join(conv))
                conv = []

    with open(corpus_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
 