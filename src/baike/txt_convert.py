import json
import re

def clean(text):
    return re.sub(r"\s+", "", text).strip()

def txt_to_jsonl(txt_path, jsonl_path):
    with open(txt_path, "r", encoding="utf-8") as fin, \
         open(jsonl_path, "w", encoding="utf-8") as fout:

        lines = [l.strip() for l in fin.readlines()]
        i = 0
        while i < len(lines):
            # 跳过空行
            if not lines[i]:
                i += 1
                continue

            q = clean(lines[i])
            i += 1

            # 找答案（跳过空行）
            while i < len(lines) and not lines[i]:
                i += 1

            if i >= len(lines):
                break

            a = clean(lines[i])
            i += 1

            if q and a:
                fout.write(
                    json.dumps({"q": q, "a": a}, ensure_ascii=False) + "\n"
                )


def qa_to_corpus(txt_path, out_path):
    with open(txt_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        lines = [l.rstrip("\n") for l in fin]
        i = 0
        while i < len(lines):
            # 跳过空行
            if not lines[i]:
                i += 1
                continue

            q = clean(lines[i])
            i += 1

            # 找答案（跳过空行）
            while i < len(lines) and not lines[i]:
                i += 1
            if i >= len(lines):
                break

            a = clean(lines[i])
            i += 1

            if q and a:
                fout.write(f"{q}\t{a}\n")



if __name__ == "__main__":
    qa_to_corpus("data/chinese_baike_origin.txt", "data/chinese_baike_corpus.txt")