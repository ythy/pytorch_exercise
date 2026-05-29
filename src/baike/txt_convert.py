BOS = "<s>"
SEP = "</s>"


def convert(qa_txt, out_txt):
    with open(qa_txt, "r", encoding="utf-8-sig") as fin, \
         open(out_txt, "w", encoding="utf-8") as fout:

        lines = [l.strip() for l in fin if l.strip()]

        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break

            q = lines[i]
            a = lines[i + 1]

            if not q.endswith(("？", "?")):
                q += "？"

            fout.write(f"{BOS}问：{q}{SEP}答：{a}{SEP}\n")


if __name__ == "__main__":
    convert("corpus/chinese_baike_origin.txt", "data/chinese_baike_qa.txt")

 
    


    
