# tokenizer.py
import sentencepiece as spm

class ChatTokenizer:
    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    def encode(self, text, add_bos=True):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def decode(self, ids):
        # ✅ 支持 list / tensor
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.sp.decode(ids)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
    

def spm_train():
    spm.SentencePieceTrainer.Train(
        input="data/chat_corpus.txt",
        model_prefix="data/tokenizer/chat",
        vocab_size=900,
        character_coverage=0.99995,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        byte_fallback=True,
    )