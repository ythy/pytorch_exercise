# train_tokenizer.py
import sentencepiece as spm

def spm_train(input, model_prefix, vocab_size = 2100):
    spm.SentencePieceTrainer.Train(
        input=input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,  # 中文必须 ≥ 0.999
        split_by_whitespace=False,  # 中文不要按空格切
        user_defined_symbols=[
            "<s>",
            "</s>",
        ], # 强制 SentencePiece 把这些“特殊符号”当作不可分割的整体 token
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
        input_sentence_size=50000, # 最多只使用前 5 万行句子来训练 tokenizer
        max_sentence_length=256, # Found too long line (293 > 256)， Too long lines are skipped in the training.
    )
 