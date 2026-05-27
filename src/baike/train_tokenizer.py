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
        user_defined_symbols="<s>,</s>",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
 