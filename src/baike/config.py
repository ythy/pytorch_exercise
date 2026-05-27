# config.py

DATA_ROOT = "data"
DIST_ROOT = "dist"

TOKENIZER_PATH = f"{DATA_ROOT}/tokenizer/chinese_baike.model"
TRAIN_FILE = f"{DATA_ROOT}/chinese_baike_corpus.txt"
MODEL_SAVE_PATH = f"{DIST_ROOT}/baike_model_4400.pth"

VOCAB_SIZE = 4400
MAX_LEN = 64
BATCH_SIZE = 16     
LR = 5e-5           
EPOCHS = 8        

 