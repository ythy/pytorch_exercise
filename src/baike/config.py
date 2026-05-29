# config.py

DATA_ROOT = "data"
DIST_ROOT = "dist"

TOKENIZER_PATH = f"{DATA_ROOT}/tokenizer/chinese_baike.model"
TRAIN_FILE = f"{DATA_ROOT}/chinese_baike_qa.txt"
MODEL_SAVE_PATH = f"{DIST_ROOT}/baike_model_4400.pth"

VOCAB_SIZE = 4400
MAX_LEN = 128
BATCH_SIZE = 32     
LR = 1e-5         
EPOCHS = 1 
DEVICE = "cpu"       

MIN_LOSS = 1.2     # 低于这个值就认为训“够好了”
PATIENCE_STEPS = 10  # 连续多少个 step 低于阈值才停