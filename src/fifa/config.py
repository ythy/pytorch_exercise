# config.py

DATA_PATH = "corpus/FIFA19.csv"
CKPT_PATH = "dist/fifa/fifa_best.pt"

FEATURES_NUM = [
    "Overall", "Potential", "Age", "height_cm", "weight_kg",
]

FEATURES_CAT = ["Nationality", "Position", "Club"]


TRAIN_SIZE = 1500
EPOCHS = 400
SEED = 42
DEVICE = "cpu"
BATCH_SIZE = 100
LR = 1e-3