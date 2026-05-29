from models.io import (
    save_model,
    load_model,
)
from runner.train import(
    train,
    re_train
)
from runner.predict import predict
import argparse
from config import *

SENTENCES = [
        "我 喜欢 足球",
        "我 喜欢 篮球",
        "我 不喜欢 下雨"
    ]

def run_train():
    model, word2idx = train(SENTENCES)
    save_model(model, word2idx, SAVE_PATH)

def run_predict():
    # test_sentences = [
    #     "我 喜欢",
    #     "我 不喜欢"
    # ]
    test_sentences = [
        "我"
    ]
    model_v, word2idx_v = load_model(SAVE_PATH)
    predict(test_sentences, model_v, word2idx_v)

def run_retrain():
    new_sentences = SENTENCES + [
        "我 喜欢 音乐",
        "我 不喜欢 噪音"
    ]
    model, word2idx_old = load_model(SAVE_PATH)
    word2idx = re_train(new_sentences, model, word2idx_old)
    save_model(model, word2idx, SAVE_PATH)

if __name__ == "__main__":
    # 在这里指定要运行的方法
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "retrain", "predict"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
       run_train()
       run_predict()
        
    #持续训练 = 新数据 + 旧知识保护
    elif args.mode == "retrain":
       run_retrain() 
       run_predict()
        
    elif args.mode == "predict":
       run_predict()
