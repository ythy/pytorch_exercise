import argparse
from torch.utils.data import DataLoader
from config import (
    DEVICE,
    BATCH_SIZE,
    CKPT_PATH,
)
from data.loader import prepare_raw_data
from models.hybrid_model import HybridPlayerModel
from models.checkpoint import load_model
from runner.train import (train, validate)
from runner.predict import predict
from data.inference import get_data_custom

def build_arg_parser():
    parser = argparse.ArgumentParser(description="FIFA Hybrid Model CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "validate"],
        required=True,
        help="Run mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    return parser.parse_args()


def run_train(resume: bool):
    data = prepare_raw_data(5000)
    model = HybridPlayerModel(
        cat_vocab_sizes = data["cat_vocab_sizes"],
        num_feat_dim = data["x_num"].shape[1],
    )
    start_epoch = 1
    if resume:
        model, ckpt = load_model(CKPT_PATH)
        print("✅ Resumed from checkpoint")
        start_epoch = ckpt["epoch"] + 1

    # -------- Train --------
    train(
        model=model,
        data=data,
        ckpt_path=CKPT_PATH,
        start_epoch = start_epoch,
    )
   

def run_invalid():
    model, ckpt = load_model(CKPT_PATH)
    model.to(DEVICE)
    val_data = prepare_raw_data(50, ckpt["stats"])
    validate(model, val_data, ckpt["stats"])


def run_predict():
    # -------- Load model --------
    model, ckpt = load_model(CKPT_PATH)
    model.to(DEVICE)
    data = get_data_custom(ckpt)
    # -------- Inference --------
    predict(model, data, ckpt["stats"])


def main():
    args = build_arg_parser()

    if args.mode == "train":
        run_train(resume=args.resume)
        run_invalid()
    if args.mode == "validate":
        run_invalid()
        
    elif args.mode == "predict":
        run_predict()


if __name__ == "__main__":
    main()