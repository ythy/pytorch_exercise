# utils.py
import torch
import os

def save_model(
    model,
    tokenizer,
    epoch,
    save_dir="dist",
    filename="chat_tokenizer_model.pt"
):
    """
    保存模型（与早停训练脚本严格配套）
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "epoch": epoch,
    }, save_path)

    print(f"✅ 模型已保存: {save_path} (Epoch {epoch}")
 

def load_model(model_path, tokenizer_path):
    from tokenizer import ChatTokenizer
    from model import TinyChatModel

    # 1. 加载 tokenizer
    tokenizer = ChatTokenizer(tokenizer_path)

    # 2. 初始化模型（结构必须与训练时一致）
    model = TinyChatModel(tokenizer.vocab_size)

    # 3. 加载权重
    ckpt = torch.load(model_path, map_location="cpu")

    # ✅ 兼容两种保存方式（更安全）
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"✅ 加载模型 | Epoch {ckpt.get('epoch')} ")
    else:
        model.load_state_dict(ckpt)

    # 4. 切换到推理模式
    model.eval()
    return model, tokenizer