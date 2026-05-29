import torch
from models.chat_model import TinyChatModel 

def save_model(model, word2idx, path):
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": len(word2idx),
        "word2idx": word2idx,
        "emb_dim": model.embed.embedding_dim,
        "hidden_dim": model.fc.in_features
    }, path)

def load_model(path):
    ckpt = torch.load(path)
    model = TinyChatModel(
        vocab_size=ckpt["vocab_size"],
        emb_dim=ckpt["emb_dim"],
        hidden_dim=ckpt["hidden_dim"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    word2idx = ckpt["word2idx"]
    return model, word2idx

