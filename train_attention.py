import json
import os
import pickle
import torch

from models import BiLSTMWithAttention
from train import create_data_loaders, evaluate_model, set_seed, train_model

base_dir = r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1"
data_dir = os.path.join(base_dir, "data")

train_path = os.path.join(data_dir, "train.pkl")
val_path = os.path.join(data_dir, "val.pkl")
test_path = os.path.join(data_dir, "test.pkl")
vocab_path = os.path.join(data_dir, "vocab.pkl")

save_model_path = os.path.join(base_dir, "bilstm_attention_best.pth")
save_info_path = os.path.join(base_dir, "bilstm_attention_info.json")

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(vocab_path, "rb") as f:
    vocab_data = pickle.load(f)

vocab_size = len(vocab_data["word2idx"])
pad_idx = vocab_data.get("pad_idx", 0)

train_loader, val_loader, test_loader = create_data_loaders(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=64,
    seed=42,
)

model = BiLSTMWithAttention(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_size=128,
    attention_dim=128,
    num_classes=2,
    dropout=0.3,
    padding_idx=pad_idx,
)

trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    lr=0.001,
    device=device,
)

test_metrics = evaluate_model(
    model=trained_model,
    test_loader=test_loader,
    device=device,
)

print("测试集结果：", test_metrics)

torch.save(trained_model.state_dict(), save_model_path)

model_info = {
    "model_name": "BiLSTMWithAttention",
    "config": {
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "hidden_size": 128,
        "attention_dim": 128,
        "num_classes": 2,
        "dropout": 0.3,
        "padding_idx": pad_idx,
    },
}
with open(save_info_path, "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print("模型已保存到：", save_model_path)
print("模型配置已保存到：", save_info_path)