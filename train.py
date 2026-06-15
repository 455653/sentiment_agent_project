import copy
import json
import os
import pickle
import random
import time
from typing import Dict, List, Tuple

import matplotlib

# 使用非交互式后端，避免在服务器或命令行环境下画图时报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preprocessing import SentimentDataset
from models import BiLSTMModel, BiLSTMWithAttention, GRUModel, LSTMModel, SimpleRNN, TextCNN


def set_seed(seed: int) -> None:
    """
    设置随机种子，尽量保证实验结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    根据真实标签和预测标签，计算四个常用分类指标。
    二分类任务这里使用 binary 平均方式。
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def extract_logits(model_output):
    """
    兼容两种 forward 返回形式：
    - 普通分类模型：直接返回 logits
    - 注意力模型：返回 (logits, attention_weights)

    训练和评估时只需要 logits 参与损失计算与预测，
    因此这里统一抽取第一个返回值。
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def run_validation(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> Tuple[float, Dict[str, float]]:
    """
    在验证集或测试集上跑一遍前向推理，返回平均损失和分类指标。
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = extract_logits(model(input_ids))
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    统一训练函数：
    - 使用 Adam 优化器
    - 使用 CrossEntropyLoss 损失函数
    - 每个 epoch 结束后在验证集上计算 accuracy / precision / recall / f1
    - 记录训练历史，便于后续画曲线和做结果分析

    这里会保留“验证集 F1 最好”的模型权重，并在训练结束后自动恢复到最佳状态。
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "epoch_time": [],
    }

    best_val_f1 = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = extract_logits(model(input_ids))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        val_loss, val_metrics = run_validation(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["epoch_time"].append(float(epoch_time))

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        # 用验证集 F1 作为“最佳模型”选择标准，避免直接用测试集挑模型
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    在测试集上评估模型，返回 accuracy / precision / recall / f1。
    """
    criterion = nn.CrossEntropyLoss()
    _, metrics = run_validation(model, test_loader, criterion, device)
    return metrics


def format_mean_std(values: List[float]) -> str:
    """
    把多个实验结果格式化为“均值 ± 标准差”，方便最终表格展示。
    """
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))
    return f"{mean_value:.4f} ± {std_value:.4f}"


def build_model(model_name: str, vocab_size: int, pretrained_embedding=None) -> Tuple[nn.Module, Dict]:
    """
    根据模型名称创建对应模型，同时返回模型配置，方便后续保存最佳模型信息。
    """
    common_config = {
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "num_classes": 2,
        "dropout": 0.3,
        "padding_idx": 0,
        "pretrained_embedding": pretrained_embedding,
    }

    if model_name == "SimpleRNN":
        config = {**common_config, "hidden_size": 128}
        model = SimpleRNN(**config)
    elif model_name == "LSTMModel":
        config = {**common_config, "hidden_size": 128}
        model = LSTMModel(**config)
    elif model_name == "GRUModel":
        config = {**common_config, "hidden_size": 128}
        model = GRUModel(**config)
    elif model_name == "BiLSTMModel":
        config = {**common_config, "hidden_size": 128}
        model = BiLSTMModel(**config)
    elif model_name == "BiLSTMWithAttention":
        config = {**common_config, "hidden_size": 128, "attention_dim": 128}
        model = BiLSTMWithAttention(**config)
    elif model_name == "TextCNN":
        config = {**common_config, "num_filters": 128, "kernel_sizes": (2, 3, 4)}
        model = TextCNN(**config)
    else:
        raise ValueError(f"不支持的模型名称：{model_name}")

    # 词向量矩阵体积较大，而且通常不适合直接写入 json，这里从配置展示中去掉
    config_for_save = {k: v for k, v in config.items() if k != "pretrained_embedding"}
    return model, config_for_save


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train / val / test 三个 DataLoader。
    训练集使用 shuffle，验证集和测试集不打乱。
    """
    train_dataset = SentimentDataset(train_path)
    val_dataset = SentimentDataset(val_path)
    test_dataset = SentimentDataset(test_path)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def plot_training_curves(
    model_histories: Dict[str, List[Dict[str, List[float]]]],
    save_path: str,
) -> None:
    """
    把 5 个模型的训练曲线画在同一张图里。
    每个子图展示：
    - 平均 train_loss
    - 平均 val_accuracy

    由于每个模型训练了 3 次，这里对 3 次结果按 epoch 求平均后再绘图，
    这样图会更平滑，也更适合做模型间比较。
    """
    model_names = list(model_histories.keys())
    num_models = len(model_names)
    cols = 2
    rows = int(np.ceil(num_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, model_name in enumerate(model_names):
        histories = model_histories[model_name]
        train_loss_array = np.array([h["train_loss"] for h in histories], dtype=float)
        val_acc_array = np.array([h["val_accuracy"] for h in histories], dtype=float)

        mean_train_loss = train_loss_array.mean(axis=0)
        mean_val_acc = val_acc_array.mean(axis=0)
        epochs = np.arange(1, len(mean_train_loss) + 1)

        ax = axes[idx]
        ax.plot(epochs, mean_train_loss, marker="o", label="train_loss")
        ax.plot(epochs, mean_val_acc, marker="s", label="val_accuracy")
        ax.set_title(model_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    # 如果子图数量多于模型数量，把多余的空白子图隐藏
    for idx in range(num_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    主程序流程：
    1. 固定随机种子；
    2. 加载词表与数据；
    3. 依次训练 5 个模型，每个模型训练 3 次；
    4. 汇总结果并保存 csv；
    5. 保存训练曲线；
    6. 保存表现最好的模型权重和配置。
    """
    set_seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    train_path = os.path.join(data_dir, "train.pkl")
    val_path = os.path.join(data_dir, "val.pkl")
    test_path = os.path.join(data_dir, "test.pkl")
    vocab_path = os.path.join(data_dir, "vocab.pkl")

    results_csv_path = os.path.join(base_dir, "results_comparison.csv")
    curves_path = os.path.join(base_dir, "training_curves.png")
    best_model_path = os.path.join(base_dir, "best_model.pth")
    best_model_info_path = os.path.join(base_dir, "best_model_info.json")

    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    vocab_size = len(vocab_data["word2idx"])
    print(f"加载词表成功，vocab_size = {vocab_size}")

    batch_size = 64
    num_epochs = 10
    lr = 0.001
    seeds = [42, 43, 44]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用设备：{device}")

    model_names = ["SimpleRNN", "LSTMModel", "GRUModel", "BiLSTMModel", "TextCNN"]

    results_rows = []
    model_histories = {}

    best_overall = {
        "model_name": None,
        "state_dict": None,
        "config": None,
        "best_val_f1": -1.0,
        "seed": None,
    }

    for model_name in model_names:
        print("\n" + "=" * 80)
        print(f"开始训练模型：{model_name}")
        print("=" * 80)

        run_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "avg_epoch_time": [],
        }
        histories_for_this_model = []
        parameter_count = None

        for seed in seeds:
            print("\n" + "-" * 60)
            print(f"模型：{model_name} | 随机种子：{seed}")
            print("-" * 60)

            set_seed(seed)
            train_loader, val_loader, test_loader = create_data_loaders(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                batch_size=batch_size,
                seed=seed,
            )

            model, model_config = build_model(model_name=model_name, vocab_size=vocab_size)
            parameter_count = model.num_parameters if hasattr(model, "num_parameters") else model.count_parameters()

            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device,
            )

            test_metrics = evaluate_model(
                model=trained_model,
                test_loader=test_loader,
                device=device,
            )

            avg_epoch_time = float(np.mean(history["epoch_time"]))
            histories_for_this_model.append(history)

            run_metrics["accuracy"].append(test_metrics["accuracy"])
            run_metrics["precision"].append(test_metrics["precision"])
            run_metrics["recall"].append(test_metrics["recall"])
            run_metrics["f1"].append(test_metrics["f1"])
            run_metrics["avg_epoch_time"].append(avg_epoch_time)

            print(
                f"测试结果 | accuracy={test_metrics['accuracy']:.4f} | "
                f"precision={test_metrics['precision']:.4f} | "
                f"recall={test_metrics['recall']:.4f} | "
                f"f1={test_metrics['f1']:.4f}"
            )

            current_best_val_f1 = max(history["val_f1"])
            if current_best_val_f1 > best_overall["best_val_f1"]:
                best_overall["model_name"] = model_name
                best_overall["state_dict"] = copy.deepcopy(trained_model.state_dict())
                best_overall["config"] = model_config
                best_overall["best_val_f1"] = current_best_val_f1
                best_overall["seed"] = seed

        model_histories[model_name] = histories_for_this_model

        results_rows.append(
            {
                "model_name": model_name,
                "parameter_count": parameter_count,
                "avg_epoch_time_sec": format_mean_std(run_metrics["avg_epoch_time"]),
                "test_accuracy": format_mean_std(run_metrics["accuracy"]),
                "test_precision": format_mean_std(run_metrics["precision"]),
                "test_recall": format_mean_std(run_metrics["recall"]),
                "test_f1": format_mean_std(run_metrics["f1"]),
                "accuracy_mean": float(np.mean(run_metrics["accuracy"])),
                "accuracy_std": float(np.std(run_metrics["accuracy"], ddof=0)),
                "f1_mean": float(np.mean(run_metrics["f1"])),
                "f1_std": float(np.std(run_metrics["f1"], ddof=0)),
            }
        )

    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 80)
    print("所有模型结果汇总：")
    print("=" * 80)
    print(results_df)

    results_df.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到：{results_csv_path}")

    plot_training_curves(model_histories, curves_path)
    print(f"训练曲线已保存到：{curves_path}")

    torch.save(best_overall["state_dict"], best_model_path)
    best_model_info = {
        "model_name": best_overall["model_name"],
        "seed": best_overall["seed"],
        "best_val_f1": best_overall["best_val_f1"],
        "config": best_overall["config"],
    }
    with open(best_model_info_path, "w", encoding="utf-8") as f:
        json.dump(best_model_info, f, ensure_ascii=False, indent=2)

    print(f"最佳模型权重已保存到：{best_model_path}")
    print(f"最佳模型信息已保存到：{best_model_info_path}")


if __name__ == "__main__":
    main()
