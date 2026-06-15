import argparse
import json
import os
import pickle
import random
from typing import Dict, List

import matplotlib

# 使用非交互式后端，避免命令行环境弹窗
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from models import BiLSTMWithAttention


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_config(model_info_path: str, vocab_size: int) -> Dict:
    """
    从 json 中读取模型配置。
    如果没有提供配置文件，就使用训练时的默认参数。
    """
    default_config = {
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "hidden_size": 128,
        "attention_dim": 128,
        "num_classes": 2,
        "dropout": 0.3,
        "padding_idx": 0,
    }

    if not model_info_path or not os.path.exists(model_info_path):
        return default_config

    with open(model_info_path, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    config = model_info.get("config", {})
    for key in default_config:
        if key in config:
            default_config[key] = config[key]
    return default_config


def decode_tokens(input_ids: List[int], idx2word: Dict[int, str], pad_idx: int, unk_idx: int) -> List[str]:
    """
    把索引序列还原成词列表。
    这里不显示 PAD 和 UNK，避免可视化结果受到干扰。
    """
    tokens = []
    for token_id in input_ids:
        token_id = int(token_id)
        if token_id in (pad_idx, unk_idx):
            continue
        tokens.append(idx2word.get(token_id, "<UNK>"))
    return tokens


def filter_attention(tokens: List[str], input_ids: List[int], attention_weights: List[float], pad_idx: int, unk_idx: int) -> List[float]:
    """
    按与 decode_tokens 相同的规则，过滤掉 PAD 和 UNK 对应的注意力权重。
    """
    filtered_weights = []
    token_cursor = 0

    for token_id, weight in zip(input_ids, attention_weights):
        token_id = int(token_id)
        if token_id in (pad_idx, unk_idx):
            continue
        if token_cursor < len(tokens):
            filtered_weights.append(float(weight))
            token_cursor += 1
    return filtered_weights


def predict_all_samples(
    model: BiLSTMWithAttention,
    input_ids_list: List[List[int]],
    labels: List[int],
    device: str,
) -> List[Dict]:
    """
    对测试集所有样本做预测，并保留 logits / 预测标签 / 注意力权重，
    方便后面挑选“预测正确”的样本进行可视化。
    """
    model.eval()
    results = []

    with torch.no_grad():
        for input_ids, label in zip(input_ids_list, labels):
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            logits, attention_weights = model(input_tensor)
            pred = int(torch.argmax(logits, dim=1).item())

            results.append(
                {
                    "input_ids": input_ids,
                    "label": int(label),
                    "pred": pred,
                    "attention_weights": attention_weights.squeeze(0).cpu().tolist(),
                }
            )
    return results


def select_samples(results: List[Dict], sample_count: int = 5, seed: int = 42) -> List[Dict]:
    """
    尽量挑选预测正确的正负样本各 2~3 条：
    - 优先从预测正确的正类中取 2~3 条
    - 优先从预测正确的负类中取 2~3 条
    - 如果某一类不够，再从其他预测正确样本里补齐
    """
    rng = random.Random(seed)

    correct_positive = [item for item in results if item["label"] == 1 and item["pred"] == 1]
    correct_negative = [item for item in results if item["label"] == 0 and item["pred"] == 0]
    other_correct = [item for item in results if item["label"] == item["pred"]]

    rng.shuffle(correct_positive)
    rng.shuffle(correct_negative)
    rng.shuffle(other_correct)

    selected = []

    pos_take = min(3, len(correct_positive))
    neg_take = min(2, len(correct_negative))

    if pos_take + neg_take < sample_count:
        remaining = sample_count - (pos_take + neg_take)
        extra_neg = min(1, len(correct_negative) - neg_take, remaining)
        neg_take += max(extra_neg, 0)

    selected.extend(correct_positive[:pos_take])
    selected.extend(correct_negative[:neg_take])

    if len(selected) < sample_count:
        used_ids = {id(item) for item in selected}
        for item in other_correct:
            if id(item) not in used_ids:
                selected.append(item)
                used_ids.add(id(item))
            if len(selected) == sample_count:
                break

    if len(selected) < sample_count:
        remaining_all = results[:]
        rng.shuffle(remaining_all)
        used_ids = {id(item) for item in selected}
        for item in remaining_all:
            if id(item) not in used_ids:
                selected.append(item)
                used_ids.add(id(item))
            if len(selected) == sample_count:
                break

    return selected[:sample_count]


def print_attention_details(samples: List[Dict], idx2word: Dict[int, str], pad_idx: int, unk_idx: int) -> None:
    """
    在控制台输出每条样本的词和对应注意力权重，便于人工检查。
    """
    for sample_idx, sample in enumerate(samples, start=1):
        tokens = decode_tokens(sample["input_ids"], idx2word, pad_idx, unk_idx)
        weights = filter_attention(tokens, sample["input_ids"], sample["attention_weights"], pad_idx, unk_idx)

        print("=" * 80)
        print(
            f"样本 {sample_idx} | 真实标签={sample['label']} | 预测标签={sample['pred']} | "
            f"是否预测正确={sample['label'] == sample['pred']}"
        )
        for token, weight in zip(tokens, weights):
            print(f"{token}\t{weight:.6f}")


def draw_attention_visualization(samples: List[Dict], idx2word: Dict[int, str], pad_idx: int, unk_idx: int, save_path: str) -> None:
    """
    把每条句子的词画成一行，并用背景颜色深浅表示注意力权重大小。
    权重越高，背景颜色越深。
    """
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 1, figsize=(18, max(2.5 * num_samples, 8)))
    if num_samples == 1:
        axes = [axes]

    cmap = plt.cm.OrRd

    for ax, sample_idx, sample in zip(axes, range(1, num_samples + 1), samples):
        tokens = decode_tokens(sample["input_ids"], idx2word, pad_idx, unk_idx)
        weights = filter_attention(tokens, sample["input_ids"], sample["attention_weights"], pad_idx, unk_idx)

        if not tokens:
            tokens = ["<EMPTY>"]
            weights = [0.0]

        max_weight = max(weights) if max(weights) > 0 else 1.0
        normalized_weights = [weight / max_weight for weight in weights]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        title = f"样本 {sample_idx} | 真实标签={sample['label']} | 预测标签={sample['pred']}"
        ax.set_title(title, fontsize=12, loc="left")

        x = 0.01
        y = 0.5
        row_step = 0.14

        for token, norm_weight, raw_weight in zip(tokens, normalized_weights, weights):
            color = cmap(0.15 + 0.85 * norm_weight)
            text = f"{token} ({raw_weight:.3f})"

            text_artist = ax.text(
                x,
                y,
                text,
                fontsize=10,
                va="center",
                ha="left",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none"),
            )

            fig.canvas.draw()
            bbox = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
            ax_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
            width_ratio = bbox.width / ax_bbox.width

            x += width_ratio + 0.01
            if x > 0.95:
                x = 0.01
                y -= row_step

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="可视化 BiLSTMWithAttention 的注意力权重")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\bilstm_attention_best.pth",
        help="训练好的 BiLSTMWithAttention 权重路径",
    )
    parser.add_argument(
        "--model_info_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\bilstm_attention_info.json",
        help="模型配置信息 json 路径，可选；如果不存在则使用默认参数",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\data\test.pkl",
        help="测试集 pkl 路径",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\data\vocab.pkl",
        help="词表 pkl 路径",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\attention_visualization.png",
        help="注意力可视化图片保存路径",
    )
    parser.add_argument("--sample_count", type=int, default=5, help="可视化样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型权重文件：{args.model_path}")

    vocab_data = load_pickle(args.vocab_path)
    test_data = load_pickle(args.test_path)

    word2idx = vocab_data["word2idx"]
    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
    pad_idx = int(vocab_data.get("pad_idx", word2idx.get("<PAD>", 0)))
    unk_idx = int(vocab_data.get("unk_idx", word2idx.get("<UNK>", 1)))

    config = load_model_config(args.model_info_path, vocab_size=len(word2idx))
    config["padding_idx"] = pad_idx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMWithAttention(**config).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    results = predict_all_samples(
        model=model,
        input_ids_list=test_data["input_ids"],
        labels=test_data["labels"],
        device=device,
    )
    selected_samples = select_samples(results, sample_count=args.sample_count, seed=args.seed)

    print_attention_details(selected_samples, idx2word, pad_idx, unk_idx)
    draw_attention_visualization(selected_samples, idx2word, pad_idx, unk_idx, args.save_path)

    print(f"\n注意力可视化图片已保存到：{args.save_path}")


if __name__ == "__main__":
    main()
