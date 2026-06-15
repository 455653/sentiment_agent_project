import argparse
import os
import pickle
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

try:
    import jieba
except ImportError as e:
    jieba = None
    _jieba_import_error = e

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    torch = None
    Dataset = object
    _torch_import_error = e


def read_online_shopping_csv(csv_path: str) -> pd.DataFrame:
    """
    读取 online_shopping_10_cats.csv，并统一成三列：cat / label / review。
    该数据集在不同来源中可能存在“有表头/无表头”的差异，这里做兼容处理。
    """
    # 1) 优先用 utf-8-sig 读取（可自动去掉 BOM），失败再尝试 utf-8
    try:
        df = pd.read_csv(csv_path, header=None, names=["cat", "label", "review"], encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, header=None, names=["cat", "label", "review"], encoding="utf-8")

    # 2) 如果第一行实际上是表头（cat,label,review），则把它去掉
    first_row = df.iloc[0].astype(str).tolist()
    if [x.strip().lower() for x in first_row] == ["cat", "label", "review"]:
        df = df.iloc[1:].reset_index(drop=True)

    # 3) label 转成整数，review 转成字符串（避免后续分词出现 NaN/float）
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df["review"] = df["review"].astype(str)
    df["cat"] = df["cat"].astype(str).str.lstrip("\ufeff")

    return df


def clean_data(df: pd.DataFrame, min_len: int = 5) -> pd.DataFrame:
    """
    数据清洗：
    - 去除空值（尤其是 label 缺失或 review 为空）
    - 去除重复的 review（按 review 文本去重）
    - 去除 review 长度小于 min_len 的样本
    """
    # 1) 去除 label 缺失、review 缺失（虽然上面把 review 转成了 str，但可能是 "nan" 这类脏数据）
    df = df.dropna(subset=["label", "review"]).copy()

    # 2) 去除明显的空字符串/纯空白
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""].copy()

    # 3) 去除重复评论（只看 review 文本，避免同一评论重复出现）
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

    # 4) 去除过短文本（这里按“字符长度”过滤，符合你的要求）
    df = df[df["review"].str.len() >= min_len].reset_index(drop=True)

    # 5) label 转为 int（此时不应再有缺失）
    df["label"] = df["label"].astype(int)

    return df


def tokenize_reviews(reviews: List[str]) -> List[List[str]]:
    """
    使用 jieba 对每条评论做分词，返回 tokens 列表。
    """
    if jieba is None:
        raise ImportError(
            f"未安装 jieba（当前解释器：{sys.executable}）。请用同一个解释器安装："
            f"{sys.executable} -m pip install jieba"
        ) from _jieba_import_error

    tokens_list: List[List[str]] = []
    for text in reviews:
        # jieba.lcut 返回 list[str]，这里顺便把空白 token 去掉
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        tokens_list.append(tokens)
    return tokens_list


def build_vocab(tokens_list: List[List[str]], max_vocab_size: int = 20000) -> Dict[str, int]:
    """
    构建词表：
    - 统计词频
    - 取出现频率最高的前 max_vocab_size 个词
    - 索引 0: PAD，索引 1: UNK
    """
    counter = Counter()
    for tokens in tokens_list:
        counter.update(tokens)

    most_common = counter.most_common(max_vocab_size)

    word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        if word not in word2idx:
            word2idx[word] = i

    return word2idx


def tokens_to_padded_ids(
    tokens_list: List[List[str]],
    word2idx: Dict[str, int],
    max_len: int = 100,
) -> List[List[int]]:
    """
    把分词后的文本转换为索引序列，并进行 padding/truncation：
    - 超过 max_len：截断
    - 不足 max_len：右侧补 0（PAD）
    """
    unk_idx = word2idx.get("<UNK>", 1)
    pad_idx = word2idx.get("<PAD>", 0)

    all_ids: List[List[int]] = []
    for tokens in tokens_list:
        ids = [word2idx.get(t, unk_idx) for t in tokens]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids = ids + [pad_idx] * (max_len - len(ids))
        all_ids.append(ids)

    return all_ids


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def stratified_split(
    input_ids: List[List[int]],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[int], List[int], List[int]]:
    """
    不依赖 sklearn 的分层划分（按 label 近似保持比例一致）：
    - 先按 label 把样本索引分组
    - 组内打乱
    - 按比例切分到 train/val/test
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    rng = random.Random(random_state)

    label_to_indices: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        label_to_indices.setdefault(int(y), []).append(i)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for _, indices in label_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    X_train = [input_ids[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_val = [input_ids[i] for i in val_indices]
    y_val = [labels[i] for i in val_indices]
    X_test = [input_ids[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test



class SentimentDataset(Dataset):
    """
    一个简单的 PyTorch Dataset：
    - 从 pkl 文件加载数据
    - 返回 (input_ids, label)
    """

    def __init__(self, pkl_path: str):
        if torch is None:
            raise ImportError(
                f"未安装 torch（当前解释器：{sys.executable}）。无法使用 SentimentDataset，请用同一个解释器安装："
                f"{sys.executable} -m pip install torch"
            ) from _torch_import_error

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.input_ids: List[List[int]] = data["input_ids"]
        self.labels: List[int] = data["labels"]

        if len(self.input_ids) != len(self.labels):
            raise ValueError(f"input_ids 与 labels 长度不一致：{len(self.input_ids)} vs {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label


def main():
    # Windows 终端默认编码可能是 gbk，直接 print 中文/特殊字符时可能报 UnicodeEncodeError
    # 这里把 stdout/stderr 统一切到 utf-8，减少踩坑（失败也不影响后续逻辑）
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="online_shopping_10_cats 数据预处理脚本")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\data\online_shopping_10_cats.csv",
        help="online_shopping_10_cats.csv 文件路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"D:\AllProjectCode\pythoncode\sentiment_agent_project\code1\data",
        help="输出目录（保存 vocab.pkl / train.pkl / val.pkl / test.pkl）",
    )
    parser.add_argument("--max_vocab_size", type=int, default=20000, help="词表大小（取词频最高的前 N 个词）")
    parser.add_argument("--max_len", type=int, default=100, help="序列最大长度 MAX_LEN")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="仅用于调试：限制读取前 N 行（不传则使用全量数据）",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    out_dir = args.out_dir
    max_vocab_size = args.max_vocab_size
    max_len = args.max_len
    random_state = args.random_state

    # ========== 1. 读取数据并打印基本信息 ==========
    df = read_online_shopping_csv(csv_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    print("===== 原始数据前 5 行 =====")
    print(df.head(5))
    print("\n===== 原始数据总条数 =====")
    print(len(df))
    print("\n===== label 数量分布 =====")
    print(df["label"].value_counts(dropna=False))
    print("\n===== cat 数量分布 =====")
    print(df["cat"].value_counts(dropna=False).head(50))

    # ========== 2. 数据清洗 ==========
    df = clean_data(df, min_len=5)
    print("\n===== 清洗后剩余样本数 =====")
    print(len(df))

    # ========== 3. jieba 分词，并保存为新列 tokens ==========
    df["tokens"] = tokenize_reviews(df["review"].tolist())

    # ========== 4. 构建词表并保存 ==========
    word2idx = build_vocab(df["tokens"].tolist(), max_vocab_size=max_vocab_size)
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_path = os.path.join(out_dir, "vocab.pkl")
    save_pickle(
        {
            "word2idx": word2idx,
            "idx2word": idx2word,
            "pad_idx": word2idx["<PAD>"],
            "unk_idx": word2idx["<UNK>"],
            "max_vocab_size": max_vocab_size,
        },
        vocab_path,
    )
    print("\n===== 词表构建完成 =====")
    print(f"词表大小（含 PAD/UNK）：{len(word2idx)}")
    print(f"已保存：{vocab_path}")

    # ========== 5. tokens -> 索引序列（MAX_LEN=100，右侧 padding） ==========
    input_ids = tokens_to_padded_ids(df["tokens"].tolist(), word2idx, max_len=max_len)
    labels = df["label"].astype(int).tolist()

    # ========== 6. 划分数据集（7:2:1，且按 label 分层）并保存 ==========
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        input_ids=input_ids,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_state=random_state,
    )

    train_path = os.path.join(out_dir, "train.pkl")
    val_path = os.path.join(out_dir, "val.pkl")
    test_path = os.path.join(out_dir, "test.pkl")

    save_pickle({"input_ids": X_train, "labels": y_train}, train_path)
    save_pickle({"input_ids": X_val, "labels": y_val}, val_path)
    save_pickle({"input_ids": X_test, "labels": y_test}, test_path)

    print("\n===== 数据集划分完成 =====")
    print(f"train: {len(y_train)} | val: {len(y_val)} | test: {len(y_test)}")
    print(f"已保存：{train_path}")
    print(f"已保存：{val_path}")
    print(f"已保存：{test_path}")

    # 额外打印一下三份数据的 label 分布，便于你确认 stratify 是否生效
    def _dist(y: List[int]) -> Dict[int, int]:
        c = Counter(y)
        return dict(sorted(c.items(), key=lambda x: x[0]))

    print("\n===== label 分布（train/val/test） =====")
    print("train:", _dist(y_train))
    print("val  :", _dist(y_val))
    print("test :", _dist(y_test))


if __name__ == "__main__":
    main()
