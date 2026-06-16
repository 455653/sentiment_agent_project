import json
import os
import pickle
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import BiLSTMModel, BiLSTMWithAttention, GRUModel, LSTMModel, SimpleRNN, TextCNN

try:
    import jieba
except ImportError as e:
    jieba = None
    _jieba_import_error = e


class SentimentPredictor:
    """
    一个可直接复用的情感预测器：
    - 初始化时自动加载词表、模型配置和最佳模型权重
    - 提供 predict(text) 接口，输入中文文本后返回标签和置信度
    """

    def __init__(
        self,
        vocab_path: str | None = None,
        model_info_path: str | None = None,
        model_path: str | None = None,
        max_len: int = 100,
        device: str | None = None,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vocab_path = vocab_path or os.path.join(base_dir, "data", "vocab.pkl")
        self.model_info_path = model_info_path or os.path.join(base_dir, "best_model_info.json")
        self.model_path = model_path or os.path.join(base_dir, "best_model.pth")
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._check_file_exists(self.vocab_path, "词表文件")
        self._check_file_exists(self.model_info_path, "模型配置文件")
        self._check_file_exists(self.model_path, "模型权重文件")

        with open(self.vocab_path, "rb") as f:
            vocab_data = pickle.load(f)

        self.word2idx: Dict[str, int] = vocab_data["word2idx"]
        self.pad_idx = int(vocab_data.get("pad_idx", self.word2idx.get("<PAD>", 0)))
        self.unk_idx = int(vocab_data.get("unk_idx", self.word2idx.get("<UNK>", 1)))

        with open(self.model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)

        self.model_name = model_info["model_name"]
        self.model_config = dict(model_info.get("config", {}))
        self.model_config["vocab_size"] = len(self.word2idx)
        self.model_config["padding_idx"] = self.pad_idx

        self.model = self._build_model(self.model_name, self.model_config)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _check_file_exists(file_path: str, file_desc: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到{file_desc}：{file_path}")

    @staticmethod
    def _extract_logits(model_output: torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    @staticmethod
    def _build_model(model_name: str, config: Dict) -> torch.nn.Module:
        model_map = {
            "SimpleRNN": SimpleRNN,
            "LSTMModel": LSTMModel,
            "GRUModel": GRUModel,
            "BiLSTMModel": BiLSTMModel,
            "BiLSTMWithAttention": BiLSTMWithAttention,
            "TextCNN": TextCNN,
        }
        if model_name not in model_map:
            raise ValueError(f"不支持的模型名称：{model_name}")
        return model_map[model_name](**config)

    def _preprocess(self, text: str) -> torch.Tensor:
        """
        用 jieba 分词后转成索引序列，并统一 padding/截断到 MAX_LEN。
        """
        if jieba is None:
            raise ImportError("当前环境未安装 jieba，请先执行：pip install jieba") from _jieba_import_error

        tokens: List[str] = [token.strip() for token in jieba.lcut(text) if token.strip()]
        input_ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        input_ids = input_ids[: self.max_len]

        if len(input_ids) < self.max_len:
            input_ids = input_ids + [self.pad_idx] * (self.max_len - len(input_ids))

        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict(self, text: str) -> dict:
        """
        输入一条中文文本，返回预测标签和对应置信度。
        """
        input_tensor = self._preprocess(text)

        with torch.no_grad():
            logits = self._extract_logits(self.model(input_tensor))
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = round(float(probs[0, pred_idx].item()), 4)

        return {
            "label": "正面" if pred_idx == 1 else "负面",
            "confidence": confidence,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 128) -> List[dict]:
        """
        批量预测多条文本。
        返回格式：[{"index": 0, "text": "...", "label": "正面", "confidence": 0.9234}, ...]
        - 内部用 DataLoader 分 batch 推理
        - 使用 torch.no_grad() + model.eval()
        - batch_size 默认为 128
        """
        if jieba is None:
            raise ImportError("当前环境未安装 jieba，请先执行：pip install jieba") from _jieba_import_error

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")

        if not texts:
            return []

        cleaned_texts = [str(text) for text in texts]

        try:
            input_ids_list: List[List[int]] = []
            for text in cleaned_texts:
                tokens: List[str] = [token.strip() for token in jieba.lcut(text) if token.strip()]
                input_ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]
                input_ids = input_ids[: self.max_len]
                if len(input_ids) < self.max_len:
                    input_ids = input_ids + [self.pad_idx] * (self.max_len - len(input_ids))
                input_ids_list.append(input_ids)

            all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            dataset = TensorDataset(all_input_ids)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            self.model.eval()
            results: List[dict] = []
            offset = 0

            with torch.no_grad():
                for (batch_input_ids,) in dataloader:
                    batch_input_ids = batch_input_ids.to(self.device)
                    logits = self._extract_logits(self.model(batch_input_ids))
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1)

                    batch_indices = pred_idx.tolist()
                    batch_confidences = probs[torch.arange(probs.size(0)), pred_idx].tolist()

                    for i, (pred, conf) in enumerate(zip(batch_indices, batch_confidences)):
                        global_index = offset + i
                        results.append(
                            {
                                "index": global_index,
                                "text": cleaned_texts[global_index],
                                "label": "正面" if int(pred) == 1 else "负面",
                                "confidence": round(float(conf), 4),
                            }
                        )

                    offset += len(batch_indices)

            return results
        except Exception as exc:
            raise RuntimeError(f"批量情感预测失败：{exc}") from exc


if __name__ == "__main__":
    predictor = SentimentPredictor()

    test_texts = [
        "这个东西质量很好，物流也很快",
        "东西很差，再也不买了",
        "包装不错，但是使用体验一般",
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"文本：{text}")
        print(f"结果：{result}")
        print("-" * 60)
