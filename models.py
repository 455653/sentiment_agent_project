from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


TensorLike = Union[torch.Tensor, list]


class BaseTextClassifier(nn.Module):
    """
    所有文本分类模型的公共父类。
    这里主要做两件事：
    1. 统一创建 Embedding 层，支持可选的预训练词向量初始化；
    2. 提供参数量统计方法，便于比较不同模型的规模。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.padding_idx = padding_idx
        self.embedding = self._build_embedding(pretrained_embedding)
        self.dropout = nn.Dropout(dropout)

    def _build_embedding(self, pretrained_embedding: Optional[TensorLike]) -> nn.Embedding:
        """
        创建 Embedding 层：
        - 如果传入预训练词向量，则用其初始化；
        - 否则使用随机初始化。
        """
        if pretrained_embedding is not None:
            embedding_weight = torch.as_tensor(pretrained_embedding, dtype=torch.float)
            if embedding_weight.dim() != 2:
                raise ValueError("pretrained_embedding 必须是二维矩阵，形状应为 [vocab_size, embedding_dim]")
            if embedding_weight.size(0) != self.vocab_size:
                raise ValueError(
                    f"预训练词向量行数与 vocab_size 不一致：{embedding_weight.size(0)} != {self.vocab_size}"
                )
            if embedding_weight.size(1) != self.embedding_dim:
                raise ValueError(
                    f"预训练词向量维度与 embedding_dim 不一致：{embedding_weight.size(1)} != {self.embedding_dim}"
                )
            embedding = nn.Embedding.from_pretrained(
                embedding_weight,
                freeze=False,
                padding_idx=self.padding_idx,
            )
        else:
            embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx,
            )
        return embedding

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        统计模型参数总量。
        - trainable_only=True: 只统计需要训练的参数；
        - trainable_only=False: 统计全部参数。
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters(self) -> int:
        """
        以属性形式返回可训练参数量，调用时更方便。
        """
        return self.count_parameters(trainable_only=True)


class SimpleRNN(BaseTextClassifier):
    """
    SimpleRNN 结构：
    Embedding -> 单层 RNN -> 取最后一个时间步 -> Dropout -> 全连接分类

    设计思路：
    - RNN 适合作为最基础的时序模型，能够按顺序处理文本；
    - 最后时间步的隐藏状态可以看作整条句子的压缩表示；
    - 结构简单，适合做基线模型。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        output, _ = self.rnn(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(self.dropout(last_output))
        return logits


class LSTMModel(BaseTextClassifier):
    """
    LSTMModel 结构：
    Embedding -> 单层 LSTM -> 取最后一个时间步 -> Dropout -> 全连接分类

    设计思路：
    - LSTM 在 RNN 的基础上加入门控机制；
    - 对长文本依赖的建模能力通常优于普通 RNN；
    - 依然保留“最后时间步表示整句”的经典做法。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        output, _ = self.lstm(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(self.dropout(last_output))
        return logits


class GRUModel(BaseTextClassifier):
    """
    GRUModel 结构：
    Embedding -> 单层 GRU -> 取最后一个时间步 -> Dropout -> 全连接分类

    设计思路：
    - GRU 也是门控循环网络；
    - 参数量通常比 LSTM 更少，训练速度可能更快；
    - 在很多文本分类任务中，GRU 是一个效果和效率都比较均衡的选择。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        output, _ = self.gru(embedded)
        last_output = output[:, -1, :]
        logits = self.fc(self.dropout(last_output))
        return logits


class BiLSTMModel(BaseTextClassifier):
    """
    BiLSTMModel 结构：
    Embedding -> 双向 LSTM -> 拼接前向/后向最后时间步 -> Dropout -> 全连接分类

    设计思路：
    - 单向 LSTM 主要从左到右编码；
    - 双向 LSTM 同时建模“前文信息”和“后文信息”；
    - 拼接双向特征后，句子表示更丰富，常用于情感分类任务。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        _, (hidden, _) = self.lstm(embedded)

        # hidden 的形状是 (num_layers * num_directions, batch_size, hidden_size)。
        # 单层双向 LSTM 中：
        # - hidden[-2] 是前向最后隐藏状态
        # - hidden[-1] 是后向最后隐藏状态
        # 将二者拼接后，得到整句的双向表示。
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        last_output = torch.cat([forward_hidden, backward_hidden], dim=1)
        logits = self.fc(self.dropout(last_output))
        return logits


class BiLSTMWithAttention(BaseTextClassifier):
    """
    BiLSTMWithAttention 结构：
    Embedding -> 双向 LSTM -> Bahdanau 风格注意力 -> 加权求和 -> Dropout -> 全连接分类

    设计思路：
    - 双向 LSTM 会为每个时间步输出一个上下文相关表示；
    - 注意力层不再只看“最后一个时间步”，而是让模型自动学习：
      哪些词对当前情感判断更重要；
    - 最终同时返回分类 logits 和 attention_weights，
      便于后续做可视化分析。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_proj = nn.Linear(hidden_size * 2, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_ids))
        lstm_output, _ = self.lstm(embedded)

        # Bahdanau 风格注意力的核心思想：
        # 1. 先把每个时间步的隐藏状态 h_t 投影到一个更小的注意力空间；
        # 2. 经过 tanh 引入非线性，学习“这个时间步是否重要”的中间表示；
        # 3. 再映射成一个标量分数 score_t；
        # 4. 对整句所有时间步的分数做 softmax，得到归一化注意力权重。
        attention_hidden = torch.tanh(self.attention_proj(lstm_output))
        attention_scores = self.attention_score(attention_hidden).squeeze(-1)

        # padding 位置不应该参与注意力分配，因此这里用 mask 把它们压到极小值。
        padding_mask = input_ids.ne(self.padding_idx)
        attention_scores = attention_scores.masked_fill(~padding_mask, torch.finfo(attention_scores.dtype).min)

        attention_weights = torch.softmax(attention_scores, dim=1)

        # 用注意力权重对所有时间步的双向 LSTM 输出做加权求和，
        # 得到整条句子的表示向量 context。
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        logits = self.fc(self.dropout(context))
        return logits, attention_weights


class TextCNN(BaseTextClassifier):
    """
    TextCNN 结构：
    Embedding -> 多个不同卷积核的一维卷积 -> ReLU -> 最大池化 -> 拼接 -> Dropout -> 全连接分类

    设计思路：
    - 不同大小的卷积核可以提取不同范围的局部 n-gram 特征；
    - 最大池化会保留每种卷积核最强的响应；
    - TextCNN 在文本分类任务中通常训练稳定、速度快、效果也不错。
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_classes: int = 2,
        num_filters: int = 128,
        kernel_sizes: tuple = (2, 3, 4),
        dropout: float = 0.3,
        padding_idx: int = 0,
        pretrained_embedding: Optional[TensorLike] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            padding_idx=padding_idx,
            pretrained_embedding=pretrained_embedding,
        )
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))

        # Conv1d 的输入格式是 (batch_size, channels, seq_len)，
        # 因此需要把 Embedding 输出从 (batch_size, seq_len, embedding_dim)
        # 转成 (batch_size, embedding_dim, seq_len)。
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []
        for conv in self.convs:
            conv_output = torch.relu(conv(embedded))
            pooled = torch.max(conv_output, dim=2).values
            pooled_outputs.append(pooled)

        features = torch.cat(pooled_outputs, dim=1)
        logits = self.fc(self.dropout(features))
        return logits


__all__ = [
    "SimpleRNN",
    "LSTMModel",
    "GRUModel",
    "BiLSTMModel",
    "BiLSTMWithAttention",
    "TextCNN",
]
