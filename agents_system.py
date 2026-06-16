from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

from data_preprocessing import clean_data, read_online_shopping_csv
from sentiment_predictor import SentimentPredictor


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
BATCH_SENTIMENT_TOOL_NAME = "sentiment_batch_analysis_tool"
_predictor: Optional[SentimentPredictor] = None


def configure_deepseek_llm() -> LLM:
    """
    加载 .env 中的 DeepSeek API Key，并创建 CrewAI 使用的 LLM 对象。
    这里同时设置 OpenAI 兼容环境变量，便于 CrewAI / LiteLLM 兼容调用。
    """
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未在 .env 中找到 DEEPSEEK_API_KEY，请先配置后再运行。")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL

    print("[系统] 已加载 DeepSeek API 配置。")
    return LLM(
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
    )


def get_predictor() -> SentimentPredictor:
    """
    懒加载情感预测器，避免模块导入时就强制加载模型文件。
    """
    global _predictor
    if _predictor is None:
        print("[工具] 正在加载情感分析模型...")
        _predictor = SentimentPredictor()
        print("[工具] 情感分析模型加载完成。")
    return _predictor


@tool(BATCH_SENTIMENT_TOOL_NAME)
def sentiment_batch_analysis_tool(reviews_text: str) -> str:
    """
    批量情感分析工具：一次性对所有评论进行情感分类。
    输入：多行评论文本，每行一条评论（用 \\n 分隔）
    输出：JSON 字符串（ensure_ascii=False），包含：
      {
        "total": 总条数,
        "positive": 正面条数,
        "negative": 负面条数,
        "positive_ratio": 正面占比(浮点数,如0.7234),
        "results": [每条 {"text": "原文", "label": "正面/负面", "confidence": 0.xxxx}],
        "representative_samples": {
          "positive_high_conf": [...高置信度正面样例(前3条)...],
          "positive_low_conf": [...低置信度正面样例(前3条)...],
          "negative_high_conf": [...高置信度负面样例(前3条)...],
          "negative_low_conf": [...低置信度负面样例(前3条)...]
        }
      }
    """
    try:
        reviews = [line.strip() for line in reviews_text.splitlines() if line.strip()]
        if not reviews:
            return json.dumps({"error": "输入评论为空，请提供每行一条评论的文本。"}, ensure_ascii=False)

        print(f"[工具] 正在进行批量情感分析，共 {len(reviews)} 条评论...")
        batch_results = get_predictor().predict_batch(reviews, batch_size=128)

        results = [
            {
                "text": str(item.get("text", "")),
                "label": str(item.get("label", "")),
                "confidence": float(item.get("confidence", 0.0)),
            }
            for item in batch_results
        ]

        positive = [r for r in results if r["label"] == "正面"]
        negative = [r for r in results if r["label"] == "负面"]

        positive_high = sorted(positive, key=lambda x: x["confidence"], reverse=True)[:3]
        positive_low = sorted(positive, key=lambda x: x["confidence"])[:3]
        negative_high = sorted(negative, key=lambda x: x["confidence"], reverse=True)[:3]
        negative_low = sorted(negative, key=lambda x: x["confidence"])[:3]

        total = len(results)
        positive_count = len(positive)
        negative_count = len(negative)
        positive_ratio = round(positive_count / total, 4) if total else 0.0

        payload = {
            "total": total,
            "positive": positive_count,
            "negative": negative_count,
            "positive_ratio": positive_ratio,
            "results": results,
            "representative_samples": {
                "positive_high_conf": positive_high,
                "positive_low_conf": positive_low,
                "negative_high_conf": negative_high,
                "negative_low_conf": negative_low,
            },
        }

        print("[工具] 批量情感分析完成。")
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"批量情感分析失败：{exc}"}, ensure_ascii=False)


def create_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
    """
    创建 3 个协作 Agent。
    """
    data_analyst = Agent(
        role="电商评论数据分析员",
        goal="对一批评论逐条进行情感分类，统计正负面比例",
        backstory="你是一位细致严谨的数据分析专家，擅长逐条核对评论并输出可复核的统计结论。",
        tools=[sentiment_batch_analysis_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    insight_analyst = Agent(
        role="用户反馈洞察分析师",
        goal="从负面评论中提炼用户主要抱怨的问题点，从正面评论中提炼用户称赞的优点",
        backstory="你是一位善于总结归纳的市场分析专家，能够从评论中快速归纳共性问题与亮点。",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    report_writer = Agent(
        role="分析报告撰写专家",
        goal="将数据统计结果和洞察分析整合成一份结构清晰的中文分析报告",
        backstory="你是一位擅长商业写作的报告专家，能够把分析结论整理成结构清晰、表达专业的中文报告。",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return data_analyst, insight_analyst, report_writer


def create_tasks(data_analyst: Agent, insight_analyst: Agent, report_writer: Agent) -> tuple[Task, Task, Task]:
    """
    创建按顺序执行的 3 个任务。
    """
    task1 = Task(
        description=(
            "你会收到变量 {reviews}，其中包含一批电商评论文本，多条评论之间以换行分隔。\n"
            f"请使用批量情感分析工具 {BATCH_SENTIMENT_TOOL_NAME}，一次性对所有评论进行分类。"
            "不要逐条调用工具！调用一次即可。\n"
            "工具会返回一个 JSON，包含所有评论的分类结果、汇总统计和代表样例。\n"
            "如果 JSON 中包含 error 字段，请停止后续分析并输出失败原因。\n"
            "你需要输出：\n"
            "1. 汇总统计：正面X条（占Y%），负面Z条（占W%）；\n"
            "2. 代表样例分析：简要说明高置信度正/负面和低置信度正/负面分别反映了什么特点；\n"
            "3. 不要逐条列出所有评论的结果（数量太多）。"
        ),
        expected_output="一段包含情感分布统计和代表样例分析的中文文本，不包含逐条结果。",
        agent=data_analyst,
    )

    task2 = Task(
        description=(
            "基于数据分析员输出的统计结果和代表样例，分析这批评论中反映的主要问题点和优点。"
            "从物流、质量、客服、包装、使用体验等维度归纳出 3-5 个主要问题；"
            "同时总结用户称赞的优点。注意代表样例中低置信度的评论可能反映模棱两可的情况，"
            "请特别留意。"
        ),
        expected_output="一段包含主要问题点、原因归纳和优点总结的中文文本。",
        agent=insight_analyst,
        context=[task1],
    )

    task3 = Task(
        description=(
            "综合前两个任务的结果，输出一份 markdown 格式的中文分析报告。"
            "报告需要包含以下部分：标题、整体情感分布、主要问题点、改进建议、优点总结。"
            "语言要专业、清晰、适合给业务同学阅读。"
        ),
        expected_output="一份完整的 markdown 格式中文分析报告。",
        agent=report_writer,
        context=[task1, task2],
    )

    return task1, task2, task3


def create_review_crew(llm: LLM) -> Crew:
    """
    创建顺序执行的 Crew。
    """
    data_analyst, insight_analyst, report_writer = create_agents(llm)
    task1, task2, task3 = create_tasks(data_analyst, insight_analyst, report_writer)

    return Crew(
        agents=[data_analyst, insight_analyst, report_writer],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )


def split_dataset_indices(
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[List[int], List[int], List[int]]:
    """
    复用数据预处理脚本的分层切分逻辑，得到 train/val/test 对应的原始行索引。
    这样可以把 test.pkl 对应回原始评论文本，而不是从全量数据里随意抽样。
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    rng = random.Random(random_state)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for indices in label_to_indices.values():
        current_indices = list(indices)
        rng.shuffle(current_indices)
        total_count = len(current_indices)

        train_count = int(round(total_count * train_ratio))
        val_count = int(round(total_count * val_ratio))
        if train_count + val_count > total_count:
            val_count = max(0, total_count - train_count)

        train_indices.extend(current_indices[:train_count])
        val_indices.extend(current_indices[train_count : train_count + val_count])
        test_indices.extend(current_indices[train_count + val_count :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def decode_reviews_from_test_pkl(test_path: Path, vocab_path: Path) -> List[str]:
    """
    如果只能拿到 test.pkl，则尝试通过 vocab 反解 token 并拼接成近似文本。
    这不是严格意义上的原始评论，但可以作为兜底演示输入。
    """
    with open(test_path, "rb") as file:
        test_data = pickle.load(file)
    with open(vocab_path, "rb") as file:
        vocab_data = pickle.load(file)

    idx2word = vocab_data.get("idx2word", {})
    pad_idx = int(vocab_data.get("pad_idx", 0))
    unk_idx = int(vocab_data.get("unk_idx", 1))

    reviews: List[str] = []
    for input_ids in test_data.get("input_ids", []):
        tokens: List[str] = []
        for token_id in input_ids:
            token_id = int(token_id)
            if token_id in {pad_idx, unk_idx}:
                continue
            token = idx2word.get(token_id, idx2word.get(str(token_id), ""))
            if token:
                tokens.append(token)
        if tokens:
            reviews.append("".join(tokens))

    return reviews


def load_reviews_from_test_split(
    sample_size: int = 20,
    csv_path: str | None = None,
    test_path: str | None = None,
    vocab_path: str | None = None,
) -> List[str]:
    """
    优先从 test.pkl 对应的测试集原始评论中抽样。
    如果无法精确映射，再退回到根据 vocab 反解出的近似文本。
    """
    base_dir = Path(__file__).resolve().parent
    final_csv_path = Path(csv_path) if csv_path else base_dir / "data" / "online_shopping_10_cats.csv"
    final_test_path = Path(test_path) if test_path else base_dir / "data" / "test.pkl"
    final_vocab_path = Path(vocab_path) if vocab_path else base_dir / "data" / "vocab.pkl"

    if not final_test_path.exists():
        raise FileNotFoundError(f"找不到测试集文件: {final_test_path}")

    if final_csv_path.exists():
        print(f"[系统] 正在根据 test.pkl 对齐测试集原始评论: {final_test_path}")
        with open(final_test_path, "rb") as file:
            test_data = pickle.load(file)

        df = read_online_shopping_csv(str(final_csv_path))
        df = clean_data(df)
        if df.empty:
            raise ValueError("评论数据为空，无法构造测试集示例输入。")

        _, _, test_indices = split_dataset_indices(df["label"].astype(int).tolist(), random_state=42)
        test_reviews = df.iloc[test_indices]["review"].astype(str).tolist()

        if len(test_reviews) != len(test_data.get("labels", [])):
            raise ValueError("根据 csv 还原出的测试集长度与 test.pkl 不一致，无法安全使用。")

        sample_count = min(sample_size, len(test_reviews))
        reviews = random.Random(42).sample(test_reviews, sample_count)
        print(f"[系统] 已从 test.pkl 对应的测试集原文中抽取 {len(reviews)} 条评论。")
        return reviews

    if final_vocab_path.exists():
        print("[系统] 未找到原始 csv，将从 test.pkl 反解近似评论文本。")
        decoded_reviews = decode_reviews_from_test_pkl(final_test_path, final_vocab_path)
        if not decoded_reviews:
            raise ValueError("test.pkl 中没有可用于演示的评论内容。")

        sample_count = min(sample_size, len(decoded_reviews))
        reviews = random.Random(42).sample(decoded_reviews, sample_count)
        print(f"[系统] 已从 test.pkl 反解文本中抽取 {len(reviews)} 条评论。")
        return reviews

    raise FileNotFoundError("存在 test.pkl，但缺少用于还原评论文本的 csv 或 vocab.pkl。")


def load_sample_reviews(sample_size: int = 20, csv_path: str | None = None) -> List[str]:
    """
    优先从 test.pkl 对应的测试集评论中抽样；如果测试集不可用，再回退到原始 csv。
    """
    base_dir = Path(__file__).resolve().parent
    final_csv_path = Path(csv_path) if csv_path else base_dir / "data" / "online_shopping_10_cats.csv"

    try:
        return load_reviews_from_test_split(sample_size=sample_size, csv_path=str(final_csv_path))
    except FileNotFoundError as exc:
        print(f"[系统] 未找到 test.pkl，改为从原始 csv 抽样。原因: {exc}")

    if not final_csv_path.exists():
        raise FileNotFoundError(f"找不到评论数据文件: {final_csv_path}")

    print(f"[系统] 正在从原始评论数据抽样: {final_csv_path}")
    df = read_online_shopping_csv(str(final_csv_path))
    df = clean_data(df)

    if df.empty:
        raise ValueError("评论数据为空，无法构造示例输入。")

    sample_count = min(sample_size, len(df))
    sampled_df = df.sample(n=sample_count, random_state=42)
    reviews = sampled_df["review"].astype(str).tolist()
    print(f"[系统] 已从原始 csv 中抽取 {len(reviews)} 条评论作为示例输入。")
    return reviews


def get_fallback_reviews() -> List[str]:
    """
    当原始数据读取失败时，使用内置评论做演示。
    """
    return [
        "这个东西质量很好，物流也很快",
        "东西很差，再也不买了",
        "包装挺精致的，送人也合适",
        "客服回复太慢，体验不好",
        "价格实惠，性价比很高",
        "用了两天就坏了，质量不行",
        "外观很好看，整体很满意",
        "物流太慢了，等了好几天",
        "做工不错，细节处理也可以",
        "描述和实物差距很大，有点失望",
    ]


def main() -> str:
    """
    运行完整的多 Agent 评论分析流程。
    """
    print("[系统] 开始初始化 CrewAI 评论分析系统...")
    llm = configure_deepseek_llm()

    print("[系统] 正在创建 Agent 和 Task...")
    crew = create_review_crew(llm)

    try:
        reviews = load_sample_reviews(sample_size=20)
    except Exception as exc:
        print(f"[系统] 读取评论样本失败，将改用内置示例评论。原因: {exc}")
        reviews = get_fallback_reviews()

    reviews_text = "\n".join(reviews)

    print("[系统] 即将启动 Crew 顺序执行流程...")
    result = crew.kickoff(inputs={"reviews": reviews_text})
    final_report = getattr(result, "raw", str(result))

    print("\n================ 最终分析报告 ================\n")
    print(final_report)
    return final_report


if __name__ == "__main__":
    main()
