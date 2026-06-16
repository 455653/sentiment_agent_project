from __future__ import annotations

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
SENTIMENT_TOOL_NAME = "sentiment_analysis_tool"
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


@tool(SENTIMENT_TOOL_NAME)
def sentiment_analysis_tool(text: str) -> str:
    """
    情感分析工具：对单条电商评论进行情感分析。
    输入一条中文评论，返回情感标签和置信度字符串。
    """
    review = text.strip()
    if not review:
        return "情感分析失败: 输入文本为空"

    print(f"[工具] 正在分析评论: {review}")
    try:
        result = get_predictor().predict(review)
        return f"情感: {result['label']}, 置信度: {result['confidence']:.4f}"
    except Exception as exc:
        error_message = f"情感分析工具执行失败: {exc}"
        print(f"[工具] {error_message}")
        raise RuntimeError(error_message) from exc


def create_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
    """
    创建 3 个协作 Agent。
    """
    data_analyst = Agent(
        role="电商评论数据分析员",
        goal="对一批评论逐条进行情感分类，统计正负面比例",
        backstory="你是一位细致严谨的数据分析专家，擅长逐条核对评论并输出可复核的统计结论。",
        tools=[sentiment_analysis_tool],
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
            f"请逐条调用工具 {SENTIMENT_TOOL_NAME} 进行分类，不要凭空猜测。"
            "这个工具对应的中文用途是“情感分析工具”。\n"
            "如果工具调用失败，请停止统计，并明确输出失败原因。\n"
            "输出内容至少包括：\n"
            "1. 每条评论的情感判断结果；\n"
            "2. 汇总统计：正面X条，负面Y条，正面占比Z%；\n"
            "3. 负面评论原文列表；\n"
            "4. 正面评论原文列表，供后续优点分析使用。"
        ),
        expected_output="一段包含逐条情感结果、整体统计、负面评论列表和正面评论列表的中文文本。",
        agent=data_analyst,
    )

    task2 = Task(
        description=(
            "基于上一个任务的输出，分析负面评论中反复出现的问题点，"
            "从物流、质量、客服、包装、使用体验等维度归纳出 3-5 个主要问题；"
            "同时结合正面评论，简要总结用户称赞的优点。"
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
