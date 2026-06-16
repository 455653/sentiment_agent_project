from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chainlit as cl

from agents_system import create_review_crew, configure_deepseek_llm


WELCOME_MESSAGE = (
    "欢迎使用电商评论智能分析系统，\n"
    "请上传一个包含评论的txt文件（每行一条评论），或直接粘贴评论文本（每行一条）"
)
MAX_REVIEW_COUNT = 30
RECOMMENDED_REVIEW_RANGE = "20-30"
CHAINLIT_VERSION_HINT = "建议安装 `chainlit>=1.0,<2.0`"


def parse_reviews(text: str) -> List[str]:
    """
    将原始文本按行拆分成评论列表，并去掉空行。
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_reviews(reviews: List[str]) -> tuple[List[str], List[str]]:
    """
    对评论数量做友好提示。
    超过 30 条时保留前 30 条，避免分析时间过长。
    """
    notices: List[str] = []
    normalized_reviews = reviews

    if len(reviews) > MAX_REVIEW_COUNT:
        notices.append(
            f"建议输入 {RECOMMENDED_REVIEW_RANGE} 条评论以获得最佳分析效果和速度，"
            f"当前收到 {len(reviews)} 条，系统将先截取前 {MAX_REVIEW_COUNT} 条继续分析。"
        )
        normalized_reviews = reviews[:MAX_REVIEW_COUNT]

    return normalized_reviews, notices


def read_txt_file(file_path: str) -> str:
    """
    读取 txt 文件内容，并尽量兼容常见中文编码。
    """
    path = Path(file_path)
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb2312")

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="ignore")


def build_stage_messages(analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    将 CrewAI 的任务输出整理成前端可展示的中间结果消息。
    """
    icon_map = {
        "电商评论数据分析员": "📊",
        "用户反馈洞察分析师": "💡",
        "分析报告撰写专家": "📝",
    }

    stage_messages: List[Dict[str, str]] = []
    for stage in analysis_result.get("stages", []):
        role = str(stage.get("role", "未知Agent"))
        content = str(stage.get("content", "")).strip() or "该阶段未返回可展示内容。"
        icon = icon_map.get(role, "🤖")
        stage_messages.append(
            {
                "author": role,
                "content": f"{icon} {role}完成：\n\n{content}",
            }
        )

    return stage_messages


def run_crew_analysis_sync(reviews_text: str) -> Dict[str, Any]:
    """
    同步执行 CrewAI 分析流程。
    该函数会被 Chainlit 包装为异步调用，避免阻塞界面。
    """
    llm = configure_deepseek_llm()
    crew = create_review_crew(llm)
    result = crew.kickoff(inputs={"reviews": reviews_text})

    stages: List[Dict[str, str]] = []
    task_outputs = getattr(result, "tasks_output", []) or []

    for task, task_output in zip(crew.tasks, task_outputs):
        agent = getattr(task, "agent", None)
        role = getattr(agent, "role", "未知Agent")
        raw_output = getattr(task_output, "raw", str(task_output))
        stages.append({"role": str(role), "content": str(raw_output)})

    return {
        "final_report": getattr(result, "raw", str(result)),
        "stages": stages,
    }


async def process_reviews(reviews: List[str], source_name: str) -> None:
    """
    统一处理评论输入，并把多阶段分析结果发送到页面。
    """
    clean_reviews = [review.strip() for review in reviews if review.strip()]
    if not clean_reviews:
        await cl.Message(content="没有解析到有效评论，请重新上传 txt 文件或粘贴多行评论文本。").send()
        return

    normalized_reviews, notices = normalize_reviews(clean_reviews)
    for notice in notices:
        await cl.Message(content=notice).send()

    await cl.Message(
        content=f"已接收来自 {source_name} 的 {len(normalized_reviews)} 条评论。"
    ).send()
    await cl.Message(content=f"正在分析 {len(normalized_reviews)} 条评论，请稍候...").send()

    try:
        async_run_crew = cl.make_async(run_crew_analysis_sync)
        analysis_result = await async_run_crew("\n".join(normalized_reviews))
    except Exception as exc:
        await cl.Message(content=f"分析过程中出现错误：{exc}").send()
        return

    for stage_message in build_stage_messages(analysis_result):
        await cl.Message(
            author=stage_message["author"],
            content=stage_message["content"],
        ).send()

    final_report = str(analysis_result.get("final_report", "")).strip() or "未生成最终报告。"
    await cl.Message(content=final_report, author="最终分析报告").send()


def extract_reviews_from_message_elements(message: cl.Message) -> tuple[List[str], str] | None:
    """
    兼容用户直接在聊天输入框中附带 txt 文件的场景。
    """
    for element in getattr(message, "elements", []) or []:
        element_path = getattr(element, "path", None)
        element_name = getattr(element, "name", "上传文件")
        if not element_path:
            continue

        suffix = Path(str(element_path)).suffix.lower()
        if suffix != ".txt":
            continue

        file_content = read_txt_file(str(element_path))
        return parse_reviews(file_content), f"聊天附件 `{element_name}`"

    return None


async def prompt_txt_upload() -> None:
    """
    在会话开始时主动提示用户上传 txt 文件。
    如果用户没有上传，也仍然可以直接在输入框粘贴评论。
    """
    uploaded_files = await cl.AskFileMessage(
        content="如需上传评论文件，请选择一个 txt 文件；如果你想直接粘贴评论，也可以忽略这个上传框。",
        accept={"text/plain": [".txt"]},
        max_size_mb=5,
        max_files=1,
        timeout=120,
        raise_on_timeout=False,
    ).send()

    if not uploaded_files:
        return

    file = uploaded_files[0]
    try:
        file_content = read_txt_file(file.path)
        reviews = parse_reviews(file_content)
        await process_reviews(reviews, source_name=f"上传文件 `{file.name}`")
    except Exception as exc:
        await cl.Message(content=f"读取上传文件失败：{exc}").send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """
    会话开始时显示欢迎说明，并提供文件上传入口。
    """
    await cl.Message(
        content=f"{WELCOME_MESSAGE}\n\n{CHAINLIT_VERSION_HINT}"
    ).send()
    await prompt_txt_upload()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    处理用户在输入框中直接粘贴的多行评论文本，
    或直接附带上传的 txt 文件。
    """
    file_reviews = extract_reviews_from_message_elements(message)
    if file_reviews is not None:
        reviews, source_name = file_reviews
        await process_reviews(reviews, source_name=source_name)
        return

    message_content = str(message.content).strip()
    if not message_content:
        await cl.Message(content="请输入评论文本，或上传一个 txt 文件。").send()
        return

    reviews = parse_reviews(message_content)
    await process_reviews(reviews, source_name="聊天输入")
