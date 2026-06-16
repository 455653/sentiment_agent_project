from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import chainlit as cl

from agents_system import create_review_crew, configure_deepseek_llm


WELCOME_MESSAGE = (
    "欢迎使用电商评论智能分析系统，\n"
    "请上传一个包含评论的txt文件（每行一条评论），或直接粘贴评论文本（每行一条）"
)
CHAINLIT_VERSION_HINT = "建议安装 `chainlit>=1.0,<2.0`"


def parse_reviews(text: str) -> List[str]:
    """
    将原始文本按行拆分成评论列表，并去掉空行。
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_reviews(reviews: List[str]) -> tuple[List[str], List[str]]:
    """
    验证评论数量并给出提示。
    - 无硬截断，允许传入任意数量
    - 去重并过滤空行
    - 超过 5000 条时给出"处理可能需要较长时间"的提示
    - 超过 10000 条时给出"建议分批处理"的提示，但不强制截断
    """
    notices: List[str] = []
    unique_reviews: List[str] = []
    seen: set[str] = set()

    for review in reviews:
        normalized = review.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_reviews.append(normalized)

    if len(unique_reviews) < len(reviews):
        notices.append(f"已自动去重并过滤空行：{len(reviews)} -> {len(unique_reviews)} 条。")

    if len(unique_reviews) > 10000:
        notices.append("评论数量超过 10000 条，建议分批处理以降低失败风险。")
    elif len(unique_reviews) > 5000:
        notices.append("评论数量超过 5000 条，处理可能需要较长时间，请耐心等待。")

    return unique_reviews, notices


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
    处理流程：
    1. "正在解析 {N} 条评论..."
    2. "正在进行批量情感分类（本地模型），共 {N} 条评论..."
    3. "情感分类完成：正面 X 条，负面 Y 条"
    4. "Agent 1 正在分析统计数据..."
    5. "Agent 2 正在提炼问题洞察..."
    6. "Agent 3 正在撰写分析报告..."
    7. 展示最终报告
    """
    await cl.Message(content=f"正在解析 {len(reviews)} 条评论...").send()

    if not reviews:
        await cl.Message(content="没有解析到有效评论，请重新上传 txt 文件或粘贴多行评论文本。").send()
        return

    normalized_reviews, notices = normalize_reviews(reviews)
    if not normalized_reviews:
        await cl.Message(content="没有解析到有效评论，请重新上传 txt 文件或粘贴多行评论文本。").send()
        return

    for notice in notices:
        await cl.Message(content=notice).send()

    await cl.Message(
        content=f"已接收来自 {source_name} 的 {len(normalized_reviews)} 条评论。"
    ).send()
    await cl.Message(
        content=f"正在进行批量情感分类（本地模型），共 {len(normalized_reviews)} 条评论..."
    ).send()

    try:
        async_run_crew = cl.make_async(run_crew_analysis_sync)
        analysis_result = await async_run_crew("\n".join(normalized_reviews))
    except Exception as exc:
        await cl.Message(content=f"分析失败：{exc}").send()
        return

    stage_messages = build_stage_messages(analysis_result)

    stage1_content = stage_messages[0]["content"] if len(stage_messages) >= 1 else ""
    positive_match = re.search(r"正面\s*(\d+)\s*条", stage1_content)
    negative_match = re.search(r"负面\s*(\d+)\s*条", stage1_content)
    if positive_match and negative_match:
        await cl.Message(
            content=f"情感分类完成：正面 {positive_match.group(1)} 条，负面 {negative_match.group(1)} 条"
        ).send()

    if len(stage_messages) >= 1:
        await cl.Message(content="Agent 1 正在分析统计数据...").send()
        await cl.Message(author=stage_messages[0]["author"], content=stage_messages[0]["content"]).send()

    if len(stage_messages) >= 2:
        await cl.Message(content="Agent 2 正在提炼问题洞察...").send()
        await cl.Message(author=stage_messages[1]["author"], content=stage_messages[1]["content"]).send()

    await cl.Message(content="Agent 3 正在撰写分析报告...").send()

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
        max_size_mb=20,
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
