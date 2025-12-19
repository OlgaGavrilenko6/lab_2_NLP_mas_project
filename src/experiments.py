from __future__ import annotations
from typing import Any, Dict, List
from .graph import build_graph_with_retry_loop
from .config import get_llm
from .nodes import ExperimentComment
from .utils import _short
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from .retry import invoke_with_parser_retry
import json

# Модель говорит что можно было бы улучшить
def make_experiment_comment(llm_factory, out: dict) -> ExperimentComment:
    parser = PydanticOutputParser(pydantic_object=ExperimentComment)

    system = (
        "Ты — строгий оценщик качества ответа мультиагентной системы.\n"
        "Оцени полезность результата и укажи, что улучшить.\n"
        "Ориентируйся на:\n"
        "- корректность intent/маршрутизации,\n"
        "- осмысленность tool calls,\n"
        "- использовалась ли память по делу,\n"
        "- полноту и конкретику ответа.\n"
        "Ответ строго JSON по схеме.\n"
        f"{parser.get_format_instructions()}"
    )

    ctx = {
        "query": out.get("query"),
        "intent": out.get("intent"),
        "activated_nodes": out.get("activated_nodes", []),
        "handoff_log": out.get("handoff_log", []),
        "tools_used_count": len(out.get("tool_calls", [])),
        "tool_calls_tail": out.get("tool_calls", [])[-8:],
        "memory_summary": out.get("memory_summary", ""),
        "memory_hits": out.get("memory_hits", []),
        "final_answer": out.get("final_answer", ""),
    }

    def make_llm(temp: float):
        return llm_factory(temp)

    comment: ExperimentComment = invoke_with_parser_retry(
        make_llm=make_llm,
        messages=[
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(ctx, ensure_ascii=False))
        ],
        parser=parser,
        max_retries=3,
        temps=(0.0, 0.3, 0.7),
    )
    return comment

# Запускаем пайплайн мультиагентоной системы
def run_system(query: str, thread_id: str = "u1", max_rounds: int = 3):
    app = build_graph_with_retry_loop()

    init = {
        "query": query,
        "intent": None,
        "plan": [],
        "tool_context": [],
        "focus": "",
        "need_more": False,
        "round": 0,
        "max_rounds": max_rounds,
        "partial": "",
        "final_answer": "",
        "history": [],
        "memory_notes": [],
        "memory_hits": [],
        "memory_summary": "",
        "activated_nodes": [],
        "tool_calls": [],
        "handoff_log": [],
        "thread_id": thread_id,
        "verbose": True,
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 120}

    last = None
    # Показываем какие ключи state обновились на каждом шаге
    for update in app.stream(init, config=config, stream_mode="updates"):
        last = update
        for node_name, patch in update.items():
            print(f"\n===---NODE: {node_name}---===")
            if "handoff_log" in patch:
                print("handoff:", patch["handoff_log"][-1:])
            if "tool_calls" in patch:
                print("tool_calls +", len(patch["tool_calls"]))
            if "partial" in patch:
                print("partial:", _short(patch["partial"], 300))
            if "final_answer" in patch:
                print("final_answer:", _short(patch["final_answer"], 300))

    # Возвращаем финальный state
    out = app.invoke(init, config=config)
    return out


# Запросы
def run_experiments():
    queries = [
        # Концептуальный вопрос
        "Объясни разницу между supervisor и planner-executor паттернами в мультиагентных системах.",
        # Архитектура
        "Как спроектировать state для LangGraph, если есть router, несколько агентов и нужна память?",
        # Программирование
        "Как в LangGraph сделать условную маршрутизацию по intent и залогировать порядок вызова узлов?",
        # Повседневная задача
        "Как приготовить штрудель?",
        # Обзор литературы
        "Дай поисковые запросы и критерии отбора литературы по теме phishing susceptibility personality traits."
    ]

    app = build_graph_with_retry_loop()
    results = []

    def llm_factory(temp: float):
        return get_llm(temperature=temp)

    for i, q in enumerate(queries, 1):
        out = run_system(q, thread_id=f"exp_{i}")
        comment_obj = make_experiment_comment(llm_factory, out)
        results.append({
            "query": q,
            "intent": out.get("intent"),
            "activated_nodes": out.get("activated_nodes", []),
            "handoff": out.get("handoff_log", []),
            "tools_used_count": len(out.get("tool_calls", [])),
            "memory_used": bool(out.get("memory_hits")),
            "memory_summary": out.get("memory_summary", ""),
            "answer_head": (out.get("final_answer", "") or "")[:400],
            "comment": comment_obj.model_dump(),
        })

    return results
