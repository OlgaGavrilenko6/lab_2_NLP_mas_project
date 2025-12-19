from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langgraph.prebuilt import create_react_agent  # оставляем (у тебя оно работает)

from .config import get_llm
from .memory_store import load_notes, simple_retrieve_notes
from .retry import invoke_with_parser_retry
from .state import MASState, Intent
from .tools import TOOLS_CODING, TOOLS_DAILY, TOOLS_LITERATURE, search_user_notes, save_user_note
from .utils import (
    now_iso,
    _coerce_text,
    _short,
    add_history,
    add_tool_log,
    add_node_log,
    trim_history,
)

# Planner схема
class RouteDecision(BaseModel):
    intent: Intent = Field(..., description="Маршрут: conceptual|architecture|coding|daily|literature")
    reasoning: str = Field("", description="Коротко почему так")

class PlanOut(BaseModel):
    plan: List[str] = Field(..., description="5–10 шагов плана")

class ReviewDecision(BaseModel):
    need_more: bool = Field(..., description="Нужно ли добирать информацию через инструменты")
    focus: str = Field("", description="Что конкретно добрать/уточнить (коротко)")
    improved_answer: str = Field("", description="Улучшенная версия текста, если уже можно улучшить без добора")

class ExperimentComment(BaseModel):
    helpful: bool = Field(..., description="Был ли ответ полезен пользователю")
    issues: List[str] = Field(default_factory=list, description="Что было не так (коротко)")
    improvements: List[str] = Field(default_factory=list, description="Что улучшить (коротко и конкретно)")


# Агенты (ноды)
def router_node(state: MASState) -> MASState:
    """
    Определяет тип запроса пользователя и фиксирует handoff
    """
    add_node_log(state, "router")

    # Долговременная память
    notes = load_notes()
    hits = simple_retrieve_notes(notes, state["query"], k=4)
    state["memory_notes"] = notes
    state["memory_hits"] = hits

    # Отчет роутера парсим с помощью PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=RouteDecision)

    system = (
        "Ты — Router мультиагентной системы.\n"
        "Определи intent запроса одним из:\n"
        "- conceptual: теоретика MAS/LLM\n"
        "- architecture: проектирование/архитектура\n"
        "- coding: реализация/код\n"
        "- daily: повседневные задачи\n"
        "- literature: поиск/обзор литературы\n"
        "Ответ строго JSON по схеме.\n"
        f"{parser.get_format_instructions()}"
    )

    ctx = {"query": state["query"], "memory_hits": hits, "history_tail": state["history"][-4:]}

    def make_llm(temp: float):
        return get_llm(temperature=temp)

    decision: RouteDecision = invoke_with_parser_retry(
        make_llm=make_llm,
        messages=[
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(ctx, ensure_ascii=False)),
        ],
        parser=parser,
        max_retries=3,
        temps=(0.1, 0.2, 0.3),
    )

    state["intent"] = decision.intent
    # Фиксируем handoff router передал управление нужному агенту
    state["handoff_log"].append(f"[handoff] router -> {decision.intent} | {decision.reasoning}")

    # Обновляем историю диалога в оперативной памяти сессии
    add_history(state, "user", state["query"])
    trim_history(state)
    return state


# Узел Planner: строит план ответа из 5–10 шагов под запрос пользователя
def planner_node(state: MASState) -> MASState:
    add_node_log(state, "planner")

    parser = PydanticOutputParser(pydantic_object=PlanOut)
    system = (
        "Ты — Planner.\n"
        "Составь короткий план решения из 5–10 шагов по запросу пользователя.\n"
        "Учитывай intent, memory_hits и историю.\n"
        "Ответ строго JSON по схеме.\n"
        f"{parser.get_format_instructions()}"
    )

    ctx = {
        "query": state["query"],
        "intent": state["intent"],
        "memory_hits": state["memory_hits"],
        "history_tail": state["history"][-4:],
    }

    def make_llm(temp: float):
        return get_llm(temperature=temp)

    out: PlanOut = invoke_with_parser_retry(
        make_llm=make_llm,
        messages=[
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(ctx, ensure_ascii=False)),
        ],
        parser=parser,
        max_retries=3,
        temps=(0.2, 0.5, 0.8),
    )

    plan = out.plan if isinstance(out.plan, list) and out.plan else []
    if not plan:
        plan = ["Уточнить цель", "Собрать контекст", "Сформировать ответ", "Проверить результат"]

    state["plan"] = [str(x) for x in plan][:12]
    return state


# Вызывает LLM и формирует теоретический структурированный ответ: определения, ключевые идеи
def conceptual_agent_node(state: MASState) -> MASState:
    add_node_log(state, "conceptual_agent")

    llm = get_llm(temperature=0.2)
    system = (
        "Ты — conceptual-агент (теория MAS/LLM-агенты).\n"
        "Дай структурированный ответ: определения, ключевые идеи, 1–2 примера.\n"
        "Опирайся на plan + memory_hits + tool_context.\n"
        "Если чего-то не хватает — явно укажи, что именно.\n"
    )

    ctx = {
        "query": state["query"],
        "plan": state.get("plan", []),
        "memory_hits": state.get("memory_hits", []),
        "tool_context_tail": state.get("tool_context", [])[-8:],
        "history_tail": state.get("history", [])[-4:],
    }

    raw = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(ctx, ensure_ascii=False))
    ])
    state["partial"] = _coerce_text(raw)
    return state


# Агент, который генерирует ответ по структуре (компоненты, state, handoff, tools, memory)
def architecture_agent_node(state: MASState) -> MASState:
    add_node_log(state, "architecture_agent")

    llm = get_llm(temperature=0.2)
    system = (
        "Ты — architecture-агент (архитектура/дизайн).\n"
        "Сформируй ответ с блоками:\n"
        "1) компоненты и роли агентов,\n"
        "2) state поля,\n"
        "3) handoff/маршрутизация,\n"
        "4) tool calling,\n"
        "5) memory management,\n"
        "6) риски и улучшения.\n"
        "Опирайся на plan + memory_hits + tool_context.\n"
    )

    ctx = {
        "query": state["query"],
        "plan": state.get("plan", []),
        "memory_hits": state.get("memory_hits", []),
        "tool_context_tail": state.get("tool_context", [])[-8:],
        "history_tail": state.get("history", [])[-4:],
    }

    raw = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(ctx, ensure_ascii=False))
    ])
    state["partial"] = _coerce_text(raw)
    return state


# Проверка похожести ответа от модели на код (нужно для кодингового агента)
def _looks_like_code(text: str) -> bool:
    t = (text or "")
    if "```" in t:
        return True
    # Эвристики
    return any(x in t for x in ["def ", "class ", "import ", "from ", "pip install", "```python"])


# Кодинговый ответ
def coding_agent_node(state: MASState) -> MASState:
    add_node_log(state, "coding_agent")

    tools = TOOLS_CODING
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — coding-агент.\n"
         "Твоя задача: выдать ИСПОЛНИМЫЙ код.\n"
         "Формат ответа ОБЯЗАТЕЛЕН:\n"
         "1) Один кодовый блок ```python ... ``` (или ```bash``` если нужно)\n"
         "2) Ниже 2–5 строк инструкции как запустить.\n"
         "3) Обязательно напиши код по синтаксическим правилам написания кода. \n"
         "Если не хватает данных — предположи разумные значения и отметь TODO в коде.\n"
         "Инструменты: calc, search_user_notes, save_user_note.\n"
         "Если полезно — сначала search_user_notes.\n"),
        ("placeholder", "{messages}")
    ])

    user_msg = (
        f"QUERY: {state['query']}\n"
        f"PLAN: {state.get('plan', [])}\n"
        f"MEMORY_HITS: {state.get('memory_hits', [])}\n"
        f"TOOL_CONTEXT_TAIL: {state.get('tool_context', [])[-8:]}\n"
        f"HISTORY_TAIL: {state.get('history', [])[-4:]}\n"
    )

    # 1-я попытка
    llm = get_llm(temperature=0.0)
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
    res = agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"recursion_limit": 40, "configurable": {"thread_id": state["thread_id"]}}
    )

    # Логируем ответы интрументов
    for m in res["messages"]:
        if m.__class__.__name__.startswith("ToolMessage"):
            content = getattr(m, "content", "")
            add_tool_log(state, "tool_message", {"content": content})
            state["tool_context"].append({"ts": now_iso(), "tool_message": content})

    text = _coerce_text(res["messages"][-1])

    # 2-я попытка, если кода нет
    if not _looks_like_code(text):
        llm2 = get_llm(temperature=0.4)
        agent2 = create_react_agent(model=llm2, tools=tools, prompt=prompt)
        res2 = agent2.invoke(
            {"messages": [HumanMessage(
                content=user_msg + "\n\nВАЖНО: верни ответ строго в формате: ```python``` + инструкции.")]},
            config={"recursion_limit": 40, "configurable": {"thread_id": state["thread_id"]}}
        )
        for m in res2["messages"]:
            if m.__class__.__name__.startswith("ToolMessage"):
                content = getattr(m, "content", "")
                add_tool_log(state, "tool_message", {"content": content})
                state["tool_context"].append({"ts": now_iso(), "tool_message": content})
        text = _coerce_text(res2["messages"][-1])

    state["partial"] = text
    return state


# Агент повседневного типа ответов
def daily_agent_node(state: MASState) -> MASState:
    add_node_log(state, "daily_agent")

    tools = TOOLS_DAILY
    llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — daily-агент (повседневные задачи).\n"
         "Инструменты: days_until, calc, search_user_notes, save_user_note.\n"
         "Если есть дата/дедлайн — days_until.\n"
         "Если есть расчёты — calc.\n"
         "Если выдаёшь полезный план/чеклист — сохрани save_user_note.\n"),
        ("placeholder", "{messages}")
    ])

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    user_msg = (
        f"QUERY: {state['query']}\n"
        f"PLAN: {state.get('plan', [])}\n"
        f"MEMORY_HITS: {state.get('memory_hits', [])}\n"
        f"TOOL_CONTEXT_TAIL: {state.get('tool_context', [])[-8:]}\n"
        f"HISTORY_TAIL: {state.get('history', [])[-4:]}\n"
    )

    res = agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"recursion_limit": 40, "configurable": {"thread_id": state["thread_id"]}}
    )

    for m in res["messages"]:
        if m.__class__.__name__.startswith("ToolMessage"):
            content = getattr(m, "content", "")
            add_tool_log(state, "tool_message", {"content": content})
            state["tool_context"].append({"ts": now_iso(), "tool_message": content})

    state["partial"] = _coerce_text(res["messages"][-1])
    return state


# Агент обзора литературы
def literature_agent_node(state: MASState) -> MASState:
    add_node_log(state, "literature_agent")

    tools = TOOLS_LITERATURE
    llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — literature-агент.\n"
         "Сформируй:\n"
         "1) 5–10 поисковых запросов (лучше на английском)\n"
         "2) критерии отбора статей\n"
         "3) структуру обзора\n"
         "Инструменты: search_user_notes, save_user_note.\n"
         "Если есть полезные выводы — сохрани заметку.\n"),
        ("placeholder", "{messages}")
    ])

    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    user_msg = (
        f"QUERY: {state['query']}\n"
        f"PLAN: {state.get('plan', [])}\n"
        f"MEMORY_HITS: {state.get('memory_hits', [])}\n"
        f"TOOL_CONTEXT_TAIL: {state.get('tool_context', [])[-8:]}\n"
        f"HISTORY_TAIL: {state.get('history', [])[-4:]}\n"
    )

    res = agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"recursion_limit": 40, "configurable": {"thread_id": state["thread_id"]}}
    )

    for m in res["messages"]:
        if m.__class__.__name__.startswith("ToolMessage"):
            content = getattr(m, "content", "")
            add_tool_log(state, "tool_message", {"content": content})
            state["tool_context"].append({"ts": now_iso(), "tool_message": content})

    state["partial"] = _coerce_text(res["messages"][-1])
    return state


# Ревьюер
def reviewer_node(state: MASState) -> MASState:
    add_node_log(state, "reviewer")

    parser = PydanticOutputParser(pydantic_object=ReviewDecision)

    system = (
        "Ты — reviewer.\n"
        "Проверь: хватает ли информации для хорошего ответа.\n"
        "Если НЕ хватает: need_more=true и в focus напиши ЧТО добрать через инструменты.\n"
        "Если хватает: need_more=false и improved_answer содержит улучшенную версию (или пусто).\n"
        "Ответ строго JSON.\n"
        f"{parser.get_format_instructions()}"
    )

    ctx = {
        "query": state["query"],
        "intent": state["intent"],
        "plan": state["plan"],
        "draft": state["partial"],
        "tool_context_tail": state["tool_context"][-8:],
        "memory_hits": state["memory_hits"],
        "round": state["round"],
        "max_rounds": state["max_rounds"],
    }

    def make_llm(temp: float):
        return get_llm(temperature=temp)

    decision: ReviewDecision = invoke_with_parser_retry(
        make_llm=make_llm,
        messages=[
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(ctx, ensure_ascii=False)),
        ],
        parser=parser,
        max_retries=3,
        temps=(0.0, 0.4, 0.8),
    )

    # Ограничение по числу циклов
    if state["round"] >= state["max_rounds"]:
        decision.need_more = False

    state["need_more"] = decision.need_more
    state["focus"] = (decision.focus or "").strip()

    if (not decision.need_more) and decision.improved_answer.strip():
        state["partial"] = decision.improved_answer.strip()

    return state


def route_after_reviewer(state: MASState) -> str:
    return "gather_tools" if state.get("need_more") else "finalize"


def finalize_node(state: MASState) -> MASState:
    add_node_log(state, "finalize")

    state["final_answer"] = (state["partial"] or "").strip()
    add_history(state, "assistant", state["final_answer"])
    trim_history(state)

    if state["memory_hits"]:
        state["memory_summary"] = "Использованы заметки: " + "; ".join(
            _short(h.get("text", ""), 60) for h in state["memory_hits"][:3])
    else:
        state["memory_summary"] = "Заметки по теме не найдены."

    return state


def route_after_planner(state: MASState) -> str:
    return state.get("intent") or "daily"


def gather_tools_node(state: MASState) -> MASState:
    """
    gather_tools – агент добора информации через инструменты

    Роль в системе:
    - Это узел, который через ReAct tool calling добирает недостающую информацию перед основным ответом
    - Узел используется также как часть цикла улучшения: reviewer может вернуть approved=False + focus, после чего мы снова заходим в gather_tools_node, чтобы добрать контекст
    """
    add_node_log(state, "gather_tools")

    if state["intent"] == "coding":
        tools = TOOLS_CODING
    elif state["intent"] == "daily":
        tools = TOOLS_DAILY
    elif state["intent"] == "literature":
        tools = TOOLS_LITERATURE
    else:
        tools = [search_user_notes, save_user_note]

    llm = get_llm(temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — агент добора информации через инструменты.\n"
         "Твоя задача — добрать недостающие данные, следуя focus.\n"
         "Всегда начинай с поиска в заметках (search_user_notes), если это уместно.\n"
         "Используй инструменты строго по необходимости.\n"
         "После добора кратко резюмируй, что нашёл.\n"),
        ("placeholder", "{messages}")
    ])

    # Испольщуем create_react_agent из примера мультиагентной системы
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    user_msg = (
        # Запрос
        f"QUERY: {state['query']}\n"

        # Тип вопроса от пользователя
        f"INTENT: {state['intent']}\n"

        # План
        f"PLAN: {state['plan']}\n"

        # Какую информацию нужно дособрать 
        f"FOCUS: {state.get('focus', '')}\n"

        # Информация, которая есть в памяти
        f"MEMORY_HITS: {state.get('memory_hits', [])}\n"

        # Какие результаты у интрументов уже были
        f"TOOL_CONTEXT_TAIL: {state.get('tool_context', [])[-5:]}\n"

        # Итеративный процесс
        f"ROUND: {state.get('round', 0)} / {state.get('max_rounds', 3)}\n"
    )

    res = agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"recursion_limit": 30, "configurable": {"thread_id": state["thread_id"]}}
    )

    state.setdefault("tool_calls", [])
    state.setdefault("tool_context", [])

    # Сохраняем результаты инструментов
    for m in res["messages"]:
        if m.__class__.__name__.startswith("ToolMessage"):
            content = getattr(m, "content", "")
            add_tool_log(state, "tool_message", {"content": content})

            state["tool_context"].append({"ts": now_iso(), "tool_message": content})
            state["tool_calls"].append({
                "ts": now_iso(),
                "node": "gather_tools",
                "type": "ToolMessage",
                "payload": {"content": content},
            })

    # Сообщение агента
    summary = _coerce_text(res["messages"][-1])
    state["tool_context"].append({"ts": now_iso(), "gather_summary": summary})

    # Счетчик итераций
    """
    Нужно для остановки цикла
    Была ошибка, что модель зацикливалась, поэтому добавила
    """
    state["round"] = int(state.get("round", 0)) + 1

    return state
