from __future__ import annotations

import datetime
import json
import re
from typing import Any, Dict, Optional


"""
Блок со вспомогательными функциями 
Данные функции используются для единообразной трассировки и стабильной сериализации данных в узалах графа
"""

# Возвращаем текущую дату и время для логов
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

# Приводит в единообразный ответ от модели и инструментов
def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if hasattr(x, "content"):
        return x.content
    return str(x)

# Подстраховка для PydanticOutputParser, чтоб точно получили ответ модели в необходимом формате
def _extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# Уменьшаем размер строки для вывода ответов в логах
def _short(s: str, n: int = 240) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"

# Храним историю вопросов и ответов (заданиче управления с памятью в лабораторной работе)
def add_history(state: "MASState", role: str, content: str):
    state["history"].append({"ts": now_iso(), "role": role, "content": content})

# Логируем факт вызова инструмента и его результат
def add_tool_log(state: "MASState", tool_name: str, payload: Any):
    state["tool_calls"].append({"ts": now_iso(), "tool": tool_name, "payload": payload})

# Логируем посещение узла графа LangGraph
def add_node_log(state: "MASState", node_name: str):
    state["activated_nodes"].append(node_name)

# Обрезаем историю диалога, оставляя только последние записи
def trim_history(state: "MASState", keep_last: int = 10):
    if len(state["history"]) > keep_last:
        state["history"] = state["history"][-keep_last:]