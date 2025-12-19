from __future__ import annotations

import datetime
import json
import math
import re

from langchain_core.tools import tool

from .memory_store import load_notes, append_note, simple_retrieve_notes


# Инструменты для агентов
@tool("calc")
def calc(expression: str) -> str:
    """
    Для вычисления простых математических формул 
    """""
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expression or ""):
        return "Ошибка: недопустимые символы в выражении"
    try:
        val = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(val)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


@tool("days_until")
def days_until(date_iso: str) -> str:
    """
    Считаем количество дней до заданной даты для планирования бытовых задач
    """""
    try:
        target = datetime.date.fromisoformat((date_iso or "").strip())
        today = datetime.date.today()
        return str((target - today).days)
    except Exception:
        return "Ошибка: ожидаю дату в формате YYYY-MM-DD"


@tool("save_user_note")
def save_user_note(text: str, tags_json: str = "[]") -> str:
    """
    Сохраняем пользовательский вопрос в файл
    """
    try:
        tags = json.loads(tags_json) if tags_json else []
        if not isinstance(tags, list):
            tags = []
    except Exception:
        tags = []

    # Загружаем текущие ответы, добавляем новую и сохраняем
    notes = load_notes()
    note = append_note(notes, text=text, tags=tags)
    return json.dumps(note, ensure_ascii=False)


@tool("search_user_notes")
def search_user_notes(query: str, k: int = 5) -> str:
    """
    Ищем релевантные ответы в файле с историей
    """
    notes = load_notes()
    hits = simple_retrieve_notes(notes, query=query, k=k)
    return json.dumps(hits, ensure_ascii=False)


# Для бытовых задач
TOOLS_DAILY = [calc, days_until, save_user_note, search_user_notes]

# Для задач программирования
TOOLS_CODING = [calc, save_user_note, search_user_notes]

# Для задач литературных
TOOLS_LITERATURE = [save_user_note, search_user_notes]