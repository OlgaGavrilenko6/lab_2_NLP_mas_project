from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import NOTES_PATH
from .utils import now_iso


"""
Хранилище памяти
Сохраняем историю на диск в json-файл и извлекаем ответы по запросу
"""

# Загружаем историю из json-файла
def load_notes() -> List[Dict[str, Any]]:
    if not os.path.exists(NOTES_PATH):
        return []
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

# Сохраняем историю в json-файл
def save_notes(notes: List[Dict[str, Any]]) -> None:
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

# Добавляем в память ответ и сохраняем в файл
def append_note(notes: List[Dict[str, Any]], text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    note = {"ts": now_iso(), "text": (text or "").strip(), "tags": tags or []}
    notes.append(note)
    save_notes(notes)
    return note

# Работа с памятью
def simple_retrieve_notes(notes: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Алгоритм:
    1) Берем запрос и приводим к нижнему регистру, извлекаем токены длиной >= 3
    2) Для каждой заметки считаем "score" = сколько токенов запроса встречается в тексте ответа или в тегах
    3) Возвращаем топ ответов с максимальной оценкой

    Параметры:
    - notes: список заметок (list[dict])
    - query: запрос от пользователя или агента
    - k: сколько ответов нужно вернуть

    Возвращает:
    - список заметок (dict), наиболее релевантных запросу
    """
    q = (query or "").lower()
    tokens = set(re.findall(r"[a-zа-я0-9]{3,}", q))
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for n in notes:
        text = (n.get("text", "") + " " + " ".join(n.get("tags", []))).lower()
        score = sum(1 for t in tokens if t in text)
        if score > 0:
            scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:k]]