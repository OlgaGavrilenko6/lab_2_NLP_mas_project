from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

"""
Передача состояний между узлами графа
Типы запросов пользовтеля: 
- conceptual – 
- architecture – вопрос архитектуры
- coding – вопрос программирования
- daily – повседеневный вопрос
- literature – обзор литературы 
"""

Intent = Literal["conceptual", "architecture", "coding", "daily", "literature"]

class MASState(TypedDict):
    query: str                           # Запрос пользователя
    intent: Optional[Intent]             # Тип запроса пользователя
    plan: List[str]                      # План решение от планировщика
    tool_context: List[Dict[str, Any]]   # Накопленная информация от интрументов
    focus: str                           # Какую информацию еще необходимо собрать
    need_more: bool                      # Флаг для понимания нужно ли делать новый запрос интрументу
    round: int                           # Итерация цикла
    max_rounds: int                      # Максимальное количество итераций цикла
    partial: str                         # Промежуточный ответ агента
    final_answer: str                    # Финальный ответ агента
    history: List[Dict[str, Any]]        # История диалога
    memory_notes: List[Dict[str, Any]]   # История пользователя из файла
    memory_hits: List[Dict[str, Any]]    # Результаты поиска по истории из файла
    memory_summary: str                  # Резюме об использовании памяти
    activated_nodes: List[str]           # Список узлов
    tool_calls: List[Dict[str, Any]]     # Лог вызова инструментов
    handoff_log: List[str]               # Передача информации между агентами
    thread_id: str                       # id сессии
    verbose: bool

