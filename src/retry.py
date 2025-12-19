from __future__ import annotations

from typing import Any, List, Optional, Tuple

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from .utils import _coerce_text, _extract_json


# PydanticOutputParser
def invoke_with_parser_retry(
        *,
        make_llm,
        messages: List[BaseMessage],
        parser: PydanticOutputParser,
        max_retries: int = 3,
        temps: Tuple[float, ...] = (0.1, 0.2, 0.3),
) -> Any:
    """
    - make_llm: функция вида make_llm(temp) -> LLM. Нужна, чтобы удобно менять температуру на каждой попытке
    - messages: список сообщений (SystemMessage), которые передаются модели
    - parser: PydanticOutputParser, который знает целевую схему ответа
    - max_retries: максимальное число попыток (ограничиваем до 3х по заданию)
    - temps: набор температур по попыткам (увеличиваем на 0,1)
    """
    last_err: Optional[Exception] = None

    n = min(max_retries, len(temps))
    for i in range(n):
        llm = make_llm(temps[i])

        extra = ""
        if i > 0:
            extra = (
                "\nВАЖНО: верни ТОЛЬКО JSON без пояснений, без markdown, без лишнего текста.\n"
                "Даже если не уверен — верни валидный JSON по схеме.\n"
            )

        patched = list(messages)
        if patched and isinstance(patched[0], SystemMessage):
            patched[0] = SystemMessage(content=patched[0].content + extra)
        else:
            patched = [SystemMessage(content=extra)] + patched

        try:
            raw = llm.invoke(patched)
            text = _coerce_text(raw)
            return parser.parse(text)
        except Exception as e:
            last_err = e

    # Извлекаем jsonиз ответа
    raw = make_llm(temps[-1]).invoke(messages)
    text = _coerce_text(raw)
    data = _extract_json(text)
    if data is not None:
        return parser.pydantic_object.model_validate(data)

    raise last_err or ValueError("Не удалось проанализировать ответ модели")