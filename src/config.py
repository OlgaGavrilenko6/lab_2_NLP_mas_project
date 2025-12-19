from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Загружаем env
load_dotenv("api_keys.env")

API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("MAS_DEFAULT_MODEL", "gpt-4o-mini")
NOTES_PATH = os.getenv("MAS_NOTES_PATH", "user_notes.json")

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=temperature, api_key=API_KEY)