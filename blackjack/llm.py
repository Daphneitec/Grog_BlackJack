"""Groq-hosted LangChain chat model used by every agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_llm(*, temperature: float = 0.4) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Put it in a .env file or set the environment variable."
        )
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    common = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        "max_retries": 3,
    }
    try:
        return ChatGroq(**common, reasoning_format="hidden")
    except TypeError:
        return ChatGroq(**common)
