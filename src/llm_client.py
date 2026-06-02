from __future__ import annotations

import json
import os

from openai import OpenAI

from .prompts import SYSTEM_PROMPT, build_extraction_prompt

JSON_SCHEMA = {
    "name": "cv_extraction",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {"type": "string"},
                        "company": {"type": "string"},
                        "period": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["role", "company", "period", "description"],
                },
            },
            "education": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "degree": {"type": "string"},
                        "institution": {"type": "string"},
                        "period": {"type": "string"},
                    },
                    "required": ["degree", "institution", "period"],
                },
            },
            "certificates": {"type": "array", "items": {"type": "string"}},
            "languages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "language": {"type": "string"},
                        "level": {"type": "string"},
                    },
                    "required": ["language", "level"],
                },
            },
        },
        "required": [
            "name",
            "skills",
            "experience",
            "education",
            "certificates",
            "languages",
        ],
    },
    "strict": True,
}


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY ist nicht gesetzt. Lege ihn in der .env an "
            "(siehe .env.example)."
        )
    return OpenAI(api_key=api_key)


def extract_cv(cv_text: str, feedback_block: str = "") -> dict:
    """Ruft die LLM-API auf und gibt die strukturierten CV-Daten als dict zurück."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    user_prompt = build_extraction_prompt(cv_text, feedback_block)

    response = _client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)
