from __future__ import annotations

import os

import anthropic
from pydantic import BaseModel

from .prompts import SYSTEM_PROMPT, build_extraction_prompt

DEFAULT_MODEL = "claude-opus-4-8"


# ---------- Strukturiertes Ausgabeschema ----------


class Experience(BaseModel):
    role: str
    company: str
    period: str
    description: str


class Education(BaseModel):
    degree: str
    institution: str
    period: str


class Language(BaseModel):
    language: str
    level: str


class CVData(BaseModel):
    name: str
    skills: list[str]
    experience: list[Experience]
    education: list[Education]
    certificates: list[str]
    languages: list[Language]


def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY ist nicht gesetzt. Lege ihn in der .env an "
            "(siehe .env.example)."
        )
    return anthropic.Anthropic(api_key=api_key)


def extract_cv(cv_text: str, feedback_block: str = "") -> dict:
    """Ruft die Claude API auf und gibt strukturierte, rein objektive
    CV-Daten als dict zurück.

    Wirft bei Problemen eine RuntimeError mit verständlicher Meldung,
    die in der UI angezeigt werden kann.
    """
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    user_prompt = build_extraction_prompt(cv_text, feedback_block)

    try:
        response = _client().messages.parse(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=CVData,
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Authentifizierung fehlgeschlagen: ungültiger oder fehlender "
            "ANTHROPIC_API_KEY."
        ) from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(
            "Rate-Limit erreicht. Bitte einen Moment warten und erneut versuchen."
        ) from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(
            "Verbindung zur Claude API fehlgeschlagen. Bitte Netzwerk prüfen."
        ) from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"API-Fehler ({e.status_code}): {e.message}") from e
    except anthropic.APIError as e:
        raise RuntimeError(f"Unerwarteter API-Fehler: {e}") from e

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            "Die Antwort konnte nicht in das erwartete Format geparst werden "
            f"(stop_reason={response.stop_reason})."
        )
    return parsed.model_dump()
