from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.json"

FEEDBACK_CATEGORIES = [
    "Skill wurde übersehen",
    "Zertifikat wurde übersehen",
    "Sprache falsch erkannt",
    "Sonstiges",
]


def _ensure_file() -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FEEDBACK_PATH.exists():
        FEEDBACK_PATH.write_text("[]", encoding="utf-8")


def load_feedback() -> list[dict]:
    _ensure_file()
    try:
        return json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def add_feedback(candidate_name: str, note: str, category: str = "Sonstiges") -> None:
    """Speichert einen Feedback-Eintrag (z. B. 'SQL wurde übersehen')."""
    note = note.strip()
    if not note:
        return
    entries = load_feedback()
    entries.append(
        {
            "candidate_name": candidate_name.strip() or "(unbekannt)",
            "category": category,
            "note": note,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )
    FEEDBACK_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_feedback_for_prompt(entries: list[dict], max_entries: int = 30) -> str:
    """Formatiert Feedback-Einträge zur Aufnahme in den Extraktionsprompt."""
    if not entries:
        return ""
    recent = entries[-max_entries:]
    lines = [
        f"- ({e['candidate_name']}) [{e.get('category', 'Sonstiges')}] {e['note']}"
        for e in recent
    ]
    return "\n".join(lines)
