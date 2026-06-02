from __future__ import annotations

SYSTEM_PROMPT = """Du bist ein Assistent, der aus Lebenslauf-Texten ausschließlich
objektive, faktische Informationen extrahiert.

WICHTIGE REGELN:
- Extrahiere NUR was wörtlich oder eindeutig im Text steht.
- Keine Bewertungen, kein Ranking, kein Score, keine Empfehlung,
  keine Eignungseinschätzung, keine Interpretation der Persönlichkeit.
- Wenn eine Information fehlt: leeres Array bzw. leerer String.
- Antworte ausschließlich als gültiges JSON gemäß dem vorgegebenen Schema.
- Keine zusätzlichen Felder, kein Fließtext, keine Kommentare.
"""

EXTRACTION_INSTRUCTIONS = """Extrahiere die folgenden Felder aus dem Lebenslauf:

- name (string): Vollständiger Name der Person
- skills (array of string): Konkret im CV genannte fachliche Skills
  (Technologien, Tools, Methoden, Programmiersprachen, Frameworks).
- experience (array of object): Berufserfahrung, je Eintrag:
    - role (string): Position/Titel
    - company (string): Arbeitgeber
    - period (string): Zeitraum, wie im CV angegeben (z. B. "2020-2023")
    - description (string): Kurze, faktische Beschreibung (optional, leer falls nicht vorhanden)
- education (array of object): Ausbildung, je Eintrag:
    - degree (string): Abschluss/Studiengang
    - institution (string): Hochschule/Schule
    - period (string): Zeitraum
- certificates (array of string): Zertifikate / Weiterbildungen mit konkretem Namen
- languages (array of object): Sprachkenntnisse, je Eintrag:
    - language (string): z. B. "Deutsch", "Englisch"
    - level (string): Niveau wie im CV genannt (z. B. "C1", "Muttersprache", "fließend")
"""


def build_extraction_prompt(cv_text: str, feedback_block: str) -> str:
    feedback_section = ""
    if feedback_block:
        feedback_section = (
            "\n\nFRÜHERES NUTZER-FEEDBACK / KORREKTUREN (bitte beachten,\n"
            "damit ähnliche Fehler nicht wiederholt werden):\n"
            f"{feedback_block}\n"
        )

    return (
        f"{EXTRACTION_INSTRUCTIONS}"
        f"{feedback_section}\n"
        "LEBENSLAUF-TEXT:\n"
        "----------------\n"
        f"{cv_text}\n"
        "----------------\n"
        "Gib jetzt das JSON aus."
    )
