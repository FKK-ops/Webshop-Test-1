"""
HR-Recruiting-Assistent (Streamlit-Prototyp, Single-File).

Lädt mehrere PDF-Lebensläufe, extrahiert per pdfplumber den Text,
ruft die Claude API auf und stellt rein objektive Informationen als
neutrale Tabelle dar. Kein Ranking, kein Score, keine Empfehlung.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # ANTHROPIC_API_KEY eintragen
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

load_dotenv()

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
FEEDBACK_PATH = Path(__file__).resolve().parent / "feedback.json"

DISCLAIMER = (
    "Hinweis: Dieses Tool extrahiert ausschließlich objektive Informationen "
    "aus Lebensläufen. Es liefert **kein Ranking, keinen Score, keine Top-5 "
    "und keine Empfehlung**. Die finale Entscheidung trifft immer ein Mensch."
)

SYSTEM_PROMPT = """Du bist ein Assistent, der aus Lebenslauf-Texten ausschließlich
objektive, faktische Informationen extrahiert.

REGELN:
- Extrahiere NUR, was wörtlich oder eindeutig im Text steht.
- Keine Bewertungen, kein Ranking, kein Score, keine Empfehlung,
  keine Eignungseinschätzung, keine Interpretation der Persönlichkeit.
- Wenn eine Information fehlt: leeres Array bzw. leerer String.
- Antworte ausschließlich im vorgegebenen strukturierten Format.
"""

EXTRACTION_INSTRUCTIONS = """Extrahiere die folgenden Felder aus dem Lebenslauf:

- name: Vollständiger Name
- skills: Konkret genannte fachliche Skills (Technologien, Tools, Methoden,
  Programmiersprachen, Frameworks)
- experience: Berufserfahrung (role, company, period, description)
- education: Ausbildung (degree, institution, period)
- certificates: Zertifikate / Weiterbildungen mit konkretem Namen
- languages: Sprachkenntnisse (language, level)
"""


# ---------------------------------------------------------------------------
# Pydantic-Schema für die strukturierte LLM-Ausgabe
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PDF-Extraktion
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrahiert den gesamten Text aus einer PDF-Datei mit pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


# ---------------------------------------------------------------------------
# Feedback-Persistenz
# ---------------------------------------------------------------------------


def load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    try:
        return json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def add_feedback(candidate_name: str, note: str) -> None:
    note = note.strip()
    if not note:
        return
    entries = load_feedback()
    entries.append(
        {
            "candidate_name": candidate_name.strip() or "(unbekannt)",
            "note": note,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )
    FEEDBACK_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def feedback_block(entries: list[dict], max_entries: int = 30) -> str:
    if not entries:
        return ""
    recent = entries[-max_entries:]
    return "\n".join(f"- ({e['candidate_name']}) {e['note']}" for e in recent)


# ---------------------------------------------------------------------------
# Claude-API-Aufruf
# ---------------------------------------------------------------------------


def extract_cv(cv_text: str, prior_feedback: str = "") -> dict:
    """Ruft die Claude API auf und liefert objektive CV-Felder als dict.

    Bei Problemen wird eine RuntimeError mit verständlicher Botschaft erhoben,
    sodass die UI sie anzeigen kann.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY ist nicht gesetzt. Trage ihn in der .env ein."
        )

    feedback_section = ""
    if prior_feedback:
        feedback_section = (
            "\n\nFRÜHERES NUTZER-FEEDBACK (bitte beachten, damit ähnliche "
            "Fehler nicht wiederholt werden):\n"
            f"{prior_feedback}\n"
        )

    user_prompt = (
        f"{EXTRACTION_INSTRUCTIONS}{feedback_section}\n"
        "LEBENSLAUF-TEXT:\n"
        "----------------\n"
        f"{cv_text}\n"
        "----------------\n"
        "Gib jetzt die strukturierte Ausgabe zurück."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=CVData,
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Authentifizierung fehlgeschlagen: ungültiger ANTHROPIC_API_KEY."
        ) from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(
            "Rate-Limit erreicht. Bitte kurz warten und erneut versuchen."
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
            "Die Antwort konnte nicht in das erwartete Format geparst werden."
        )
    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die UI
# ---------------------------------------------------------------------------


def candidates_to_dataframe(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        d = c["data"]
        rows.append(
            {
                "Datei": c["filename"],
                "Name": d.get("name", ""),
                "Skills": ", ".join(d.get("skills", [])),
                "Berufserfahrung": " | ".join(
                    f"{e.get('role', '')} @ {e.get('company', '')} "
                    f"({e.get('period', '')})".strip()
                    for e in d.get("experience", [])
                ),
                "Ausbildung": " | ".join(
                    f"{e.get('degree', '')}, {e.get('institution', '')} "
                    f"({e.get('period', '')})".strip()
                    for e in d.get("education", [])
                ),
                "Zertifikate": ", ".join(d.get("certificates", [])),
                "Sprachen": ", ".join(
                    f"{l.get('language', '')} ({l.get('level', '')})".strip()
                    for l in d.get("languages", [])
                ),
            }
        )
    return pd.DataFrame(rows)


def matches_all(haystack: str, query: str) -> bool:
    """Case-insensitive UND-Match aller per Komma getrennten Suchbegriffe."""
    terms = [t.strip().lower() for t in query.split(",") if t.strip()]
    if not terms:
        return True
    h = haystack.lower()
    return all(t in h for t in terms)


def filter_candidates(
    candidates: list[dict], skills_q: str, languages_q: str, certs_q: str
) -> list[dict]:
    result = []
    for c in candidates:
        d = c["data"]
        skills_str = " ".join(d.get("skills", []))
        langs_str = " ".join(
            f"{l.get('language', '')} {l.get('level', '')}"
            for l in d.get("languages", [])
        )
        certs_str = " ".join(d.get("certificates", []))

        if not matches_all(skills_str, skills_q):
            continue
        if not matches_all(langs_str, languages_q):
            continue
        if not matches_all(certs_str, certs_q):
            continue
        result.append(c)
    return result


# ---------------------------------------------------------------------------
# Streamlit-UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HR-Recruiting-Assistent", layout="wide")

st.title("HR-Recruiting-Assistent für KMU")
st.write(
    "Lade PDF-Lebensläufe hoch. Der Assistent extrahiert ausschließlich "
    "objektive Informationen — Name, Skills, Berufserfahrung, Ausbildung, "
    "Zertifikate und Sprachen — und zeigt sie als neutrale Tabelle. "
    "Es findet **keine Bewertung, kein Ranking und keine Empfehlung** statt."
)
st.info(DISCLAIMER)

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ---- Sidebar: Upload ----

with st.sidebar:
    st.header("Lebensläufe hochladen")
    uploaded = st.file_uploader(
        "PDF-Dateien auswählen",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Extraktion starten", type="primary", disabled=not uploaded):
        prior_feedback = feedback_block(load_feedback())
        new_files = [
            f for f in uploaded if f.name not in st.session_state.processed_files
        ]
        if not new_files:
            st.warning("Alle ausgewählten Dateien wurden bereits verarbeitet.")
        else:
            progress = st.progress(0.0)
            ok = 0
            for i, f in enumerate(new_files, start=1):
                with st.spinner(f"Verarbeite {f.name} ..."):
                    try:
                        text = extract_text_from_pdf(f.read())
                        if not text:
                            st.warning(
                                f"{f.name}: kein Text extrahierbar "
                                "(evtl. gescanntes PDF ohne OCR)."
                            )
                            continue
                        data = extract_cv(text, prior_feedback)
                        st.session_state.candidates.append(
                            {"filename": f.name, "data": data}
                        )
                        st.session_state.processed_files.add(f.name)
                        ok += 1
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Fehler bei {f.name}: {e}")
                progress.progress(i / len(new_files))
            if ok:
                st.success(f"{ok} Lebensläufe extrahiert.")

    if st.session_state.candidates and st.button("Alle Kandidaten löschen"):
        st.session_state.candidates = []
        st.session_state.processed_files = set()
        st.rerun()

# ---- Hauptbereich ----

if not st.session_state.candidates:
    st.write("Lade links Lebensläufe hoch, um zu starten.")
    st.stop()

st.subheader("Filter")
col1, col2, col3 = st.columns(3)
with col1:
    skills_q = st.text_input("Skills (Komma-getrennt)", "")
with col2:
    languages_q = st.text_input("Sprachen", "")
with col3:
    certs_q = st.text_input("Zertifikate", "")

filtered = filter_candidates(
    st.session_state.candidates, skills_q, languages_q, certs_q
)

st.subheader(f"Bewerber ({len(filtered)} von {len(st.session_state.candidates)})")
df = candidates_to_dataframe(filtered)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Details + Feedback ----

st.subheader("Details & Feedback")
options = [
    f"{c['data'].get('name') or '(ohne Name)'} – {c['filename']}" for c in filtered
]
if not options:
    st.write("Keine Bewerber entsprechen den Filtern.")
else:
    idx = st.selectbox(
        "Bewerber auswählen", range(len(options)), format_func=lambda i: options[i]
    )
    selected = filtered[idx]
    st.json(selected["data"])

    with st.form("feedback_form", clear_on_submit=True):
        st.markdown(
            "**Feedback / Korrektur eintragen** "
            "(z. B. „SQL wurde übersehen“, „Sprache Französisch falsch erkannt“). "
            "Das Feedback wird in `feedback.json` gespeichert und beim nächsten "
            "Extraktionslauf in den Prompt einbezogen."
        )
        note = st.text_area("Anmerkung")
        submitted = st.form_submit_button("Feedback speichern")
        if submitted:
            if not note.strip():
                st.warning("Bitte eine Anmerkung eingeben.")
            else:
                add_feedback(selected["data"].get("name", ""), note)
                st.success("Feedback gespeichert.")

with st.expander("Bisheriges Feedback"):
    entries = load_feedback()
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.write("Noch kein Feedback vorhanden.")

st.caption(DISCLAIMER)
