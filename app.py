"""
HR-Recruiting-Assistent (Streamlit-Prototyp, Single-File).

Lädt mehrere PDF-Lebensläufe, extrahiert per pdfplumber den Text und ruft
die Claude API auf, um daraus rein objektive Informationen zu strukturieren.
Optional kann ein Stellenprofil eingegeben werden — der Agent strukturiert
es nur, er bewertet nichts und gewichtet nichts.

Grundregel:
- Der Agent trifft keine Personalentscheidung.
- Der Agent erstellt keine Empfehlung.
- Der Agent extrahiert objektive Informationen und macht Bewerbungen
  vergleichbar. Die finale Bewertung trifft immer der Geschäftsführer.

Autonomie-Stufe: Rot — Mensch entscheidet, Agent liefert nur Daten.

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

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_PATH = BASE_DIR / "feedback.json"
AUDIT_LOG_PATH = BASE_DIR / "audit_log.json"
TEST_RESULTS_PATH = BASE_DIR / "test_results.json"

AUTONOMY_LEVEL = "Rot: Mensch entscheidet, Agent liefert nur Daten"

DISCLAIMER = (
    "Der Agent trifft **keine Personalentscheidung**. "
    "Der Agent erstellt **keine Empfehlung**. "
    "Er extrahiert objektive Informationen und macht Bewerbungen vergleichbar. "
    "Die finale Bewertung trifft immer der Geschäftsführer."
)

FEEDBACK_CATEGORIES = [
    "Skill übersehen",
    "Zertifikat übersehen",
    "Sprache falsch erkannt",
    "Ausbildung falsch erkannt",
    "Berufserfahrung falsch erkannt",
    "Sonstiges",
]

SYSTEM_PROMPT = """Du bist ein Assistent, der aus Lebenslauf-Texten und
Stellenprofilen ausschließlich objektive, faktische Informationen extrahiert
und strukturiert.

REGELN:
- Extrahiere NUR, was wörtlich oder eindeutig im Text steht.
- Keine Bewertungen, kein Ranking, kein Score, keine Empfehlung,
  keine Eignungseinschätzung, keine Personalentscheidung.
- Wenn eine Information fehlt: leeres Array bzw. leerer String.
- Antworte ausschließlich im vorgegebenen strukturierten Format.
"""

EXTRACTION_INSTRUCTIONS = """Extrahiere die folgenden Felder aus dem
Lebenslauf:

- name: Vollständiger Name
- skills: Konkret genannte fachliche Skills (Technologien, Tools, Methoden,
  Programmiersprachen, Frameworks)
- experience: Berufserfahrung (role, company, period, description)
- education: Ausbildung (degree, institution, period)
- certificates: Zertifikate / Weiterbildungen mit konkretem Namen
- languages: Sprachkenntnisse (language, level)
- evidence: Pro erkannter Qualifikation (Skill, Sprache, Zertifikat, wichtige
  Berufserfahrung) ein kurzer wörtlicher Textausschnitt aus dem Lebenslauf
  als Beleg. Format pro Eintrag: {item, excerpt}. Wenn kein eindeutiger
  Beleg möglich ist, lasse den jeweiligen Eintrag hier aus.
- data_quality_issues: Hinweise auf unklare oder fehlende Informationen,
  z. B. "Sprachlevel nicht eindeutig angegeben",
  "Berufserfahrung nur grob beschrieben",
  "Zertifikat erwähnt, aber Name unklar",
  "PDF möglicherweise unvollständig ausgelesen".
  Das ist keine Bewertung des Bewerbers, sondern ein Datenqualitäts-Hinweis.
"""

QUALITY_INSTRUCTIONS = """Du führst eine Datenqualitäts-Prüfung der
Bewerbungsunterlage durch. Du bewertest NICHT die Eignung des Bewerbers,
du gibst keine Empfehlung, du priorisierst nicht und du vergibst keinen
Score.

Du prüfst ausschließlich, ob die Unterlage vollständig und eindeutig ist:

- missing_information: Felder oder Angaben, die im Lebenslauf gar nicht
  vorkommen. Beispiele:
    "Sprachkenntnisse nicht angegeben"
    "Zertifikate nicht angegeben"
    "Berufserfahrung enthält keine Zeiträume"
- unclear_information: Angaben, die zwar vorhanden, aber unklar,
  unvollständig oder mehrdeutig sind. Beispiele:
    "Dauer der Berufserfahrung unklar"
    "Sprachlevel nicht eindeutig angegeben"
    "Zertifikat erwähnt, aber Name fehlt"
    "PDF möglicherweise unvollständig ausgelesen"
- suggested_questions: konkrete Rückfragen an den Bewerber, die helfen
  würden, fehlende oder unklare Informationen zu klären. Beispiele:
    "Bitte nennen Sie den genauen Zeitraum Ihrer Tätigkeit bei XY."
    "Welches Sprachniveau (z. B. nach GER) haben Sie in Englisch?"

Wenn alles klar und vollständig ist, lasse die jeweiligen Arrays leer.
Keine Bewertung des Bewerbers. Keine Empfehlung. Kein Ranking.
"""

JOB_PROFILE_INSTRUCTIONS = """Strukturiere das folgende Stellenprofil.
Du bewertest nichts und gewichtest nichts — du strukturierst nur, was im
Text steht.

Extrahiere:
- must_criteria: Muss-Kriterien (Pflichtanforderungen)
- nice_criteria: Kann-Kriterien ("wünschenswert", "von Vorteil")
- desired_skills: gewünschte fachliche Skills
- desired_languages: gewünschte Sprachen (inkl. Niveau, wenn genannt)
- desired_certificates: gewünschte Zertifikate
- desired_experience: gewünschte Berufserfahrung (Branchen, Rollen, Jahre)

Wenn etwas nicht im Text steht, lasse das jeweilige Array leer.
"""


# ---------------------------------------------------------------------------
# Pydantic-Schemas für die strukturierte LLM-Ausgabe
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


class EvidenceItem(BaseModel):
    item: str
    excerpt: str


class CVData(BaseModel):
    name: str
    skills: list[str]
    experience: list[Experience]
    education: list[Education]
    certificates: list[str]
    languages: list[Language]
    evidence: list[EvidenceItem]
    data_quality_issues: list[str]


class JobProfile(BaseModel):
    must_criteria: list[str]
    nice_criteria: list[str]
    desired_skills: list[str]
    desired_languages: list[str]
    desired_certificates: list[str]
    desired_experience: list[str]


class QualityCheck(BaseModel):
    missing_information: list[str]
    unclear_information: list[str]
    suggested_questions: list[str]


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
# JSON-Hilfsfunktionen für die kleinen Persistenz-Dateien
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_json_list(path: Path, entries: list[dict]) -> None:
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Feedback-Persistenz (mit Kategorie)
# ---------------------------------------------------------------------------


def load_feedback() -> list[dict]:
    return _read_json_list(FEEDBACK_PATH)


def add_feedback(candidate_name: str, category: str, note: str) -> None:
    note = note.strip()
    if not note:
        return
    entries = load_feedback()
    entries.append(
        {
            "candidate_name": candidate_name.strip() or "(unbekannt)",
            "category": category,
            "note": note,
            "created_at": _now_iso(),
        }
    )
    _write_json_list(FEEDBACK_PATH, entries)


def feedback_block(entries: list[dict], max_entries: int = 30) -> str:
    """Aufbereitung für den Prompt — verbessert nur die Extraktion."""
    if not entries:
        return ""
    recent = entries[-max_entries:]
    lines = [
        f"- ({e['candidate_name']}) [{e.get('category', 'Sonstiges')}] {e['note']}"
        for e in recent
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit-Log
# ---------------------------------------------------------------------------


def load_audit_log() -> list[dict]:
    return _read_json_list(AUDIT_LOG_PATH)


def log_audit(
    action: str,
    target: str = "",
    result_type: str = "",
    autonomy_level: str = AUTONOMY_LEVEL,
) -> None:
    """Hängt einen Audit-Log-Eintrag an audit_log.json an."""
    entries = load_audit_log()
    entries.append(
        {
            "timestamp": _now_iso(),
            "action": action,
            "target": target,
            "result_type": result_type,
            "autonomy_level": autonomy_level,
        }
    )
    _write_json_list(AUDIT_LOG_PATH, entries)


# ---------------------------------------------------------------------------
# Test-Center
# ---------------------------------------------------------------------------


TEST_CASES = [
    {
        "id": "happy_path",
        "name": "Happy Path",
        "description": "Normaler, gut lesbarer Lebenslauf mit vollständigen Angaben.",
    },
    {
        "id": "edge_case",
        "name": "Edge Case",
        "description": "Lebenslauf mit fehlenden oder unklaren Angaben.",
    },
    {
        "id": "failure_mode",
        "name": "Failure Mode",
        "description": "Gescanntes oder nicht maschinenlesbares PDF.",
    },
]

TEST_OUTCOMES = ["bestanden", "teilweise bestanden", "fehlgeschlagen"]


def load_test_results() -> list[dict]:
    return _read_json_list(TEST_RESULTS_PATH)


def save_test_result(test_id: str, outcome: str, note: str = "") -> None:
    entries = load_test_results()
    entries.append(
        {
            "test_id": test_id,
            "outcome": outcome,
            "note": note.strip(),
            "created_at": _now_iso(),
        }
    )
    _write_json_list(TEST_RESULTS_PATH, entries)


# ---------------------------------------------------------------------------
# Claude-API-Aufrufe
# ---------------------------------------------------------------------------


def _anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY ist nicht gesetzt. Trage ihn in der .env ein."
        )
    return anthropic.Anthropic(api_key=api_key)


def _call_parse(user_prompt: str, output_format):
    """Gemeinsamer Wrapper für messages.parse() mit verständlichen Fehlern."""
    try:
        client = _anthropic_client()
        return client.messages.parse(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=output_format,
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


def extract_cv(cv_text: str, prior_feedback: str = "") -> dict:
    """Strukturiert objektive CV-Daten inkl. Belegen und Datenqualitäts-Hinweisen."""
    feedback_section = ""
    if prior_feedback:
        feedback_section = (
            "\n\nFRÜHERES NUTZER-FEEDBACK (bitte berücksichtigen, damit "
            "ähnliche Extraktions-Fehler nicht wiederholt werden — Feedback "
            "verbessert nur die Extraktion und Darstellung, nicht die "
            "Bewerberauswahl):\n"
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

    response = _call_parse(user_prompt, CVData)
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            "Die Antwort konnte nicht in das erwartete Format geparst werden."
        )
    return parsed.model_dump()


def extract_job_profile(job_text: str) -> dict:
    """Strukturiert ein Stellenprofil — ohne Bewertung, ohne Gewichtung."""
    user_prompt = (
        f"{JOB_PROFILE_INSTRUCTIONS}\n"
        "STELLENPROFIL:\n"
        "--------------\n"
        f"{job_text}\n"
        "--------------\n"
        "Gib jetzt die strukturierte Ausgabe zurück."
    )

    response = _call_parse(user_prompt, JobProfile)
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            "Das Stellenprofil konnte nicht in das erwartete Format geparst werden."
        )
    return parsed.model_dump()


def analyze_data_quality(cv_text: str, cv_data: dict) -> dict:
    """Vollständigkeits- und Klarheitsprüfung der Bewerbungsunterlage.

    Liefert {"missing_information", "unclear_information",
    "suggested_questions"}. Keine Eignungsbewertung — nur Datenqualität.
    """
    user_prompt = (
        f"{QUALITY_INSTRUCTIONS}\n"
        "STRUKTURIERTE CV-DATEN (aus der Extraktion):\n"
        "--------------------------------------------\n"
        f"{json.dumps(cv_data, ensure_ascii=False, indent=2)}\n"
        "--------------------------------------------\n\n"
        "ORIGINAL-LEBENSLAUF-TEXT:\n"
        "-------------------------\n"
        f"{cv_text}\n"
        "-------------------------\n"
        "Gib jetzt die strukturierte Datenqualitäts-Prüfung zurück."
    )

    response = _call_parse(user_prompt, QualityCheck)
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            "Die Datenqualitäts-Prüfung konnte nicht geparst werden."
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


def _searchable_text(cv_data: dict) -> str:
    """Volltext aus CV-Feldern für Stichwortsuche."""
    parts: list[str] = list(cv_data.get("skills", []))
    parts.extend(cv_data.get("certificates", []))
    for lang in cv_data.get("languages", []):
        parts.append(f"{lang.get('language', '')} {lang.get('level', '')}")
    for exp in cv_data.get("experience", []):
        parts.append(
            f"{exp.get('role', '')} {exp.get('company', '')} "
            f"{exp.get('description', '')}"
        )
    return " | ".join(parts)


def _evidence_for(query: str, cv_data: dict) -> str | None:
    """Sucht in der evidence-Liste nach einem passenden Beleg."""
    q = query.lower()
    for ev in cv_data.get("evidence", []):
        item = (ev.get("item") or "").lower()
        if not item:
            continue
        if q in item or item in q:
            return ev.get("excerpt") or None
    return None


def check_criterion(criterion: str, cv_data: dict) -> dict:
    """Prüft neutral, ob ein Kriterium im CV vorkommt.

    Liefert {"status": "vorhanden" | "unklar" | "nicht_gefunden",
             "evidence": str | None, "note": str | None}.

    Keine Bewertung, keine Gewichtung, keine Prozentwerte — nur drei
    sachliche Status mit optionalem Beleg und optionalem Hinweis.
    """
    crit = criterion.strip()
    if not crit:
        return {"status": "nicht_gefunden", "evidence": None, "note": None}

    haystack = _searchable_text(cv_data).lower()
    crit_low = crit.lower()

    # Voller Treffer → vorhanden
    if crit_low in haystack:
        return {
            "status": "vorhanden",
            "evidence": _evidence_for(crit_low, cv_data),
            "note": None,
        }

    # Token-basierter Teil-Match → unklar
    tokens = [t for t in crit_low.split() if len(t) >= 2]
    matched_tokens = [t for t in tokens if t in haystack]
    if tokens and matched_tokens and len(matched_tokens) < len(tokens):
        missing = [t for t in tokens if t not in matched_tokens]
        note = (
            f"Teiltreffer gefunden ({', '.join(matched_tokens)}); "
            f"nicht eindeutig erkennbar: {', '.join(missing)}."
        )
        return {
            "status": "unklar",
            "evidence": _evidence_for(matched_tokens[0], cv_data),
            "note": note,
        }

    # Datenqualitäts-Hinweis erwähnt das Kriterium → unklar
    issues_blob = " ".join(cv_data.get("data_quality_issues", [])).lower()
    for token in tokens or [crit_low]:
        if token and token in issues_blob:
            return {
                "status": "unklar",
                "evidence": None,
                "note": "Datenqualitäts-Hinweis betrifft dieses Kriterium.",
            }

    return {"status": "nicht_gefunden", "evidence": None, "note": None}


def render_checklist(job: dict, cv_data: dict) -> None:
    """Zeigt eine neutrale Kriterien-Checkliste pro Bewerber.

    Keine Prozentzahl, kein Score, kein Ranking, keine Priorisierung.
    """
    groups = [
        ("Muss-Kriterien", job.get("must_criteria", [])),
        ("Kann-Kriterien", job.get("nice_criteria", [])),
        ("Gewünschte Skills", job.get("desired_skills", [])),
        ("Gewünschte Sprachen", job.get("desired_languages", [])),
        ("Gewünschte Zertifikate", job.get("desired_certificates", [])),
        ("Gewünschte Berufserfahrung", job.get("desired_experience", [])),
    ]
    any_rendered = False
    for title, items in groups:
        if not items:
            continue
        any_rendered = True
        st.markdown(f"**{title}**")
        for c in items:
            res = check_criterion(c, cv_data)
            status = res["status"]
            if status == "vorhanden":
                evidence = res["evidence"] or "Kein eindeutiger Beleg gefunden"
                st.markdown(
                    f"- **{c}** — vorhanden  \n  *Beleg:* „{evidence}“"
                )
            elif status == "unklar":
                lines = [f"- **{c}** — unklar"]
                if res.get("note"):
                    lines.append(f"  *Hinweis:* {res['note']}")
                if res.get("evidence"):
                    lines.append(f"  *Teilbeleg:* „{res['evidence']}“")
                st.markdown("  \n".join(lines))
            else:
                st.markdown(f"- **{c}** — nicht gefunden")
    if not any_rendered:
        st.write("Stellenprofil enthält keine prüfbaren Kriterien.")


# ---------------------------------------------------------------------------
# Streamlit-UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="KMU Recruiting Agent",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Dashboard Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #07172A;
    }
    [data-testid="stHeader"] { background: transparent; }
    .stApp { color: #F8FAFC; }
    [data-testid="stSidebar"] {
        background: #0B1220 !important;
        border-right: 1px solid #1E3A5F;
    }
    [data-testid="stSidebar"] * { color: #F8FAFC; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea,
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: #10233A !important;
        color: #F8FAFC !important;
        border: 1px solid #1E3A5F !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: #10233A !important;
        border: 1px dashed #1E3A5F !important;
        color: #94A3B8 !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; }
    [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #14B8A6 !important;
        color: #07172A !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    .stButton > button:hover { background: #0F9181 !important; }
    .stButton > button:disabled {
        background: #1E3A5F !important;
        color: #94A3B8 !important;
    }
    [data-testid="stTabs"] [role="tablist"] {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 6px;
        gap: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        color: #94A3B8 !important;
        background: transparent !important;
        border-radius: 6px !important;
        padding: 8px 14px !important;
        font-weight: 500 !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: #0F9181 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stExpander"] {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary { color: #F8FAFC !important; }
    [data-testid="stDataFrame"] {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 8px;
    }
    /* === Custom Dashboard Components === */
    .kmu-card {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .kmu-card-title {
        color: #F8FAFC;
        font-weight: 600;
        font-size: 17px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kmu-card-title small { color: #94A3B8; font-weight: 400; font-size: 13px; }
    .kpi-card {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
        min-height: 150px;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .kpi-card.red::before { background: #EF4444; }
    .kpi-card.orange::before { background: #F59E0B; }
    .kpi-card.teal::before { background: #14B8A6; }
    .kpi-value {
        font-size: 38px;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 6px;
    }
    .kpi-value.red { color: #EF4444; }
    .kpi-value.orange { color: #F59E0B; }
    .kpi-value.teal { color: #14B8A6; }
    .kpi-label {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 12px;
        line-height: 1.4;
    }
    .kmu-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 12px 22px;
        margin-bottom: 22px;
    }
    .kmu-topbar-left { font-size: 22px; color: #94A3B8; }
    .kmu-topbar-right {
        display: flex;
        align-items: center;
        gap: 18px;
        color: #94A3B8;
    }
    .kmu-topbar-icon {
        position: relative;
        font-size: 18px;
        cursor: default;
    }
    .kmu-topbar-icon .badge {
        position: absolute;
        top: -8px; right: -8px;
        background: #EF4444;
        color: white;
        border-radius: 50%;
        width: 16px; height: 16px;
        font-size: 10px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
    }
    .kmu-avatar {
        background: #14B8A6;
        color: #07172A;
        width: 32px; height: 32px;
        border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700;
        font-size: 13px;
    }
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 20px;
    }
    .dashboard-header h1 {
        font-size: 30px !important;
        margin: 0 !important;
        font-weight: 700;
    }
    .dashboard-header .subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 4px;
    }
    .kmu-quote {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 22px;
        display: flex;
        gap: 18px;
        align-items: flex-start;
    }
    .kmu-quote-mark {
        color: #14B8A6;
        font-size: 44px;
        line-height: 0.8;
        font-family: Georgia, serif;
    }
    .kmu-quote-body { flex: 1; }
    .kmu-quote-text {
        font-style: italic;
        color: #F8FAFC;
        font-size: 16px;
        line-height: 1.5;
    }
    .kmu-quote-source {
        color: #14B8A6;
        font-size: 13px;
        margin-top: 14px;
    }
    .kmu-disclaimer {
        background: rgba(20, 184, 166, 0.08);
        border: 1px solid #14B8A6;
        border-left: 4px solid #14B8A6;
        border-radius: 8px;
        padding: 12px 18px;
        color: #94A3B8;
        font-size: 13px;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .kmu-disclaimer strong { color: #F8FAFC; }
    .kmu-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .kmu-badge-ok { background: rgba(16, 185, 129, 0.16); color: #10B981; }
    .kmu-badge-warn { background: rgba(245, 158, 11, 0.16); color: #F59E0B; }
    .kmu-badge-err { background: rgba(239, 68, 68, 0.16); color: #EF4444; }
    .kmu-list-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #1E3A5F;
        gap: 12px;
    }
    .kmu-list-item:last-child { border-bottom: none; }
    .kmu-list-avatar {
        width: 36px; height: 36px;
        border-radius: 50%;
        background: #0F9181;
        color: #07172A;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: 700;
        font-size: 13px;
        flex-shrink: 0;
    }
    .kmu-list-name { font-weight: 600; color: #F8FAFC; font-size: 14px; }
    .kmu-list-file { color: #94A3B8; font-size: 12px; margin-top: 2px; }
    .kmu-logo {
        font-size: 17px;
        font-weight: 700;
        color: #14B8A6;
        padding: 4px 8px 18px 8px;
    }
    .kmu-nav-section {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 16px 8px 6px 8px;
    }
    .kmu-nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 12px;
        border-radius: 8px;
        color: #94A3B8;
        font-size: 14px;
        margin: 2px 4px;
    }
    .kmu-nav-item.active {
        background: #0F9181;
        color: #F8FAFC;
        font-weight: 600;
    }
    .kmu-nav-item .badge-num {
        margin-left: auto;
        background: #14B8A6;
        color: #07172A;
        padding: 1px 7px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
    }
    .donut-wrap {
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .donut-legend { font-size: 13px; flex: 1; }
    .donut-legend-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        color: #F8FAFC;
    }
    .donut-dot {
        width: 12px; height: 12px;
        border-radius: 3px;
        flex-shrink: 0;
    }
    .donut-count {
        margin-left: auto;
        color: #94A3B8;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "job_profile" not in st.session_state:
    st.session_state.job_profile = None
if "job_profile_source" not in st.session_state:
    st.session_state.job_profile_source = ""


# ---------------------------------------------------------------------------
# Dashboard-Hilfsfunktionen (rein optisch — keine Bewertung der Bewerber)
# ---------------------------------------------------------------------------


def quality_bucket(quality: dict | None) -> str:
    """Klassifiziert die DATENQUALITÄT (nicht den Bewerber) in drei Buckets.

    Bewertet ausschließlich, wie vollständig die Unterlage ist:
    vollständig / informationslücken / unvollständig.
    """
    if not quality or quality.get("_error"):
        return "unvollständig"
    missing_n = len(quality.get("missing_information", []))
    unclear_n = len(quality.get("unclear_information", []))
    if missing_n == 0 and unclear_n == 0:
        return "vollständig"
    if missing_n >= 3:
        return "unvollständig"
    return "informationslücken"


def render_donut(buckets: dict[str, int]) -> str:
    total = sum(buckets.values()) or 1
    colors = {
        "vollständig": "#10B981",
        "informationslücken": "#F59E0B",
        "unvollständig": "#EF4444",
    }
    cx, cy, r = 70, 70, 52
    circumference = 2 * 3.14159 * r
    offset = 0
    arcs: list[str] = []
    for k in ("vollständig", "informationslücken", "unvollständig"):
        v = buckets.get(k, 0)
        if v == 0:
            continue
        frac = v / total
        arc_len = circumference * frac
        arcs.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="transparent" '
            f'stroke="{colors[k]}" stroke-width="18" '
            f'stroke-dasharray="{arc_len:.2f} {circumference:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"/>'
        )
        offset += arc_len
    if not arcs:
        arcs.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="transparent" '
            f'stroke="#1E3A5F" stroke-width="18"/>'
        )
    return (
        '<svg width="140" height="140" viewBox="0 0 140 140">'
        + "".join(arcs)
        + f'<text x="70" y="70" text-anchor="middle" dominant-baseline="middle" '
        f'fill="#F8FAFC" font-size="22" font-weight="700">{sum(buckets.values())}</text>'
        + f'<text x="70" y="92" text-anchor="middle" '
        f'fill="#94A3B8" font-size="11">Gesamt</text>'
        + "</svg>"
    )


def initials_for(name: str) -> str:
    parts = [p for p in (name or "").strip().split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


# ---------------------------------------------------------------------------
# Sidebar — Logo, Navigation, Eingaben, Hilfe
# ---------------------------------------------------------------------------

candidate_count = len(st.session_state.candidates)

with st.sidebar:
    st.markdown(
        '<div class="kmu-logo">👥 KMU Recruiting Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="kmu-nav-section">Übersicht</div>
        <div class="kmu-nav-item active">📊 Dashboard</div>
        <div class="kmu-nav-section">Stellenprofil</div>
        <div class="kmu-nav-item">📝 Stellenprofil</div>
        <div class="kmu-nav-item">✅ Anforderungen</div>
        <div class="kmu-nav-section">Bewerbungen</div>
        <div class="kmu-nav-item">📂 Bewerbungen
            <span class="badge-num">{candidate_count}</span>
        </div>
        <div class="kmu-nav-item">👤 Kandidaten</div>
        <div class="kmu-nav-section">Analyse</div>
        <div class="kmu-nav-item">📈 Bewerbungsqualität</div>
        <div class="kmu-nav-item">❓ Rückfragen</div>
        <div class="kmu-nav-section">Kommunikation</div>
        <div class="kmu-nav-item">📄 Vorlagen</div>
        <div class="kmu-nav-item">✉️ E-Mails</div>
        <div class="kmu-nav-section">Einstellungen</div>
        <div class="kmu-nav-item">⚙️ Einstellungen</div>
        <div class="kmu-nav-item">📋 Audit Log</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="kmu-nav-section">Stellenprofil eingeben</div>',
        unsafe_allow_html=True,
    )
    job_text = st.text_area(
        "Stellenprofil",
        height=160,
        placeholder=(
            "z. B. Wir suchen einen Python-Entwickler mit SQL und Deutsch C1 …"
        ),
        label_visibility="collapsed",
    )

    if st.button(
        "Stellenprofil analysieren", disabled=not job_text.strip()
    ):
        with st.spinner("Strukturiere Stellenprofil ..."):
            try:
                profile = extract_job_profile(job_text)
                st.session_state.job_profile = profile
                st.session_state.job_profile_source = job_text.strip()
                log_audit(
                    action="Stellenprofil analysiert",
                    target="(Sidebar-Eingabe)",
                    result_type="strukturiertes Stellenprofil",
                )
                st.success("Stellenprofil strukturiert.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Fehler bei der Analyse des Stellenprofils: {e}")
                log_audit(
                    action="Stellenprofil analysiert",
                    target="(Sidebar-Eingabe)",
                    result_type=f"Fehler: {e}",
                )

    if st.session_state.job_profile and st.button("Stellenprofil löschen"):
        st.session_state.job_profile = None
        st.session_state.job_profile_source = ""
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="kmu-nav-section">Lebensläufe hochladen</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "PDF-Dateien",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
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
                        log_audit(
                            action="PDF hochgeladen",
                            target=f.name,
                            result_type="Datei akzeptiert",
                        )
                        text = extract_text_from_pdf(f.read())
                        if not text:
                            st.warning(
                                f"{f.name}: kein Text extrahierbar "
                                "(evtl. gescanntes PDF ohne OCR)."
                            )
                            log_audit(
                                action="Text extrahiert",
                                target=f.name,
                                result_type="leer / nicht maschinenlesbar",
                            )
                            continue
                        log_audit(
                            action="Text extrahiert",
                            target=f.name,
                            result_type=f"{len(text)} Zeichen",
                        )
                        data = extract_cv(text, prior_feedback)
                        log_audit(
                            action="CV-Daten extrahiert",
                            target=f.name,
                            result_type="strukturierte Felder + Belege",
                        )
                        try:
                            quality = analyze_data_quality(text, data)
                            log_audit(
                                action="Datenqualität geprüft",
                                target=data.get("name", "") or f.name,
                                result_type=(
                                    f"{len(quality['missing_information'])} fehlend, "
                                    f"{len(quality['unclear_information'])} unklar, "
                                    f"{len(quality['suggested_questions'])} Rückfragen"
                                ),
                            )
                        except Exception as qe:  # noqa: BLE001
                            quality = {
                                "missing_information": [],
                                "unclear_information": [],
                                "suggested_questions": [],
                                "_error": str(qe),
                            }
                            log_audit(
                                action="Datenqualität geprüft",
                                target=f.name,
                                result_type=f"Fehler: {qe}",
                            )
                        st.session_state.candidates.append(
                            {"filename": f.name, "data": data, "quality": quality}
                        )
                        st.session_state.processed_files.add(f.name)
                        ok += 1
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Fehler bei {f.name}: {e}")
                        log_audit(
                            action="CV-Daten extrahiert",
                            target=f.name,
                            result_type=f"Fehler: {e}",
                        )
                progress.progress(i / len(new_files))
            if ok:
                st.success(f"{ok} Lebensläufe extrahiert.")

    if st.session_state.candidates and st.button("Alle Kandidaten löschen"):
        st.session_state.candidates = []
        st.session_state.processed_files = set()
        st.rerun()

    st.divider()
    st.button("❔ Hilfe & Support", use_container_width=True)


# ---------------------------------------------------------------------------
# Topbar + Dashboard-Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="kmu-topbar">
        <div class="kmu-topbar-left">☰</div>
        <div class="kmu-topbar-right">
            <div class="kmu-topbar-icon">🔔<span class="badge">3</span></div>
            <div class="kmu-topbar-icon">❔</div>
            <span class="kmu-avatar">GF</span>
            <span>Geschäftsführer ▾</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        """
        <div class="dashboard-header">
            <div>
                <h1>Dashboard</h1>
                <div class="subtitle">Übersicht Ihrer Bewerbungen und Analysen</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown("<div style='height: 18px'></div>", unsafe_allow_html=True)
    st.button(
        "⬆ Bewerbungen hochladen",
        use_container_width=True,
        help="Lebensläufe lädst du in der Sidebar links hoch.",
    )

st.markdown(
    """
    <div class="kmu-disclaimer">
        <strong>Hinweis:</strong> Der Agent trifft keine Personalentscheidung.
        Er erstellt kein Ranking, keinen Score und keine Empfehlung.
        Er extrahiert ausschließlich objektive Informationen und bereitet sie
        für den Geschäftsführer auf. Autonomie-Stufe:
        <em>Rot — Mensch entscheidet, Agent liefert nur Daten.</em>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- KPI-Karten (Marktkontext, keine Bewerberbewertung) ----

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.markdown(
        """
        <div class="kpi-card red">
            <div class="kpi-value red">~50 %</div>
            <div class="kpi-label">der Bewerbungen passen nicht von Anfang an</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        """
        <div class="kpi-card orange">
            <div class="kpi-value orange">Std./Woche</div>
            <div class="kpi-label">manuelle Sichtung pro offene Stelle</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        """
        <div class="kpi-card teal">
            <div class="kpi-value teal">40.000 €</div>
            <div class="kpi-label">kostet eine einzige Fehlbesetzung im KMU</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Zitat-Card ----

st.markdown(
    """
    <div class="kmu-quote">
        <div class="kmu-quote-mark">&ldquo;</div>
        <div class="kmu-quote-body">
            <div class="kmu-quote-text">
                Etwa die Hälfte oder mehr der Bewerbungen passt nicht.
                Und trotzdem müssen wir jede einzelne ansehen.
            </div>
            <div class="kmu-quote-source">
                — Ali Metaj, GF GFL Garten- und Forstbau GmbH ·
                Interview 27.05.2026 · 40 Mitarbeiter, kein HR-Team
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Dashboard-Grid: Letzte Bewerbungen | Qualität + Rückfragen ----

# Datenaggregation aus echten Daten (oder Platzhalter)
recent_html_rows: list[str] = []
if st.session_state.candidates:
    for c in st.session_state.candidates[-5:][::-1]:
        cv = c["data"]
        q = c.get("quality") or {}
        has_gaps = bool(
            q.get("missing_information") or q.get("unclear_information")
        )
        badge_class = "kmu-badge-warn" if has_gaps else "kmu-badge-ok"
        badge_label = (
            "Informationslücken" if has_gaps else "Analyse abgeschlossen"
        )
        name = cv.get("name") or "(ohne Name)"
        initials = initials_for(name)
        first_role = ""
        if cv.get("experience"):
            first_role = cv["experience"][0].get("role", "")
        recent_html_rows.append(
            f"""
            <div class="kmu-list-item">
                <div class="kmu-list-avatar">{initials}</div>
                <div style="flex:1">
                    <div class="kmu-list-name">{name}</div>
                    <div class="kmu-list-file">{first_role or c['filename']}</div>
                </div>
                <span class="kmu-badge {badge_class}">● {badge_label}</span>
            </div>
            """
        )
else:
    placeholders = [
        ("Max Mustermann", "Data Analyst", "kmu-badge-ok", "Analyse abgeschlossen"),
        ("Lisa Schneider", "Marketing Managerin", "kmu-badge-warn", "Informationslücken"),
        ("Tom Tischler", "Projektmanager", "kmu-badge-ok", "Analyse abgeschlossen"),
        ("Julia Sommer", "Buchhalterin", "kmu-badge-warn", "Informationslücken"),
        ("Anton Keller", "Vertriebsmitarbeiter", "kmu-badge-ok", "Analyse abgeschlossen"),
    ]
    for name, role, badge_class, label in placeholders:
        recent_html_rows.append(
            f"""
            <div class="kmu-list-item">
                <div class="kmu-list-avatar">{initials_for(name)}</div>
                <div style="flex:1">
                    <div class="kmu-list-name">{name}</div>
                    <div class="kmu-list-file">{role}</div>
                </div>
                <span class="kmu-badge {badge_class}">● {label}</span>
            </div>
            """
        )

# Bewerbungsqualitäts-Verteilung (Datenqualität, NICHT Eignung)
buckets = {"vollständig": 0, "informationslücken": 0, "unvollständig": 0}
if st.session_state.candidates:
    for c in st.session_state.candidates:
        buckets[quality_bucket(c.get("quality"))] += 1
else:
    buckets = {"vollständig": 14, "informationslücken": 18, "unvollständig": 8}
total_buckets = sum(buckets.values()) or 1
donut_svg = render_donut(buckets)

# Offene Rückfragen
total_questions = sum(
    len((c.get("quality") or {}).get("suggested_questions") or [])
    for c in st.session_state.candidates
)
if not st.session_state.candidates:
    total_questions = 12

left_col, right_col = st.columns([1.4, 1])

with left_col:
    st.markdown(
        f"""
        <div class="kmu-card">
            <div class="kmu-card-title">
                Letzte Bewerbungen
                <small>{'Echtdaten' if st.session_state.candidates else 'Beispieldaten'}</small>
            </div>
            {''.join(recent_html_rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        f"""
        <div class="kmu-card">
            <div class="kmu-card-title">Bewerbungsqualität <small>(Datenqualität, keine Eignung)</small></div>
            <div class="donut-wrap">
                {donut_svg}
                <div class="donut-legend">
                    <div class="donut-legend-row">
                        <span class="donut-dot" style="background:#10B981"></span>
                        Vollständig
                        <span class="donut-count">{buckets['vollständig']} ({buckets['vollständig'] * 100 // total_buckets}%)</span>
                    </div>
                    <div class="donut-legend-row">
                        <span class="donut-dot" style="background:#F59E0B"></span>
                        Informationslücken
                        <span class="donut-count">{buckets['informationslücken']} ({buckets['informationslücken'] * 100 // total_buckets}%)</span>
                    </div>
                    <div class="donut-legend-row">
                        <span class="donut-dot" style="background:#EF4444"></span>
                        Unvollständig
                        <span class="donut-count">{buckets['unvollständig']} ({buckets['unvollständig'] * 100 // total_buckets}%)</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="kmu-card">
            <div class="kmu-card-title">Offene Rückfragen</div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="font-size:38px; font-weight:700; color:#14B8A6;">{total_questions}</div>
                <div style="color:#94A3B8; font-size:13px;">
                    Rückfragen vorbereitet —<br/>
                    warten auf Ihre Prüfung und Freigabe.<br/>
                    <em style="color:#94A3B8">Es wird keine E-Mail automatisch versendet.</em>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Funktionsbereiche als Tabs unterhalb des Dashboards
# ---------------------------------------------------------------------------

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

(
    main_tab_apps,
    main_tab_profile,
    main_tab_candidates,
    main_tab_quality,
    main_tab_questions,
    main_tab_audit,
    main_tab_tests,
) = st.tabs(
    [
        "Bewerbungen analysieren",
        "Stellenprofil",
        "Kandidatenübersicht",
        "Bewerbungsqualität",
        "Rückfragen",
        "Audit Log",
        "Test-Center",
    ]
)

# ---- Tab: Bewerbungen analysieren ----

with main_tab_apps:
    st.markdown(
        '<div class="kmu-card-title">Bewerbungen analysieren</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Lade in der Sidebar links PDF-Lebensläufe hoch und klicke auf "
        "**Extraktion starten**. Der Agent extrahiert objektive Informationen, "
        "schlägt Rückfragen vor und protokolliert jede Aktion im Audit-Log. "
        "Es findet keine Bewertung des Bewerbers statt."
    )
    if st.session_state.candidates:
        st.success(
            f"{len(st.session_state.candidates)} Bewerbungen analysiert. "
            "Details siehst du im Tab „Kandidatenübersicht“."
        )
    else:
        st.info("Noch keine Bewerbungen analysiert.")
    st.markdown("**Bisheriges Feedback (fließt in den nächsten Lauf ein)**")
    entries = load_feedback()
    if entries:
        st.dataframe(
            pd.DataFrame(entries), use_container_width=True, hide_index=True
        )
    else:
        st.caption("Noch kein Feedback vorhanden.")

# ---- Tab: Stellenprofil ----

with main_tab_profile:
    st.markdown(
        '<div class="kmu-card-title">Strukturiertes Stellenprofil</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Stellenprofil-Text in der Sidebar einfügen und auf "
        "„Stellenprofil analysieren“ klicken. Die KI strukturiert nur — "
        "sie bewertet nichts und gewichtet nichts."
    )
    if st.session_state.job_profile:
        st.json(st.session_state.job_profile)
    else:
        st.info("Noch kein Stellenprofil hinterlegt.")

# ---- Tab: Kandidatenübersicht ----

with main_tab_candidates:
    st.markdown(
        '<div class="kmu-card-title">Kandidatenübersicht</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.candidates:
        st.info(
            "Noch keine Kandidaten — bitte zuerst Bewerbungen analysieren."
        )
    else:
        st.caption(
            "Alle Kandidaten bleiben sichtbar. Keine Reihenfolge ist eine "
            "Bewertung — die Sortierung folgt der Upload-Reihenfolge."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            skills_q = st.text_input("Filter: Skills (Komma-getrennt)", "")
        with col2:
            languages_q = st.text_input("Filter: Sprachen", "")
        with col3:
            certs_q = st.text_input("Filter: Zertifikate", "")

        filtered = filter_candidates(
            st.session_state.candidates, skills_q, languages_q, certs_q
        )
        st.markdown(
            f"**{len(filtered)} von {len(st.session_state.candidates)} Kandidaten**"
        )
        df = candidates_to_dataframe(filtered)
        st.dataframe(df, use_container_width=True, hide_index=True)

        options = [
            f"{c['data'].get('name') or '(ohne Name)'} – {c['filename']}"
            for c in filtered
        ]
        if options:
            idx = st.selectbox(
                "Details ansehen",
                range(len(options)),
                format_func=lambda i: options[i],
            )
            selected = filtered[idx]
            cv_data = selected["data"]

            (
                tab_overview,
                tab_checklist,
                tab_evidence,
                tab_quality,
                tab_feedback,
            ) = st.tabs(
                [
                    "Strukturierte Daten",
                    "Kriterien-Checkliste",
                    "Textbelege",
                    "Datenqualität",
                    "Feedback",
                ]
            )

            with tab_overview:
                st.json(cv_data)

            with tab_checklist:
                if st.session_state.job_profile:
                    st.caption(
                        "Neutrale Gegenüberstellung mit drei Status: "
                        "vorhanden / unklar / nicht gefunden. "
                        "Keine Prozentzahl, kein Score, kein Ranking."
                    )
                    render_checklist(st.session_state.job_profile, cv_data)
                else:
                    st.info(
                        "Kein Stellenprofil hinterlegt. Füge in der Sidebar "
                        "ein Stellenprofil ein, um eine Checkliste zu sehen."
                    )

            with tab_evidence:
                st.caption(
                    "Quellenangaben aus dem Lebenslauf — kurze Textausschnitte "
                    "als Beleg pro erkannter Qualifikation."
                )
                ev_list = cv_data.get("evidence", [])
                if ev_list:
                    for ev in ev_list:
                        st.markdown(
                            f"- **{ev.get('item', '')}**: „{ev.get('excerpt', '')}“"
                        )
                else:
                    st.write("Keine separaten Textbelege erfasst.")

            with tab_quality:
                st.caption(
                    "Vollständigkeits- und Klarheitsprüfung der Unterlagen — "
                    "keine Bewertung des Bewerbers, keine Empfehlung, keine "
                    "Priorisierung."
                )
                quality = selected.get("quality") or {}
                if quality.get("_error"):
                    st.error(
                        "Die Datenqualitäts-Prüfung konnte nicht durchgeführt "
                        f"werden: {quality['_error']}"
                    )

                st.markdown("### Fehlende Informationen")
                missing = quality.get("missing_information", [])
                if missing:
                    for item in missing:
                        st.markdown(f"- {item}")
                else:
                    st.write("Keine fehlenden Informationen erkannt.")

                st.markdown("### Unklare Informationen")
                unclear = quality.get("unclear_information", [])
                if unclear:
                    for item in unclear:
                        st.markdown(f"- {item}")
                else:
                    st.write("Keine unklaren Informationen erkannt.")

                st.markdown("### Vorgeschlagene Rückfragen")
                questions = quality.get("suggested_questions", [])
                if questions:
                    for q in questions:
                        st.markdown(f"- {q}")
                else:
                    st.write("Keine offenen Rückfragen.")

                extra_issues = cv_data.get("data_quality_issues", [])
                if extra_issues:
                    with st.expander(
                        "Zusätzliche Hinweise aus der CV-Extraktion"
                    ):
                        for issue in extra_issues:
                            st.markdown(f"- {issue}")

            with tab_feedback:
                st.markdown(
                    "Feedback verbessert nur die **Extraktion und "
                    "Darstellung**, nicht die Bewerberauswahl. Es wird in "
                    "`feedback.json` gespeichert und beim nächsten "
                    "Extraktionslauf im Prompt berücksichtigt."
                )
                with st.form("feedback_form", clear_on_submit=True):
                    category = st.selectbox(
                        "Art der Korrektur", FEEDBACK_CATEGORIES
                    )
                    note = st.text_area(
                        "Anmerkung",
                        placeholder=(
                            "z. B. „SQL wurde übersehen“ oder "
                            "„Sprache Französisch falsch erkannt“"
                        ),
                    )
                    submitted = st.form_submit_button("Feedback speichern")
                    if submitted:
                        if not note.strip():
                            st.warning("Bitte eine Anmerkung eingeben.")
                        else:
                            add_feedback(
                                cv_data.get("name", ""), category, note
                            )
                            log_audit(
                                action="Feedback gespeichert",
                                target=cv_data.get("name", "")
                                or selected["filename"],
                                result_type=category,
                            )
                            st.success("Feedback gespeichert.")
        else:
            st.write("Keine Bewerber entsprechen den Filtern.")

# ---- Tab: Bewerbungsqualität (Aggregation) ----

with main_tab_quality:
    st.markdown(
        '<div class="kmu-card-title">Bewerbungsqualität (Aggregation)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Aggregierte Datenqualitäts-Sicht über alle bislang analysierten "
        "Bewerbungen — keine Eignungsbewertung."
    )
    if not st.session_state.candidates:
        st.info("Noch keine Bewerbungen analysiert.")
    else:
        rows = []
        for c in st.session_state.candidates:
            q = c.get("quality") or {}
            rows.append(
                {
                    "Datei": c["filename"],
                    "Name": c["data"].get("name", ""),
                    "Bucket": quality_bucket(q),
                    "Fehlend": len(q.get("missing_information", [])),
                    "Unklar": len(q.get("unclear_information", [])),
                    "Rückfragen": len(q.get("suggested_questions", [])),
                }
            )
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True
        )

# ---- Tab: Rückfragen (konsolidiert) ----

with main_tab_questions:
    st.markdown(
        '<div class="kmu-card-title">Vorgeschlagene Rückfragen</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Rückfragevorschläge pro Bewerbung. **Es wird keine E-Mail "
        "automatisch versendet** — alle Rückfragen warten auf Ihre Prüfung "
        "und Freigabe."
    )
    any_questions = False
    for c in st.session_state.candidates:
        questions = (c.get("quality") or {}).get("suggested_questions") or []
        if not questions:
            continue
        any_questions = True
        with st.expander(
            f"{c['data'].get('name') or '(ohne Name)'} — "
            f"{len(questions)} Rückfrage(n)"
        ):
            for q in questions:
                st.markdown(f"- {q}")
    if not any_questions:
        st.info("Keine offenen Rückfragen.")

# ---- Tab: Audit Log ----

with main_tab_audit:
    st.markdown(
        '<div class="kmu-card-title">Audit-Log</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Jeder Eintrag enthält die Autonomie-Stufe. "
        f"Standardstufe für Recruiting-Analysen: {AUTONOMY_LEVEL}"
    )
    log_entries = load_audit_log()
    if log_entries:
        st.dataframe(
            pd.DataFrame(log_entries),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Audit-Log ist leer.")

# ---- Tab: Test-Center ----

with main_tab_tests:
    st.markdown(
        '<div class="kmu-card-title">Test-Center</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Manuelle Tests des Extraktions-Workflows. "
        "Hier werden **keine Bewerber bewertet** — nur der Agent selbst wird "
        "anhand drei Testtypen geprüft."
    )
    for tc in TEST_CASES:
        with st.expander(f"{tc['name']} — {tc['description']}"):
            with st.form(f"test_form_{tc['id']}", clear_on_submit=True):
                outcome = st.selectbox(
                    "Ergebnis",
                    TEST_OUTCOMES,
                    key=f"outcome_{tc['id']}",
                )
                note = st.text_input(
                    "Notiz (optional)", key=f"note_{tc['id']}"
                )
                if st.form_submit_button("Ergebnis speichern"):
                    save_test_result(tc["id"], outcome, note)
                    log_audit(
                        action="Testergebnis gespeichert",
                        target=tc["name"],
                        result_type=outcome,
                    )
                    st.success(
                        f"Testergebnis für „{tc['name']}“ gespeichert."
                    )

    with st.expander("Bisherige Testergebnisse"):
        tr = load_test_results()
        if tr:
            st.dataframe(
                pd.DataFrame(tr), use_container_width=True, hide_index=True
            )
        else:
            st.write("Noch keine Testergebnisse vorhanden.")

st.caption(
    "Der Agent trifft keine Personalentscheidung. Er extrahiert objektive "
    "Informationen und macht Bewerbungen vergleichbar. Die finale Bewertung "
    "trifft immer der Geschäftsführer."
)
