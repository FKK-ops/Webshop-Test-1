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


def check_criterion(criterion: str, cv_data: dict) -> dict:
    """Prüft neutral, ob ein Kriterium im CV vorkommt, und liefert ggf. Beleg.

    Liefert {"present": bool, "evidence": str | None}.
    Keine Bewertung, keine Gewichtung — nur Vorhanden / Nicht-Gefunden plus
    Textausschnitt aus dem Lebenslauf.
    """
    crit = criterion.strip()
    if not crit:
        return {"present": False, "evidence": None}

    haystack = _searchable_text(cv_data).lower()
    present = crit.lower() in haystack

    evidence: str | None = None
    if present:
        crit_low = crit.lower()
        for ev in cv_data.get("evidence", []):
            item = (ev.get("item") or "").lower()
            if not item:
                continue
            if crit_low in item or item in crit_low:
                evidence = ev.get("excerpt") or None
                break
    return {"present": present, "evidence": evidence}


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
            result = check_criterion(c, cv_data)
            if result["present"]:
                evidence = result["evidence"] or "Kein eindeutiger Beleg gefunden"
                st.markdown(f"- **{c}** — vorhanden  \n  *Beleg:* „{evidence}“")
            else:
                st.markdown(f"- **{c}** — nicht gefunden")
    if not any_rendered:
        st.write("Stellenprofil enthält keine prüfbaren Kriterien.")


# ---------------------------------------------------------------------------
# Streamlit-UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HR-Recruiting-Assistent", layout="wide")

st.title("HR-Recruiting-Assistent für KMU")
st.write(
    "Lade PDF-Lebensläufe hoch und füge optional ein Stellenprofil ein. "
    "Der Agent extrahiert objektive Informationen aus den Lebensläufen, "
    "strukturiert das Stellenprofil und zeigt pro Bewerber eine neutrale "
    "Kriterien-Checkliste. "
    "**Der Agent trifft keine Personalentscheidung und erstellt keine "
    "Empfehlung.** Die finale Bewertung trifft immer der Geschäftsführer."
)
st.info(DISCLAIMER)
st.caption(f"Autonomie-Stufe für alle Analysen: {AUTONOMY_LEVEL}")

# Session State
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "job_profile" not in st.session_state:
    st.session_state.job_profile = None
if "job_profile_source" not in st.session_state:
    st.session_state.job_profile_source = ""

# ---- Sidebar: Stellenprofil + Upload ----

with st.sidebar:
    st.header("Stellenprofil")
    job_text = st.text_area(
        "Stellenprofil einfügen oder eingeben",
        height=180,
        placeholder="z. B. Wir suchen einen Python-Entwickler mit SQL und Deutsch C1 …",
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
                        st.session_state.candidates.append(
                            {"filename": f.name, "data": data}
                        )
                        st.session_state.processed_files.add(f.name)
                        ok += 1
                        log_audit(
                            action="CV-Daten extrahiert",
                            target=f.name,
                            result_type="strukturierte Felder + Belege",
                        )
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

# ---- Stellenprofil-Anzeige ----

if st.session_state.job_profile:
    with st.expander(
        "Strukturiertes Stellenprofil (nur Struktur, keine Bewertung)",
        expanded=False,
    ):
        st.json(st.session_state.job_profile)

# ---- Hauptbereich: Bewerber-Tabelle, Filter, Details ----

if st.session_state.candidates:
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

    st.subheader(
        f"Bewerber ({len(filtered)} von {len(st.session_state.candidates)})"
    )
    st.caption(
        "Alle Bewerber bleiben sichtbar. Keine Reihenfolge ist eine "
        "Bewertung — die Sortierung folgt der Upload-Reihenfolge."
    )
    df = candidates_to_dataframe(filtered)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Details pro Bewerber")
    options = [
        f"{c['data'].get('name') or '(ohne Name)'} – {c['filename']}"
        for c in filtered
    ]
    if not options:
        st.write("Keine Bewerber entsprechen den Filtern.")
    else:
        idx = st.selectbox(
            "Bewerber auswählen",
            range(len(options)),
            format_func=lambda i: options[i],
        )
        selected = filtered[idx]
        cv_data = selected["data"]

        tab_overview, tab_checklist, tab_evidence, tab_quality, tab_feedback = (
            st.tabs(
                [
                    "Strukturierte Daten",
                    "Kriterien-Checkliste",
                    "Textbelege",
                    "Unklarheiten",
                    "Feedback",
                ]
            )
        )

        with tab_overview:
            st.json(cv_data)

        with tab_checklist:
            if st.session_state.job_profile:
                st.caption(
                    "Neutrale Gegenüberstellung: Vorhanden / Nicht gefunden. "
                    "Keine Prozentzahl, kein Score, kein Ranking."
                )
                render_checklist(st.session_state.job_profile, cv_data)
            else:
                st.info(
                    "Kein Stellenprofil hinterlegt. Füge in der Sidebar ein "
                    "Stellenprofil ein, um eine Checkliste zu sehen."
                )

        with tab_evidence:
            ev_list = cv_data.get("evidence", [])
            if ev_list:
                for ev in ev_list:
                    st.markdown(
                        f"- **{ev.get('item', '')}**: „{ev.get('excerpt', '')}“"
                    )
            else:
                st.write("Keine separaten Textbelege erfasst.")

        with tab_quality:
            issues = cv_data.get("data_quality_issues", [])
            st.caption(
                "Datenqualitäts-Hinweise — keine Bewertung des Bewerbers."
            )
            if issues:
                for issue in issues:
                    st.markdown(f"- {issue}")
            else:
                st.write("Keine Unklarheiten erkannt.")

        with tab_feedback:
            st.markdown(
                "Feedback verbessert nur die **Extraktion und Darstellung**, "
                "nicht die Bewerberauswahl. Es wird in `feedback.json` "
                "gespeichert und beim nächsten Extraktionslauf im Prompt "
                "berücksichtigt."
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
                            target=cv_data.get("name", "") or selected["filename"],
                            result_type=category,
                        )
                        st.success("Feedback gespeichert.")
else:
    st.write("Lade links Lebensläufe hoch, um zu starten.")

# ---- Bisheriges Feedback ----

with st.expander("Bisheriges Feedback"):
    entries = load_feedback()
    if entries:
        st.dataframe(
            pd.DataFrame(entries), use_container_width=True, hide_index=True
        )
    else:
        st.write("Noch kein Feedback vorhanden.")

# ---- Audit-Log ----

with st.expander("Audit-Log (alle Aktionen)"):
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
        st.write("Audit-Log ist leer.")

# ---- Test-Center ----

st.subheader("Test-Center")
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
                st.success(f"Testergebnis für „{tc['name']}“ gespeichert.")

with st.expander("Bisherige Testergebnisse"):
    tr = load_test_results()
    if tr:
        st.dataframe(
            pd.DataFrame(tr), use_container_width=True, hide_index=True
        )
    else:
        st.write("Noch keine Testergebnisse vorhanden.")

st.caption(DISCLAIMER)
