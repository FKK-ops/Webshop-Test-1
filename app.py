"""
HR-Recruiting-Assistent (Streamlit-Prototyp, Single-File).

Demo-Version eines KI-Recruiting-Agenten für KMU. Strukturiert Stellen-
profile und Lebensläufe, prüft Informationslücken und schlägt Rückfragen
vor. Die KI trifft keine Personalentscheidung — der Geschäftsführer prüft
und entscheidet final.

Pipeline (jeder Schritt ist ein eigener Claude-Agent):
    1. analyze_job_profile()           Stellenprofil-Agent
    2. extract_cv()                    CV-Agent
    3. analyze_information_gaps()      Informationslücken-Agent
    4. generate_follow_up_questions()  Rückfragen-Agent

Grundregel:
- Die KI trifft keine Personalentscheidung.
- Die KI erstellt kein Ranking, keinen Score und keine Empfehlung.
- Die KI extrahiert objektive Informationen und macht Bewerbungen
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
import streamlit.components.v1 as components
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
    "Die KI trifft **keine Personalentscheidung**. "
    "Sie erstellt **kein Ranking, keinen Score und keine Empfehlung**. "
    "Sie strukturiert ausschließlich objektive Informationen. "
    "Der Geschäftsführer prüft und entscheidet final."
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
- Keine Interpretation von Persönlichkeit, Motivation, Alter, Herkunft,
  Geschlecht, Name, Foto oder anderen weichen Merkmalen.
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

Keine Bewertung des Bewerbers, keine Interpretation weicher Merkmale.
"""

QUALITY_INSTRUCTIONS = """Du bist der Informationslücken-Agent. Du
analysierst ausschließlich die Datenqualität und Vollständigkeit der
Bewerbungsunterlage. Du bewertest NICHT die Eignung des Bewerbers, du
gibst keine Empfehlung, du priorisierst nicht und du vergibst keinen Score.

Liefere drei Listen:

- missing_information: Felder oder Angaben, die im Lebenslauf gar nicht
  vorkommen. Beispiele:
    "Sprachkenntnisse nicht angegeben"
    "Zertifikate nicht angegeben"
    "Verfügbarkeit nicht angegeben"
    "Berufserfahrung enthält keine Zeiträume"
- unclear_information: Angaben, die zwar vorhanden, aber unklar,
  unvollständig oder mehrdeutig sind. Beispiele:
    "Dauer der SQL-Erfahrung unklar"
    "Sprachlevel nicht eindeutig angegeben"
    "Zertifikat erwähnt, aber Name fehlt"
    "Berufserfahrung nur allgemein beschrieben"
- suggested_questions: konkrete Rückfragen an den Bewerber, die helfen
  würden, fehlende oder unklare Informationen zu klären.

Wenn alles klar und vollständig ist, lasse die jeweiligen Arrays leer.
Keine Bewertung des Bewerbers. Keine Empfehlung. Kein Ranking.
"""

JOB_PROFILE_INSTRUCTIONS = """Du bist der Stellenprofil-Agent. Du
strukturierst Anforderungen aus einem Stellenprofil-Text. Du bewertest
nichts und gewichtest nichts.

KERNREGEL — sehr wichtig:
Extrahiere AUSSCHLIESSLICH objektiv prüfbare Qualifikationen in die
fachlichen Felder. Soft Skills, Persönlichkeitsmerkmale und nicht
messbare Eigenschaften gehören NIEMALS in must_criteria, nice_criteria,
desired_skills, desired_languages, desired_certificates oder
desired_experience. Sie kommen ausschließlich in das separate Feld
non_checkable_requirements.

Felder:
- role: Stellentitel (z. B. "Senior Python-Entwickler/in")
- must_criteria: Muss-Kriterien — NUR objektiv prüfbar (Skills, Tools,
  Frameworks, Methoden, Sprachen, Zertifikate, konkrete Berufserfahrung)
- nice_criteria: Kann-Kriterien („wünschenswert", „von Vorteil") —
  ebenfalls NUR objektiv prüfbar
- desired_skills: konkrete fachliche Skills (Technologien, Tools,
  Frameworks, Programmiersprachen, Methoden) — KEINE Soft Skills
- desired_languages: Sprachen inkl. Niveau (A1–C2, Muttersprache),
  wenn genannt
- desired_certificates: konkret benannte Zertifikate / Abschlüsse
- desired_experience: konkrete Berufserfahrung (Branche, Rolle, Jahre,
  Projekte)
- non_checkable_requirements: Liste aller im Stellenprofil genannten
  Soft Skills, Persönlichkeitsmerkmale und nicht messbaren Eigenschaften
  (z. B. Kreativität, Teamfähigkeit, Motivation, Belastbarkeit,
  Eigeninitiative, Lernbereitschaft, Kommunikationsstärke, kulturelle
  Passung, Persönlichkeit, Mindset, Leidenschaft, Flexibilität ohne
  Konkretisierung, technische Affinität ohne konkrete Tools,
  selbstständige Arbeitsweise, proaktives Denken, Innovationsfähigkeit,
  Hands-on-Mentalität). Diese fließen NICHT in den fachlichen Abgleich,
  werden aber transparent angezeigt.

VERBOTEN in den objektiven Feldern (gehört entweder in
non_checkable_requirements oder gar nicht in das Profil):
- Soft Skills jeder Art (Kreativität, Teamfähigkeit, Motivation,
  Belastbarkeit, Eigeninitiative, Lernbereitschaft, Kommunikationsstärke,
  selbstständige Arbeitsweise, proaktives Denken, Hands-on-Mentalität,
  Innovationsfähigkeit, …)
- Persönlichkeitsmerkmale (Persönlichkeit, Mindset, Leidenschaft,
  Engagement, freundlich, sympathisch …)
- weiche Formulierungen ohne objektiven Nachweis („technische Affinität"
  ohne konkrete Tools, „Flexibilität" ohne Konkretisierung,
  „lösungsorientiert", „kundenorientiert", „serviceorientiert", …)
- organisatorische Angaben (Gehalt, Vollzeit/Teilzeit, Remote/Homeoffice,
  Standort, Verfügbarkeit, Startdatum, Referenzen, Anschreiben/Foto,
  Alter, Geschlecht, Herkunft, Familienstand, …) — diese gehören
  überhaupt nicht ins Profil (auch nicht in non_checkable_requirements).

REGEL „technische Affinität / Flexibilität / Hands-on":
- Wenn die Formulierung konkrete Tools/Technologien nennt
  (z. B. „technische Affinität mit Power BI und SQL"), extrahiere NUR
  die konkreten Tools („Power BI", „SQL") in desired_skills.
- Die übergeordnete weiche Phrase („technische Affinität") wandert
  zusätzlich in non_checkable_requirements.

Wenn ein Feld leer bleibt, gib ein leeres Array zurück.
Keine Bewertung, keine Empfehlung, keine Gewichtung.
"""

FOLLOWUP_INSTRUCTIONS = """Du bist der Rückfragen-Agent. Du formulierst
aus einer Liste von Informationslücken professionelle, höfliche
Rückfragen an den Bewerber. Du bewertest nichts und gibst keine
Empfehlung.

Anforderungen:
- Höfliche Sie-Form.
- Pro Lücke eine konkrete, eindeutige Frage.
- Keine Bewertung, keine Anspielung auf Eignung.
- Maximal 8 Fragen, sortiert nach Relevanz für die ausgeschriebene Rolle
  (aber ohne Reihenfolge als Ranking zu verstehen).
- Wenn vorhandene Vorschläge schon gut sind, übernimm sie sinngemäß.

Beispiele für den Stil:
  "Könnten Sie bitte angeben, auf welchem Niveau Sie Englisch beherrschen?"
  "Würden Sie uns mitteilen, wie lange Sie bereits mit SQL gearbeitet haben?"
  "Können Sie uns den genauen Namen des Zertifikats nennen?"
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
    role: str
    must_criteria: list[str]
    nice_criteria: list[str]
    desired_skills: list[str]
    desired_languages: list[str]
    desired_certificates: list[str]
    desired_experience: list[str]
    non_checkable_requirements: list[str] = []


class QualityCheck(BaseModel):
    missing_information: list[str]
    unclear_information: list[str]
    suggested_questions: list[str]


class FollowUpQuestions(BaseModel):
    questions: list[str]


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
    """Hängt einen Audit-Log-Eintrag an audit_log.json an.

    Felder: Zeitstempel, Aktion, betroffene Datei/Bewerber (target),
    Ergebnis (result_type), Autonomie-Stufe.
    """
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
# Claude-API-Aufrufe (vier Agenten teilen sich denselben API-Wrapper)
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
    """CV-Agent. Strukturiert objektive CV-Daten inkl. Belegen und Datenqualitäts-Hinweisen."""
    feedback_section = ""
    if prior_feedback:
        feedback_section = (
            "\n\nFRÜHERES NUTZER-FEEDBACK (bitte berücksichtigen, damit "
            "ähnliche Extraktions-Fehler nicht wiederholt werden — Feedback "
            "verbessert nur die Extraktion und Darstellung, nicht die "
            "Auswahl):\n"
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


def analyze_job_profile(job_text: str) -> dict:
    """Stellenprofil-Agent. Strukturiert ausschließlich objektiv prüfbare
    Qualifikationen. Soft Skills landen separat in
    non_checkable_requirements und fließen nicht in den fachlichen
    Abgleich ein. Organisatorische Angaben (Gehalt, Standort, …) werden
    komplett entfernt.
    """
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
    return _sanitize_job_profile(parsed.model_dump())


def _sanitize_job_profile(profile: dict) -> dict:
    """Härte die LLM-Ausgabe gegen Soft-Skill- / Organisations-Lecks.

    - Items in fachlichen Feldern, die nicht objektiv prüfbar sind, werden
      entfernt. Soft Skills wandern in non_checkable_requirements;
      organisatorische Angaben (Gehalt, Vollzeit, Standort …) werden
      komplett verworfen — sie gehören weder in den Abgleich noch in die
      Soft-Skill-Liste.
    - Doppelte Einträge werden entfernt (case-insensitive, Reihenfolge
      bleibt erhalten).
    """
    cleaned: dict = dict(profile)
    fachliche_felder = (
        "must_criteria",
        "nice_criteria",
        "desired_skills",
        "desired_languages",
        "desired_certificates",
        "desired_experience",
    )
    soft_collected: list[str] = list(profile.get("non_checkable_requirements") or [])
    seen_soft = {s.lower(): True for s in soft_collected if s}

    for key in fachliche_felder:
        kept: list[str] = []
        seen_low: set[str] = set()
        for raw in profile.get(key) or []:
            text = (raw or "").strip()
            if not text:
                continue
            low = text.lower()
            if low in seen_low:
                continue
            # Organisatorisch (Gehalt, Vollzeit, Standort, …) → komplett raus
            if not is_performance_criterion(text):
                continue
            if is_objectively_checkable(text):
                seen_low.add(low)
                kept.append(text)
            else:
                # weiches Merkmal → in non_checkable_requirements verschieben
                if low not in seen_soft:
                    soft_collected.append(text)
                    seen_soft[low] = True
        cleaned[key] = kept

    # auch die non_checkable-Liste deduplizieren und organisatorische
    # Angaben dort herausfiltern, falls die LLM sie dort abgelegt hat.
    dedup_soft: list[str] = []
    seen2: set[str] = set()
    for raw in soft_collected:
        text = (raw or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in seen2:
            continue
        if not is_performance_criterion(text):
            continue
        seen2.add(low)
        dedup_soft.append(text)
    cleaned["non_checkable_requirements"] = dedup_soft
    return cleaned


# Kompatibilitäts-Alias (alter Funktionsname)
extract_job_profile = analyze_job_profile


def analyze_information_gaps(cv_text: str, cv_data: dict) -> dict:
    """Informationslücken-Agent. Prüft nur Datenqualität, keine Eignung."""
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
        "Gib jetzt die strukturierte Informationslücken-Prüfung zurück."
    )

    response = _call_parse(user_prompt, QualityCheck)
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            "Die Informationslücken-Prüfung konnte nicht geparst werden."
        )
    return parsed.model_dump()


# Kompatibilitäts-Alias (alter Funktionsname)
analyze_data_quality = analyze_information_gaps


def generate_follow_up_questions(
    gaps: dict, job_profile: dict | None = None
) -> dict:
    """Rückfragen-Agent. Formuliert aus Lücken professionelle Rückfragen.

    Versendet NICHTS — die Fragen werden nur vorgeschlagen und warten auf
    Prüfung und Freigabe durch den Geschäftsführer.
    """
    missing = gaps.get("missing_information", []) or []
    unclear = gaps.get("unclear_information", []) or []
    existing = gaps.get("suggested_questions", []) or []
    if not missing and not unclear and not existing:
        return {"questions": []}

    role_hint = ""
    if job_profile and job_profile.get("role"):
        role_hint = f"Ausgeschriebene Rolle: {job_profile['role']}\n"

    user_prompt = (
        f"{FOLLOWUP_INSTRUCTIONS}\n"
        f"{role_hint}"
        "INFORMATIONSLÜCKEN:\n"
        f"Fehlend: {json.dumps(missing, ensure_ascii=False)}\n"
        f"Unklar:  {json.dumps(unclear, ensure_ascii=False)}\n"
        "BEREITS VORGESCHLAGENE FRAGEN (zur Verfeinerung):\n"
        f"{json.dumps(existing, ensure_ascii=False)}\n\n"
        "Gib jetzt die finalen Rückfragen als Liste zurück."
    )

    try:
        response = _call_parse(user_prompt, FollowUpQuestions)
    except RuntimeError:
        # Fallback: bestehende suggested_questions verwenden
        return {"questions": existing}
    parsed = response.parsed_output
    if parsed is None:
        return {"questions": existing}
    return parsed.model_dump()


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die UI
# ---------------------------------------------------------------------------


NICHT_GEFUNDEN = "nicht gefunden"


def _or_missing(value, marker: str = NICHT_GEFUNDEN) -> str:
    """Gibt Wert zurück oder ein neutrales Marker-Wort, wenn leer."""
    if value is None:
        return marker
    if isinstance(value, str) and not value.strip():
        return marker
    if isinstance(value, (list, dict)) and not value:
        return marker
    return value


def _short_experience(d: dict) -> str:
    """Kurze, kompakte Erfahrungs-Zusammenfassung (max. 1 Eintrag)."""
    exp = d.get("experience") or []
    if not exp:
        return "—"
    e = exp[0]
    role = e.get("role", "") or "?"
    company = e.get("company", "") or "?"
    period = e.get("period", "") or ""
    rest = f" (+{len(exp) - 1})" if len(exp) > 1 else ""
    suffix = f" · {period}" if period else ""
    return f"{role} @ {company}{suffix}{rest}"


def _short_languages(d: dict) -> str:
    langs = d.get("languages") or []
    if not langs:
        return "—"
    parts = []
    for l in langs[:3]:
        name = l.get("language", "")
        lvl = l.get("level", "")
        parts.append(f"{name} {lvl}".strip())
    rest = f" +{len(langs) - 3}" if len(langs) > 3 else ""
    return ", ".join(parts) + rest


def candidates_to_dataframe(
    candidates: list[dict], job_profile: dict | None = None
) -> pd.DataFrame:
    """Kompakte Übersichtstabelle ohne Prozentlogik als Hauptanzeige."""
    rows = []
    for c in candidates:
        d = c["data"]
        skills = d.get("skills") or []
        top_skills = ", ".join(skills[:4]) + (
            f" +{len(skills) - 4}" if len(skills) > 4 else ""
        )
        row = {
            "Name": d.get("name") or NICHT_GEFUNDEN,
        }
        if job_profile is not None:
            req_rows = evaluate_candidate_requirements(job_profile, d)
            counts = status_counts(req_rows)
            total = len(req_rows)
            row["Gefunden"] = (
                f"{counts['Gefunden']} / {total}" if total else "—"
            )
            row["Klärungspunkte"] = counts["Teilweise gefunden"]
        row["Wichtigste Skills"] = top_skills or "—"
        row["Sprachen"] = _short_languages(d)
        rows.append(row)
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


def find_source_excerpt(requirement: str, cv_data: dict) -> str | None:
    """Liefert eine kurze, konkrete Fundstelle aus dem Lebenslauf.

    Sucht zuerst in der evidence-Liste; sonst in Skills, Zertifikaten,
    Sprachen und Berufserfahrung; sonst über bekannte Synonyme. Liefert
    None, wenn nichts Belastbares gefunden wurde.
    """
    if not requirement:
        return None
    ev = _evidence_for(requirement, cv_data)
    if ev:
        return ev

    crit_low = requirement.lower()

    def _from_list(label: str, items: list[str], needle: str) -> str | None:
        for it in items:
            if it and (needle in it.lower() or it.lower() in needle):
                preview = ", ".join(items)[:160]
                return f"{label}: {preview}"
        return None

    skills = cv_data.get("skills") or []
    hit = _from_list("Skills", skills, crit_low)
    if hit:
        return hit
    certs = cv_data.get("certificates") or []
    hit = _from_list("Zertifikate", certs, crit_low)
    if hit:
        return hit
    for l in cv_data.get("languages") or []:
        name = (l.get("language") or "").lower()
        if name and (name in crit_low or crit_low in name):
            lvl = l.get("level") or "ohne Niveau"
            return f"Sprachen: {l.get('language','?')} – {lvl}"
    for exp in cv_data.get("experience") or []:
        text = " ".join(
            x for x in [exp.get("role"), exp.get("company"), exp.get("description")] if x
        )
        if crit_low in text.lower():
            snippet = (exp.get("description") or "")[:160] or text[:160]
            return f"{exp.get('role','?')} @ {exp.get('company','?')}: {snippet}"

    # Synonym-Fallback
    for syn in _synonyms_of(crit_low):
        if not syn:
            continue
        hit = _from_list(f"Skills (sinngemäß: {syn})", skills, syn)
        if hit:
            return hit
        for exp in cv_data.get("experience") or []:
            text = " ".join(
                x for x in [exp.get("role"), exp.get("company"), exp.get("description")] if x
            )
            if syn in text.lower():
                snippet = (exp.get("description") or "")[:160] or text[:160]
                return (
                    f"{exp.get('role','?')} @ {exp.get('company','?')} "
                    f"(sinngemäß: {syn}): {snippet}"
                )
    return None



# Bidirektional: jeder Eintrag matcht in beide Richtungen.
SYNONYMS: dict[str, list[str]] = {
    "excel": [
        "tabellenkalkulation", "ms excel", "microsoft excel", "spreadsheet",
        "ms office", "microsoft office", "office",
    ],
    "ms office": ["microsoft office", "office", "office-paket", "office paket"],
    "office": ["ms office", "microsoft office", "office-paket"],
    "deutsch": ["deutschkenntnisse", "german", "muttersprache deutsch"],
    "englisch": ["englischkenntnisse", "english", "business english"],
    "französisch": ["französischkenntnisse", "french"],
    "spanisch": ["spanischkenntnisse", "spanish"],
    "sql": ["datenbanken", "relationale datenbanken", "mysql", "postgresql", "postgres", "mariadb", "oracle db", "sqlite"],
    "python": ["python3", "py"],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "powerpoint": ["präsentationen", "ms powerpoint", "präsentation"],
    "word": ["ms word", "textverarbeitung"],
    "projektmanagement": ["pm", "project management"],
    "agil": ["agile", "scrum", "kanban"],
    "scrum": ["agil", "agile"],
    "buchhaltung": ["accounting", "rechnungswesen"],
    "marketing": ["online-marketing", "digital marketing"],
    "seo": ["search engine optimization", "suchmaschinenoptimierung"],
    "sea": ["search engine advertising", "google ads", "google adwords"],
    "google analytics": ["ga4", "web analytics"],
    "crm": ["customer relationship management", "salesforce", "hubspot"],
    "erp": ["sap", "navision", "dynamics"],
    "kommunikation": ["kommunikationsstärke", "kommunikationsfähigkeit"],
    "teamarbeit": ["teamfähigkeit", "teamplayer"],
}


def _synonyms_of(term: str) -> set[str]:
    """Gibt alle bekannten Synonyme eines Begriffs zurück (bidirektional)."""
    t = term.strip().lower()
    syns: set[str] = set()
    if t in SYNONYMS:
        syns.update(s.lower() for s in SYNONYMS[t])
    for key, vals in SYNONYMS.items():
        if t in (v.lower() for v in vals):
            syns.add(key.lower())
            syns.update(v.lower() for v in vals if v.lower() != t)
    syns.discard(t)
    return syns


def check_criterion(criterion: str, cv_data: dict) -> dict:
    """Prüft nutzerfreundlich, ob ein Kriterium im CV vorkommt.

    Liefert {"status": "vorhanden" | "teilweise_vorhanden" | "unklar" |
             "nicht_gefunden", "evidence": str | None, "note": str | None}.

    Die Prüfung ist großzügig: direkte Treffer zählen voll, Synonym-Treffer
    und Teiltreffer als "teilweise vorhanden". Keine Bewertung der Person,
    keine Eignungsaussage, keine Gewichtung der Person.
    """
    crit = criterion.strip()
    if not crit:
        return {"status": "nicht_gefunden", "evidence": None, "note": None}

    haystack = _searchable_text(cv_data).lower()
    crit_low = crit.lower()

    # 1) Direkter Volltreffer → vorhanden
    if crit_low in haystack:
        return {
            "status": "vorhanden",
            "evidence": _evidence_for(crit_low, cv_data),
            "note": None,
        }

    # 2) Synonym-Treffer für den vollen Begriff → teilweise vorhanden
    for syn in _synonyms_of(crit_low):
        if syn and syn in haystack:
            return {
                "status": "teilweise_vorhanden",
                "evidence": _evidence_for(syn, cv_data),
                "note": f"Sinngemäß über „{syn}“ erkannt.",
            }

    # 3) Token-basierter Teil-Match (auch mit Synonymen je Token)
    tokens = [t for t in crit_low.split() if len(t) >= 2]
    matched_tokens: list[str] = []
    for t in tokens:
        if t in haystack:
            matched_tokens.append(t)
            continue
        if any(s in haystack for s in _synonyms_of(t) if s):
            matched_tokens.append(t)
    if tokens and matched_tokens:
        if len(matched_tokens) == len(tokens):
            # alle Tokens (per Wort oder Synonym) gefunden
            return {
                "status": "teilweise_vorhanden",
                "evidence": _evidence_for(matched_tokens[0], cv_data),
                "note": "Sinngemäß erkannt (Synonym/Teiltreffer).",
            }
        missing = [t for t in tokens if t not in matched_tokens]
        return {
            "status": "teilweise_vorhanden",
            "evidence": _evidence_for(matched_tokens[0], cv_data),
            "note": (
                f"Teiltreffer ({', '.join(matched_tokens)}); "
                f"nicht eindeutig: {', '.join(missing)}."
            ),
        }

    # 4) Datenqualitäts-Hinweis betrifft das Kriterium → unklar
    issues_blob = " ".join(cv_data.get("data_quality_issues", [])).lower()
    for token in tokens or [crit_low]:
        if token and token in issues_blob:
            return {
                "status": "unklar",
                "evidence": None,
                "note": "Datenqualitäts-Hinweis betrifft dieses Kriterium.",
            }

    return {"status": "nicht_gefunden", "evidence": None, "note": None}


STATUS_BADGE = {
    "vorhanden": ("kmu-badge-ok", "vorhanden"),
    "teilweise_vorhanden": ("kmu-badge-warn", "teilweise vorhanden"),
    "unklar": ("kmu-badge-info", "unklar"),
    "nicht_gefunden": ("kmu-badge-err", "nicht gefunden"),
}


def _status_badge_html(status: str) -> str:
    cls, label = STATUS_BADGE.get(status, ("kmu-badge-err", "nicht gefunden"))
    return f'<span class="kmu-badge {cls}">{label}</span>'


def render_checklist(job: dict, cv_data: dict) -> None:
    """Zeigt eine neutrale Kriterien-Checkliste pro Bewerber.

    Keine Bewertung der Person — nur 4 sachliche Status:
    vorhanden / teilweise vorhanden / unklar / nicht gefunden.
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
            badge = _status_badge_html(res["status"])
            st.markdown(
                f'<div style="margin:4px 0">• {c} &nbsp; {badge}</div>',
                unsafe_allow_html=True,
            )
    if not any_rendered:
        st.write("Stellenprofil enthält keine prüfbaren Kriterien.")


def calculate_requirement_coverage(
    job_profile: dict | None, candidate_data: dict
) -> dict:
    """Regelbasierter Abdeckungsgrad der Anforderungen.

    Berechnet: erfüllte objektiv gefundene Kriterien / insgesamt prüfbare
    Kriterien * 100. Geprüft werden nur objektiv prüfbare Kriterien:
    gewünschte Skills, Sprachen, Zertifikate und explizit genannte
    Berufserfahrung. Die Bewertung erfolgt regelbasiert über
    check_criterion() — KEINE freie LLM-Bewertung.

    WICHTIG: Dies ist KEIN Score, KEIN Ranking, KEINE Eignungsaussage und
    KEINE Entscheidungsgrundlage — nur ein transparenter Abdeckungsgrad der
    objektiv im Lebenslauf gefundenen Anforderungen. Unklare Kriterien
    zählen NICHT als erfüllt und werden zusätzlich als Informationslücke
    zurückgegeben.

    Rückgabe:
        {"computed": bool, "percent": int | None, "fulfilled": int,
         "total": int, "details": [{criterion, status}], "unclear": [..]}
    """
    empty = {
        "computed": False,
        "percent": None,
        "fulfilled": 0,
        "total": 0,
        "details": [],
        "unclear": [],
    }
    if not job_profile:
        return empty

    checkable: list[str] = []
    for key in (
        "desired_skills",
        "desired_languages",
        "desired_certificates",
        "desired_experience",
    ):
        for crit in job_profile.get(key, []) or []:
            if crit and crit.strip():
                checkable.append(crit.strip())

    # Duplikate entfernen, Reihenfolge erhalten
    seen: set[str] = set()
    items: list[str] = []
    for c in checkable:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            items.append(c)

    if not items:
        return empty

    fulfilled = 0.0
    details: list[dict] = []
    unclear: list[str] = []
    partial: list[str] = []
    for c in items:
        status = check_criterion(c, candidate_data)["status"]
        # Großzügige Wertung: vorhanden zählt voll, teilweise zählt 0.75,
        # unklar/nicht_gefunden zählen nicht.
        if status == "vorhanden":
            fulfilled += 1.0
        elif status == "teilweise_vorhanden":
            fulfilled += 0.75
            partial.append(c)
        elif status == "unklar":
            unclear.append(c)
        details.append({"criterion": c, "status": status})

    percent = round(fulfilled / len(items) * 100)
    return {
        "computed": True,
        "percent": percent,
        "fulfilled": round(fulfilled, 2),
        "total": len(items),
        "details": details,
        "unclear": unclear,
        "partial": partial,
    }


# ---------------------------------------------------------------------------
# Qualifikationsabdeckung — nur leistungsbezogene Anforderungen
# ---------------------------------------------------------------------------

# Schlüsselwörter, die nicht-leistungsbezogen sind und aus der
# Qualifikationsabdeckung ausgeschlossen werden müssen. Wenn ein
# Anforderungstext eines dieser Wörter enthält, fließt er NICHT in die
# Qualifikationsabdeckung ein (er kann separat als organisatorischer
# Hinweis angezeigt werden).
NON_PERFORMANCE_KEYWORDS: tuple[str, ...] = (
    "gehalt", "gehaltsvorstellung", "vergütung", "verguetung", "lohn",
    "arbeitszeit", "vollzeit", "teilzeit", "stundenmodell",
    "remote", "homeoffice", "home office", "home-office", "mobiles arbeiten",
    "standort", "umzug", "umzugsbereitschaft", "wohnort",
    "startdatum", "verfügbarkeit", "verfuegbarkeit", "eintrittstermin",
    "kündigungsfrist", "kuendigungsfrist",
    "referenz", "referenzen",
    "anschreiben", "bewerbungsfoto", "foto", "lichtbild",
    "alter", "geburtsdatum",
    "geschlecht", "gender",
    "herkunft", "nationalität", "nationalitaet", "staatsangehörigkeit",
    "name ", "adresse", "anschrift", "telefon", "e-mail", "email",
    "familienstand", "familie", "kinder",
    "motivation", "persönlichkeit", "persoenlichkeit",
    "kulturelle passung", "cultural fit", "sympathie", "team-fit",
)

# Leistungsbezogene Kategorien (positive Liste für die Begründung).
PERFORMANCE_CATEGORIES: dict[str, str] = {
    "desired_skills": "Skill",
    "desired_languages": "Sprache",
    "desired_certificates": "Zertifikat",
    "desired_experience": "Berufserfahrung",
}


def is_performance_criterion(text: str) -> bool:
    """Filter: True, wenn die Anforderung leistungsbezogen ist.

    Schließt nicht-leistungsbezogene Angaben (Gehalt, Arbeitszeit,
    Standort, Persönlichkeit, Sympathie, …) aus.
    """
    if not text or not text.strip():
        return False
    low = text.lower()
    for kw in NON_PERFORMANCE_KEYWORDS:
        if kw in low:
            return False
    return True


# Soft-Skill / nicht-automatisch-prüfbar — wenn ein Stichwort auftaucht,
# wird die Anforderung NICHT in den fachlichen Abgleich aufgenommen.
SOFT_REQUIREMENT_KEYWORDS: tuple[str, ...] = (
    "kreativität", "kreativ",
    "offenheit", "offen für",
    "motivation", "motiviert", "motivierten",
    "teamfähig", "teamplayer", "teamarbeit",
    "kommunikationsstärke", "kommunikativ", "kommunikationsfähig",
    "belastbar", "belastbarkeit",
    "eigeninitiative", "eigenverantwortlich", "eigenverantwortung",
    "lernbereitschaft", "lernbereit",
    "kulturelle passung", "cultural fit", "team-fit",
    "persönlichkeit", "persoenlichkeit",
    "mindset",
    "leidenschaft", "passion", "passioniert", "begeistert",
    "flexibilität", "flexibel",
    "technische affinität", "tech-affinität",
    "selbstständige arbeitsweise", "selbständige arbeitsweise",
    "selbstständig", "selbständig",
    "proaktiv", "proaktives denken",
    "innovationsfähigkeit", "innovativ",
    "hands-on", "hands on",
    "soft skill", "soft-skill", "soft skills",
    "engagement", "engagiert",
    "zuverlässig", "zuverlässigkeit",
    "verantwortungsbewusst", "verantwortungsvoll",
    "lösungsorientiert", "loesungsorientiert",
    "kundenorientiert", "serviceorientiert",
    "freundlich", "sympathisch",
)


def is_objectively_checkable(text: str) -> bool:
    """True, wenn die Anforderung objektiv aus einem CV prüfbar ist.

    Schließt organisatorische Angaben (Gehalt, Arbeitszeit, Standort …)
    UND weiche/nicht-messbare Anforderungen (Kreativität, Teamfähigkeit,
    Motivation, Persönlichkeit, technische Affinität …) aus.
    """
    if not text or not text.strip():
        return False
    if not is_performance_criterion(text):
        return False
    low = text.lower()
    for kw in SOFT_REQUIREMENT_KEYWORDS:
        if kw in low:
            return False
    return True


def filter_objectively_checkable_requirements(
    items: list[str],
) -> tuple[list[str], list[str]]:
    """Partitioniert eine Liste in (prüfbar, nicht_prüfbar)."""
    checkable: list[str] = []
    not_checkable: list[str] = []
    for it in items or []:
        text = (it or "").strip()
        if not text:
            continue
        if is_objectively_checkable(text):
            checkable.append(text)
        else:
            not_checkable.append(text)
    return checkable, not_checkable


def _collect_all_requirements(job_profile: dict | None) -> list[str]:
    """Sammelt alle Anforderungstexte aus dem Stellenprofil (dedupliziert)."""
    if not job_profile:
        return []
    items: list[str] = []
    for key in (
        "must_criteria",
        "nice_criteria",
        "desired_skills",
        "desired_languages",
        "desired_certificates",
        "desired_experience",
    ):
        for it in job_profile.get(key) or []:
            t = (it or "").strip()
            if t:
                items.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        k = it.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def get_not_checkable_requirements(job_profile: dict | None) -> list[str]:
    """Liefert nur die nicht automatisch prüfbaren Anforderungen.

    Bevorzugt das explizite Feld non_checkable_requirements (das der
    Stellenprofil-Agent sauber befüllt). Fallback: durchsucht die alten
    fachlichen Felder eines Profils, das ohne dieses Feld erstellt wurde.
    """
    if not job_profile:
        return []
    explicit = job_profile.get("non_checkable_requirements") or []
    if explicit:
        # Dedup case-insensitive, Reihenfolge erhalten.
        seen: set[str] = set()
        out: list[str] = []
        for it in explicit:
            t = (it or "").strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out
    items = _collect_all_requirements(job_profile)
    _, not_checkable = filter_objectively_checkable_requirements(items)
    return not_checkable


# Kompatibilitäts-Alias: extract_job_requirements = analyze_job_profile
extract_job_requirements = analyze_job_profile


# Übersetzung interner Status → menschlich
QUALIFICATION_STATUS_LABEL = {
    "vorhanden": "Erfüllt",
    "teilweise_vorhanden": "Teilweise erfüllt",
    "nicht_gefunden": "Nicht gefunden",
    "unklar": "Unklar",
}


def _qualification_reason(criterion: str, res: dict, category: str) -> str:
    """Liefert eine nachvollziehbare Begründung für den Status."""
    status = res["status"]
    note = (res.get("note") or "").strip()
    if status == "vorhanden":
        return f"{category} „{criterion}“ im Lebenslauf gefunden."
    if status == "teilweise_vorhanden":
        return note or "Ähnliche Qualifikation erkannt."
    if status == "unklar":
        return note or "Angabe vorhanden, aber unklar (z. B. Niveau fehlt)."
    return "Kein Hinweis im Lebenslauf gefunden."


def build_qualification_coverage(
    job_profile: dict | None, cv_data: dict
) -> dict:
    """Baut die Qualifikationsabdeckung — nur leistungsbezogene Anforderungen.

    Nutzt die bereits vorhandene check_criterion()-Logik und filtert
    nicht-leistungsbezogene Anforderungen über is_performance_criterion()
    aus. Liefert:
        {"computed": bool, "fulfilled": int, "total": int,
         "rows": [{requirement, status, status_label, reason, category}],
         "excluded": [{requirement, reason}]}
    Erfüllt = status "vorhanden". Teilweise/Unklar/Nicht gefunden zählen
    NICHT als erfüllt im Zähler "x von y". Keine Bewertung der Person,
    keine Sortierung, keine Empfehlung.
    """
    empty = {
        "computed": False,
        "fulfilled": 0,
        "total": 0,
        "rows": [],
        "excluded": [],
    }
    if not job_profile:
        return empty

    rows: list[dict] = []
    excluded: list[dict] = []
    seen: set[str] = set()
    for key, category in PERFORMANCE_CATEGORIES.items():
        for crit in job_profile.get(key, []) or []:
            text = (crit or "").strip()
            if not text:
                continue
            k = text.lower()
            if k in seen:
                continue
            seen.add(k)
            if not is_performance_criterion(text):
                excluded.append(
                    {
                        "requirement": text,
                        "reason": "nicht leistungsbezogen (organisatorisch).",
                    }
                )
                continue
            res = check_criterion(text, cv_data)
            rows.append(
                {
                    "requirement": text,
                    "status": res["status"],
                    "status_label": QUALIFICATION_STATUS_LABEL.get(
                        res["status"], res["status"]
                    ),
                    "reason": _qualification_reason(text, res, category),
                    "category": category,
                }
            )

    if not rows and not excluded:
        return empty

    fulfilled = sum(1 for r in rows if r["status"] == "vorhanden")
    return {
        "computed": True,
        "fulfilled": fulfilled,
        "total": len(rows),
        "rows": rows,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Stellenprofil-Checkliste pro Bewerber
# ---------------------------------------------------------------------------

import re as _re_eval

LEVEL_RANKING = {"a1": 1, "a2": 2, "b1": 3, "b2": 4, "c1": 5, "c2": 6}
NATIVE_KEYWORDS = (
    "muttersprache",
    "muttersprachler",
    "muttersprachlerin",
    "native speaker",
    "native",
    "mother tongue",
)
KNOWN_LANGUAGES = {
    "deutsch": "Deutsch",
    "german": "Deutsch",
    "englisch": "Englisch",
    "english": "Englisch",
    "französisch": "Französisch",
    "franzoesisch": "Französisch",
    "french": "Französisch",
    "spanisch": "Spanisch",
    "spanish": "Spanisch",
    "italienisch": "Italienisch",
    "italian": "Italienisch",
}

STATUS_ICON = {
    "vorhanden": "✅",
    "teilweise_vorhanden": "🟡",
    "nicht_gefunden": "❌",
    "unklar": "❓",
}
STATUS_HUMAN = {
    "vorhanden": "Vorhanden",
    "teilweise_vorhanden": "Teilweise vorhanden",
    "nicht_gefunden": "Nicht gefunden",
    "unklar": "Unklar",
}


def _extract_ger_level(text: str) -> str | None:
    if not text:
        return None
    m = _re_eval.search(r"\b([abc][12])\b", text.lower())
    return m.group(1) if m else None


def _is_native(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in NATIVE_KEYWORDS)


def _detect_language_in_criterion(crit_low: str) -> str | None:
    """Liefert die kanonische Sprache (z. B. 'Deutsch'), wenn das Kriterium
    auf eine bekannte Sprache verweist."""
    for trigger, canonical in KNOWN_LANGUAGES.items():
        if _re_eval.search(rf"\b{trigger}\b", crit_low):
            return canonical
    return None


def _find_language_in_cv(canonical: str, cv_data: dict) -> dict | None:
    """Sucht in cv_data.languages einen passenden Eintrag (bidirektional)."""
    triggers = {t for t, c in KNOWN_LANGUAGES.items() if c == canonical}
    triggers.add(canonical.lower())
    for entry in cv_data.get("languages") or []:
        name = (entry.get("language") or "").lower()
        if any(t in name for t in triggers):
            return entry
    return None


def _evaluate_language_requirement(
    criterion: str, cv_data: dict
) -> dict | None:
    """Sprach-spezifische Bewertung. Liefert {status, reason} oder None.

    Erkennt Muttersprache (auch native/native speaker) und vergleicht
    GER-Niveaus (A1–C2) numerisch.
    """
    crit_low = criterion.lower()
    canonical = _detect_language_in_criterion(crit_low)
    if not canonical:
        return None
    requested_level = _extract_ger_level(crit_low)
    cv_entry = _find_language_in_cv(canonical, cv_data)
    if not cv_entry:
        return {
            "status": "nicht_gefunden",
            "reason": f"{canonical} nicht im Lebenslauf angegeben.",
        }
    level_text = (cv_entry.get("level") or "")
    name_text = (cv_entry.get("language") or "")
    if _is_native(level_text) or _is_native(name_text):
        return {
            "status": "vorhanden",
            "reason": (
                f"{canonical} als Muttersprache angegeben — "
                "erfüllt jedes geforderte Niveau."
            ),
        }
    cv_level = _extract_ger_level(level_text)
    if requested_level:
        if cv_level:
            if LEVEL_RANKING.get(cv_level, 0) >= LEVEL_RANKING.get(
                requested_level, 0
            ):
                return {
                    "status": "vorhanden",
                    "reason": (
                        f"{canonical} {cv_level.upper()} im Lebenslauf "
                        f"genannt — erreicht das geforderte Niveau "
                        f"{requested_level.upper()}."
                    ),
                }
            return {
                "status": "teilweise_vorhanden",
                "reason": (
                    f"{canonical} {cv_level.upper()} im Lebenslauf — "
                    f"gefordert war {requested_level.upper()}."
                ),
            }
        return {
            "status": "unklar",
            "reason": (
                f"{canonical} genannt, Niveau nicht angegeben "
                f"(gefordert: {requested_level.upper()})."
            ),
        }
    # Kein Niveau gefordert — Sprache reicht.
    return {
        "status": "vorhanden",
        "reason": f"{canonical} im Lebenslauf angegeben.",
    }


def evaluate_candidate_requirements(
    job_profile: dict | None, candidate_data: dict
) -> list[dict]:
    """Zentraler Abgleich der **fachlich prüfbaren** Anforderungen.

    Nicht-automatisch-prüfbare Anforderungen (Kreativität, Teamfähigkeit,
    Motivation, …) und organisatorische Angaben (Gehalt, Standort, …)
    werden über filter_objectively_checkable_requirements() ausgeschlossen
    und nicht in dieser Liste zurückgegeben — sie sind über
    get_not_checkable_requirements(job_profile) abrufbar.

    Liefert pro prüfbarer Anforderung:
        {"requirement", "status", "reason", "evidence"}
    Status-Werte (UI-Schreibweise):
        "Gefunden" · "Teilweise gefunden" · "Nicht gefunden"
    """
    if not job_profile:
        return []
    all_items = _collect_all_requirements(job_profile)
    checkable, _ = filter_objectively_checkable_requirements(all_items)
    rows: list[dict] = []
    for text in checkable:
        # Sprach-spezifische Bewertung zuerst (Muttersprache, GER-Niveau)
        lang_res = _evaluate_language_requirement(text, candidate_data)
        if lang_res is not None:
            internal = lang_res["status"]
            reason = lang_res["reason"]
        else:
            res = check_criterion(text, candidate_data)
            internal = res["status"]
            note = (res.get("note") or "").strip()
            if internal == "vorhanden":
                reason = "Im Lebenslauf gefunden."
            elif internal == "teilweise_vorhanden":
                reason = note or "Ähnliche Qualifikation erkannt."
            elif internal == "unklar":
                reason = note or "Angabe vorhanden, Niveau/Detail unklar."
            else:
                reason = "Kein Nachweis im Lebenslauf gefunden."
        # UI-Status: nur drei Werte; "unklar" → "Teilweise gefunden".
        if internal == "vorhanden":
            label = "Gefunden"
        elif internal == "nicht_gefunden":
            label = "Nicht gefunden"
        else:
            label = "Teilweise gefunden"
        evidence = find_source_excerpt(text, candidate_data) or ""
        rows.append(
            {
                "requirement": text,
                "status": label,
                "reason": reason,
                "evidence": evidence,
            }
        )
    return rows


def status_counts(rows: list[dict]) -> dict[str, int]:
    """Zählt die UI-Status in einer evaluate_candidate_requirements-Liste."""
    counts = {
        "Gefunden": 0,
        "Teilweise gefunden": 0,
        "Nicht gefunden": 0,
    }
    for r in rows:
        s = r.get("status")
        if s in counts:
            counts[s] += 1
    return counts


def log_coverage_once(candidate: dict, coverage: dict, job_profile: dict | None) -> None:
    """Loggt 'Abdeckungsgrad berechnet' genau einmal pro Kandidat/Profilstand."""
    if not coverage.get("computed"):
        return
    role = (job_profile or {}).get("role", "")
    signature = (
        f"{candidate.get('filename', '')}|{role}|"
        f"{coverage['fulfilled']}/{coverage['total']}"
    )
    logged = st.session_state.setdefault("_coverage_logged", set())
    if signature in logged:
        return
    logged.add(signature)
    log_audit(
        action="Abdeckungsgrad berechnet",
        target=candidate["data"].get("name", "") or candidate["filename"],
        result_type=f"{coverage['percent']} %",
    )


def render_cv_summary(cv_data: dict) -> None:
    """Menschenlesbare CV-Zusammenfassung mit ‚nicht gefunden‘-Markern."""
    name = cv_data.get("name") or NICHT_GEFUNDEN
    st.markdown(f"**Name:** {name}")

    skills = cv_data.get("skills", [])
    st.markdown(
        f"**Skills:** {', '.join(skills) if skills else NICHT_GEFUNDEN}"
    )

    experience = cv_data.get("experience", [])
    if experience:
        st.markdown("**Berufserfahrung:**")
        for e in experience:
            line = (
                f"- {e.get('role') or NICHT_GEFUNDEN} @ "
                f"{e.get('company') or NICHT_GEFUNDEN} "
                f"({e.get('period') or NICHT_GEFUNDEN})"
            )
            desc = e.get("description")
            if desc:
                line += f" — {desc}"
            st.markdown(line)
    else:
        st.markdown(f"**Berufserfahrung:** {NICHT_GEFUNDEN}")

    education = cv_data.get("education", [])
    if education:
        st.markdown("**Ausbildung:**")
        for e in education:
            st.markdown(
                f"- {e.get('degree') or NICHT_GEFUNDEN}, "
                f"{e.get('institution') or NICHT_GEFUNDEN} "
                f"({e.get('period') or NICHT_GEFUNDEN})"
            )
    else:
        st.markdown(f"**Ausbildung:** {NICHT_GEFUNDEN}")

    certificates = cv_data.get("certificates", [])
    st.markdown(
        f"**Zertifikate:** "
        f"{', '.join(certificates) if certificates else NICHT_GEFUNDEN}"
    )

    languages = cv_data.get("languages", [])
    if languages:
        lang_str = ", ".join(
            f"{l.get('language', '')} ({l.get('level') or NICHT_GEFUNDEN})"
            for l in languages
        )
        st.markdown(f"**Sprachen:** {lang_str}")
    else:
        st.markdown(f"**Sprachen:** {NICHT_GEFUNDEN}")


# ---------------------------------------------------------------------------
# Streamlit-UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recruiting AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Design-System (Premium-SaaS): Tokens, Glasflächen, Animationen, SVG-Icons
# ---------------------------------------------------------------------------
#
# Informationsarchitektur (vom Marketing-Stil zum Produkt-Stil reduziert):
#
#   Top-Nav  ─►  Dashboard  · Recruiting · Kandidaten · Audit Log
#   User-Menü (Dropdown) ─►  Einstellungen · Hilfe · Sign out
#
#   Dashboard       Überblick, KPIs, Workflow-Status, AI Activity Feed,
#                   "Letzte Bewerbungen" (read-only)
#   Recruiting      Stellenprofil-Eingabe & Analyse | Bewerbungen-Upload
#                   & Verarbeitung. Strukturiertes Stellenprofil unten.
#   Kandidaten      Filter, Karten, Profil-Detail mit Anforderungs-
#                   abgleich, Belegen, Rückfragen und Feedback inline.
#   Audit Log       Timeline der Agentenschritte, Feedback-Übersicht,
#                   Tabellen-Ansicht.
#
# Designprinzipien: viel Weißraum · klare Hierarchie · große Typo · keine
# Dashboard-Überladung · alle Icons als Inline-SVG (keine Emoji) · keine
# Inline-styles (Utility-/Komponenten-Klassen) · markenspezifische
# Gradients (Lila→Indigo→Blau, Indigo→Türkis) · dezente Glow-Orbs als
# Hintergrund · Hover-Lift + Glow · CSS-Scroll-Reveal (animation-timeline:
# view()), mit fallback-fade-up beim ersten Render.
# ---------------------------------------------------------------------------


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Inter+Tight:wght@500;600;700;800&display=swap');

    :root{
      --ink:#0B1020;
      --ink-2:#1F2937;
      --body:#334155;
      --muted:#64748B;
      --soft:#94A3B8;

      --bg:#FAFAFE;
      --bg-2:#F4F4FB;
      --surface:rgba(255,255,255,0.72);
      --surface-2:rgba(255,255,255,0.55);
      --surface-strong:rgba(255,255,255,0.92);
      --line:rgba(15,23,42,0.07);
      --line-2:rgba(15,23,42,0.05);
      --line-3:rgba(15,23,42,0.10);

      --purple:#6F6EFF;
      --purple-2:#8E8CFF;
      --indigo:#4F46E5;
      --indigo-2:#6366F1;
      --blue:#2563EB;
      --teal:#14B8A6;

      --ok:#16A34A;
      --ok-bg:#E8F7EE;
      --warn:#B45309;
      --warn-bg:#FDF1DA;
      --err:#B91C1C;
      --err-bg:#FCE8E8;
      --info:#1D4ED8;
      --info-bg:#EAF0FE;

      --grad:linear-gradient(135deg,#6F6EFF 0%,#4F46E5 55%,#2563EB 100%);
      --grad-soft:linear-gradient(135deg,rgba(111,110,255,0.18) 0%,rgba(79,70,229,0.14) 50%,rgba(20,184,166,0.12) 100%);
      --grad-text:linear-gradient(120deg,#4F46E5 0%,#6F6EFF 45%,#14B8A6 100%);

      --shadow-1:0 1px 2px rgba(15,23,42,0.04), 0 1px 1px rgba(15,23,42,0.03);
      --shadow-2:0 8px 24px rgba(15,23,42,0.06), 0 2px 6px rgba(15,23,42,0.04);
      --shadow-3:0 22px 44px rgba(15,23,42,0.08), 0 4px 12px rgba(15,23,42,0.05);
      --shadow-glow:0 0 0 1px rgba(79,70,229,0.18), 0 16px 40px rgba(79,70,229,0.20);

      --radius-sm:10px;
      --radius:14px;
      --radius-lg:20px;
      --radius-xl:24px;

      --sans:'Inter','Inter Tight',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      --display:'Inter Tight','Inter',-apple-system,sans-serif;
    }

    html, body, [class*="css"], .stApp,
    p, span, label, li, div, input, textarea, button {
      font-family: var(--sans);
      font-feature-settings:'cv11','ss01','ss03','cv02';
    }
    .stApp { color: var(--ink); background: transparent; }

    /* ---------- Animated background (orbs + grain) ---------- */
    [data-testid="stAppViewContainer"] {
      position: relative;
      background-color: var(--bg);
      background-image:
        radial-gradient(900px 700px at 8% -6%, rgba(111,110,255,0.18), transparent 60%),
        radial-gradient(720px 560px at 96% 4%, rgba(79,70,229,0.16), transparent 62%),
        radial-gradient(700px 620px at 86% 92%, rgba(20,184,166,0.16), transparent 62%),
        radial-gradient(820px 680px at -2% 96%, rgba(37,99,235,0.14), transparent 60%);
      background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
      content:""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      opacity:.04;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }
    [data-testid="stMain"] { background: transparent; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.4rem; padding-bottom: 5rem; max-width: 1200px; position: relative; z-index: 1; }

    /* slow drifting orbs as decoration on the body */
    .bg-fx { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
    .bg-fx .orb { position: absolute; border-radius: 50%; filter: blur(80px); opacity: .55; mix-blend-mode: normal; will-change: transform; }
    .bg-fx .orb.a { width: 520px; height: 520px; left: -80px; top: -120px; background: radial-gradient(closest-side, rgba(111,110,255,0.55), transparent 70%); animation: floatA 26s ease-in-out infinite alternate; }
    .bg-fx .orb.b { width: 560px; height: 560px; right: -120px; top: 12%; background: radial-gradient(closest-side, rgba(79,70,229,0.40), transparent 70%); animation: floatB 30s ease-in-out infinite alternate; }
    .bg-fx .orb.c { width: 460px; height: 460px; left: 30%; bottom: -160px; background: radial-gradient(closest-side, rgba(20,184,166,0.38), transparent 70%); animation: floatC 34s ease-in-out infinite alternate; }
    @keyframes floatA { from{transform:translate3d(0,0,0) scale(1);} to{transform:translate3d(60px,30px,0) scale(1.05);} }
    @keyframes floatB { from{transform:translate3d(0,0,0) scale(1);} to{transform:translate3d(-50px,40px,0) scale(0.96);} }
    @keyframes floatC { from{transform:translate3d(0,0,0) scale(1);} to{transform:translate3d(20px,-30px,0) scale(1.08);} }

    /* ---------- Headings & text ---------- */
    h1, h2, h3, h4, h5, h6 {
      color: var(--ink) !important; font-family: var(--display);
      letter-spacing:-0.018em; font-weight: 700;
    }
    h1 { font-size: 40px; line-height: 1.08; }
    h2 { font-size: 26px; line-height: 1.18; }
    h3 { font-size: 19px; line-height: 1.25; }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size: 13px; }
    p, span, label, li { color: var(--body); }
    a { color: var(--indigo); text-decoration: none; }
    .stApp hr { border: none; border-top: 1px solid var(--line); }

    /* ---------- Sidebar (collapsed by default, keeps minimal commands) ---------- */
    [data-testid="stSidebar"] { background: var(--surface-strong) !important; border-right: 1px solid var(--line); backdrop-filter: blur(16px); }
    [data-testid="stSidebar"] * { color: var(--ink); }

    /* ---------- Inputs ---------- */
    .stTextInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
      background: var(--surface-strong) !important;
      color: var(--ink) !important;
      border: 1px solid var(--line-3) !important;
      border-radius: var(--radius-sm) !important;
      font-family: var(--sans) !important;
      box-shadow: var(--shadow-1) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
      border-color: var(--indigo) !important;
      box-shadow: 0 0 0 4px rgba(79,70,229,0.14) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
      background: rgba(255,255,255,0.62) !important;
      border: 1.4px dashed rgba(15,23,42,0.18) !important;
      color: var(--muted) !important;
      border-radius: var(--radius) !important;
      backdrop-filter: blur(10px);
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color: var(--indigo) !important;
      background: rgba(255,255,255,0.82) !important;
    }

    /* ---------- Buttons (premium) ---------- */
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
      font-family: var(--sans) !important; font-weight: 600 !important;
      border-radius: 12px !important;
      padding: 10px 18px !important;
      transition: transform .15s ease, box-shadow .2s ease, background .2s ease, border-color .2s ease, color .2s ease;
      letter-spacing: 0;
    }
    .stButton > button {
      background: var(--surface-strong) !important;
      color: var(--ink) !important;
      border: 1px solid var(--line-3) !important;
      box-shadow: var(--shadow-1) !important;
    }
    .stButton > button:hover {
      transform: translateY(-1px);
      border-color: rgba(79,70,229,0.40) !important;
      box-shadow: 0 8px 22px rgba(15,23,42,0.08), 0 0 0 4px rgba(79,70,229,0.10) !important;
    }
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stFormSubmitButton"] > button,
    .stDownloadButton > button {
      background: var(--grad) !important;
      background-size: 200% 200% !important;
      color: #FFFFFF !important;
      border: 1px solid rgba(255,255,255,0.18) !important;
      box-shadow: 0 1px 0 rgba(255,255,255,0.20) inset, 0 10px 24px rgba(79,70,229,0.30) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
      transform: translateY(-1px);
      background-position: 100% 0 !important;
      box-shadow: 0 16px 34px rgba(79,70,229,0.42), 0 0 0 4px rgba(79,70,229,0.18) !important;
    }
    .stButton > button:disabled { opacity: .50 !important; transform: none !important; box-shadow: none !important; }
    [data-testid="stSpinner"] { color: var(--indigo); }

    /* ---------- Tabs ---------- */
    [data-testid="stTabs"] [role="tablist"] {
      background: var(--surface);
      backdrop-filter: blur(20px);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 6px;
      gap: 4px;
      flex-wrap: wrap;
      box-shadow: var(--shadow-1);
    }
    [data-testid="stTabs"] [role="tab"] {
      color: var(--muted) !important; background: transparent !important;
      border-radius: 10px !important; padding: 8px 16px !important; font-weight: 600 !important; font-size: 14px;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
      background: var(--surface-strong) !important;
      color: var(--ink) !important;
      box-shadow: var(--shadow-1);
    }

    /* ---------- Cards / expanders / dataframe (Glas) ---------- */
    [data-testid="stExpander"],
    [data-testid="stDataFrame"],
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--surface) !important;
      backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--line) !important;
      border-radius: var(--radius-lg) !important;
      box-shadow: var(--shadow-2);
      transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-3);
      border-color: rgba(79,70,229,0.22) !important;
    }
    [data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600; }
    [data-testid="stDataFrame"] { padding: 6px; }

    /* ========== APP SHELL: Top-Nav ========== */
    .shell-nav {
      display: flex; align-items: center; gap: 18px;
      padding: 10px 14px; margin: 4px 0 20px;
      background: var(--surface);
      backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow-2);
    }
    .shell-brand { display: flex; align-items: center; gap: 11px; padding-left: 6px; }
    .shell-mark {
      width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
      background: var(--grad); color: #fff;
      display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 6px 14px rgba(79,70,229,0.32), inset 0 1px 0 rgba(255,255,255,0.30);
    }
    .shell-mark svg { width: 18px; height: 18px; }
    .shell-name { font-family: var(--display); font-weight: 700; font-size: 16px; color: var(--ink); letter-spacing: -0.01em; line-height: 1; }
    .shell-name small { display: block; font-family: var(--sans); font-weight: 500; font-size: 10.5px; color: var(--muted); letter-spacing: .14em; text-transform: uppercase; margin-top: 3px; }
    .shell-grow { flex: 1; }

    /* The nav buttons themselves are Streamlit buttons - we restyle them
       only inside the .shell-nav container scope via a parent class. */
    .shell-tag {
      display: inline-flex; align-items: center; gap: 6px;
      font-size: 12px; font-weight: 600;
      color: var(--indigo); background: rgba(79,70,229,0.10);
      padding: 4px 10px; border-radius: 999px;
    }
    .shell-tag .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px rgba(22,163,74,0.18); }

    /* nav button restyle (applies only to buttons within the nav columns) */
    .nav-btn-wrap .stButton > button {
      background: transparent !important;
      color: var(--muted) !important;
      border: 1px solid transparent !important;
      box-shadow: none !important;
      padding: 8px 14px !important;
      font-weight: 600 !important;
    }
    .nav-btn-wrap .stButton > button:hover {
      background: rgba(15,23,42,0.04) !important;
      color: var(--ink) !important;
      transform: none;
    }
    .nav-btn-wrap.active .stButton > button {
      background: var(--surface-strong) !important;
      color: var(--ink) !important;
      border: 1px solid var(--line-3) !important;
      box-shadow: var(--shadow-1) !important;
    }

    /* User chip + dropdown popover */
    .user-chip {
      display: inline-flex; align-items: center; gap: 10px;
      padding: 5px 8px 5px 5px; background: var(--surface-strong); border: 1px solid var(--line);
      border-radius: 999px; box-shadow: var(--shadow-1);
    }
    .user-chip .av {
      width: 28px; height: 28px; border-radius: 50%;
      background: var(--grad); color: #fff;
      display: inline-flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 11px; letter-spacing: .04em;
    }
    .user-chip b { color: var(--ink); font-size: 13px; font-weight: 600; }
    .user-chip small { color: var(--muted); font-size: 11px; }

    /* ========== Page header (eyebrow + title + lead) ========== */
    .ph { padding: 12px 0 4px; }
    .ph .eyebrow {
      display: inline-flex; align-items: center; gap: 8px;
      font-size: 11.5px; font-weight: 700; letter-spacing: .18em;
      text-transform: uppercase; color: var(--indigo);
    }
    .ph .eyebrow::before { content:""; width: 22px; height: 1.5px; background: linear-gradient(90deg, var(--indigo), var(--teal)); border-radius: 2px; }
    .ph h1 {
      font-family: var(--display); font-weight: 700; font-size: 44px; line-height: 1.06;
      letter-spacing:-0.02em; color: var(--ink); margin: 12px 0 10px;
    }
    .ph h1 em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .ph .lead { color: var(--muted); font-size: 17px; line-height: 1.55; max-width: 720px; }

    /* ========== Section + utility ========== */
    .sec { margin-top: 44px; }
    .sec-head { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 18px; gap: 18px; }
    .sec-head h2 { font-family: var(--display); font-weight: 700; font-size: 22px; color: var(--ink); margin: 0; }
    .sec-head .sub { color: var(--muted); font-size: 13.5px; margin-top: 4px; }
    .sec-head .right { color: var(--muted); font-size: 12.5px; }

    .row { display: flex; gap: 14px; align-items: center; }
    .grow { flex: 1; }
    .mt-2 { margin-top: 8px; } .mt-3 { margin-top: 14px; } .mt-4 { margin-top: 22px; } .mt-6 { margin-top: 36px; }
    .mb-2 { margin-bottom: 8px; } .mb-3 { margin-bottom: 14px; } .mb-4 { margin-bottom: 22px; }

    /* ========== KPI cards ========== */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
    .kpi {
      position: relative; overflow: hidden;
      background: var(--surface);
      backdrop-filter: blur(22px);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 22px 22px 20px;
      box-shadow: var(--shadow-2);
      transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .kpi:hover { transform: translateY(-4px); box-shadow: var(--shadow-3); border-color: rgba(79,70,229,0.22); }
    .kpi-head { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
    .kpi-ic {
      width: 36px; height: 36px; border-radius: 11px; flex-shrink: 0;
      display: inline-flex; align-items: center; justify-content: center;
      color: #fff;
    }
    .kpi-ic svg { width: 18px; height: 18px; }
    .kpi-ic.purple { background: linear-gradient(135deg,#6F6EFF,#4F46E5); }
    .kpi-ic.indigo { background: linear-gradient(135deg,#4F46E5,#3730A3); }
    .kpi-ic.blue   { background: linear-gradient(135deg,#2563EB,#1E40AF); }
    .kpi-ic.teal   { background: linear-gradient(135deg,#14B8A6,#0D9488); }
    .kpi-label { color: var(--muted); font-size: 12.5px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
    .kpi-value { font-family: var(--display); font-weight: 700; font-size: 38px; color: var(--ink); line-height: 1.05; letter-spacing: -0.02em; }
    .kpi-delta { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12.5px; margin-top: 8px; }
    .kpi-spark { position: absolute; right: 14px; top: 14px; opacity: .8; }

    /* ========== Glass card (reusable) ========== */
    .gcard {
      background: var(--surface);
      backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      padding: 22px 24px;
      box-shadow: var(--shadow-2);
      transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .gcard:hover { transform: translateY(-2px); box-shadow: var(--shadow-3); }
    .gcard-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; gap: 10px; }
    .gcard-title { font-family: var(--display); font-weight: 700; font-size: 16px; color: var(--ink); }
    .gcard-meta { color: var(--muted); font-size: 12.5px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; }
    .gcard-meta .live { width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 3px rgba(20,184,166,0.18); animation: pulseLive 1.8s ease-in-out infinite; }
    @keyframes pulseLive { 0%,100% { box-shadow: 0 0 0 3px rgba(20,184,166,0.18); } 50% { box-shadow: 0 0 0 7px rgba(20,184,166,0); } }

    /* ========== Hero band (Dashboard) ========== */
    .hero {
      position: relative; overflow: hidden;
      background: var(--surface);
      backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      padding: 40px 40px 36px;
      box-shadow: var(--shadow-3);
      margin-bottom: 28px;
    }
    .hero::before {
      content:""; position: absolute; right: -120px; top: -120px;
      width: 380px; height: 380px; border-radius: 50%;
      background: radial-gradient(closest-side, rgba(111,110,255,0.34), transparent 70%);
      filter: blur(10px); pointer-events: none;
    }
    .hero::after {
      content:""; position: absolute; left: -100px; bottom: -120px;
      width: 320px; height: 320px; border-radius: 50%;
      background: radial-gradient(closest-side, rgba(20,184,166,0.28), transparent 70%);
      filter: blur(8px); pointer-events: none;
    }
    .hero-in { position: relative; z-index: 1; max-width: 760px; }
    .hero-eye {
      display: inline-flex; align-items: center; gap: 8px;
      font-size: 12px; font-weight: 600; color: var(--indigo);
      background: rgba(79,70,229,0.08); border: 1px solid rgba(79,70,229,0.18);
      padding: 6px 12px; border-radius: 999px;
    }
    .hero-eye .sp { width: 5px; height: 5px; border-radius: 50%; background: var(--indigo); box-shadow: 0 0 0 3px rgba(79,70,229,0.20); }
    .hero h1 {
      font-family: var(--display); font-weight: 700; font-size: 50px; line-height: 1.04;
      letter-spacing:-0.025em; color: var(--ink); margin: 16px 0 12px;
    }
    .hero h1 em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .hero .sub { color: var(--muted); font-size: 17px; line-height: 1.55; max-width: 620px; }
    .hero-trust { display: flex; gap: 22px; flex-wrap: wrap; margin-top: 22px; }
    .hero-trust span { display: inline-flex; align-items: center; gap: 8px; color: var(--body); font-size: 13.5px; font-weight: 500; }
    .hero-trust .ic { color: var(--teal); }
    .hero-trust .ic svg { width: 16px; height: 16px; }

    /* ========== Workflow visualizer ========== */
    .wf { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
    .wf-step {
      position: relative;
      background: var(--surface);
      backdrop-filter: blur(20px);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px 14px 16px;
      box-shadow: var(--shadow-1);
      transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    }
    .wf-step:hover { transform: translateY(-3px); box-shadow: var(--shadow-2); border-color: rgba(79,70,229,0.25); }
    .wf-step.is-final { border: 1px solid rgba(20,184,166,0.30); background: linear-gradient(180deg, rgba(20,184,166,0.10), var(--surface) 60%); }
    .wf-step::after {
      content:""; position: absolute; right: -8px; top: 50%;
      width: 14px; height: 2px; background: linear-gradient(90deg, rgba(79,70,229,0.45), transparent);
      transform: translateY(-50%);
    }
    .wf-step:last-child::after { content: none; }
    .wf-ic {
      width: 30px; height: 30px; border-radius: 9px; margin-bottom: 10px;
      display: inline-flex; align-items: center; justify-content: center;
      color: var(--indigo); background: rgba(79,70,229,0.10);
    }
    .wf-step.is-final .wf-ic { color: var(--teal); background: rgba(20,184,166,0.12); }
    .wf-ic svg { width: 16px; height: 16px; }
    .wf-step b { display: block; color: var(--ink); font-size: 13.5px; font-weight: 600; margin-bottom: 3px; line-height: 1.2; }
    .wf-step small { color: var(--muted); font-size: 11.5px; line-height: 1.4; display: block; }
    @media (max-width: 1080px) { .wf { grid-template-columns: repeat(3, 1fr); } .wf-step::after { content: none; } }
    @media (max-width: 640px) { .wf { grid-template-columns: repeat(2, 1fr); } }

    /* ========== Activity feed ========== */
    .feed { display: flex; flex-direction: column; }
    .feed-row {
      display: grid; grid-template-columns: 34px 1fr auto;
      align-items: center; gap: 12px; padding: 12px 4px;
      border-bottom: 1px solid var(--line-2);
    }
    .feed-row:last-child { border-bottom: none; }
    .feed-ic {
      width: 34px; height: 34px; border-radius: 10px;
      display: inline-flex; align-items: center; justify-content: center;
      background: rgba(79,70,229,0.08); color: var(--indigo);
    }
    .feed-ic svg { width: 16px; height: 16px; }
    .feed-row.is-success .feed-ic { background: rgba(22,163,74,0.10); color: var(--ok); }
    .feed-row.is-warn .feed-ic { background: rgba(180,83,9,0.10); color: var(--warn); }
    .feed-row.is-err .feed-ic { background: rgba(185,28,28,0.10); color: var(--err); }
    .feed-row.is-teal .feed-ic { background: rgba(20,184,166,0.12); color: var(--teal); }
    .feed-body { min-width: 0; }
    .feed-body b { display: block; color: var(--ink); font-size: 13.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .feed-body small { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
    .feed-time { color: var(--soft); font-size: 11.5px; font-weight: 600; white-space: nowrap; }

    /* ========== Last applicants list ========== */
    .applist { display: flex; flex-direction: column; }
    .app-row {
      display: grid; grid-template-columns: 40px 1fr auto;
      align-items: center; gap: 14px; padding: 14px 4px;
      border-bottom: 1px solid var(--line-2);
    }
    .app-row:last-child { border-bottom: none; }
    .app-av {
      width: 40px; height: 40px; border-radius: 50%;
      background: var(--grad); color: #fff;
      display: inline-flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 13px; letter-spacing: .03em;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.25);
    }
    .app-name { color: var(--ink); font-weight: 600; font-size: 14px; }
    .app-role { color: var(--muted); font-size: 12.5px; margin-top: 2px; }

    /* ========== Badges ========== */
    .badge {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 3px 10px; border-radius: 999px;
      font-size: 11.5px; font-weight: 600;
      border: 1px solid transparent;
    }
    .badge svg { width: 12px; height: 12px; }
    .badge-ok    { background: var(--ok-bg);   color: var(--ok);   border-color: rgba(22,163,74,0.18); }
    .badge-warn  { background: var(--warn-bg); color: var(--warn); border-color: rgba(180,83,9,0.18); }
    .badge-err   { background: var(--err-bg);  color: var(--err);  border-color: rgba(185,28,28,0.18); }
    .badge-info  { background: var(--info-bg); color: var(--info); border-color: rgba(29,78,216,0.18); }
    .badge-muted { background: rgba(15,23,42,0.05); color: var(--muted); border-color: rgba(15,23,42,0.06); }
    /* Backward-compat aliases used inside business code */
    .kmu-badge        { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 600; border: 1px solid transparent; margin: 1px 0; }
    .kmu-badge-ok     { background: var(--ok-bg);   color: var(--ok);   border-color: rgba(22,163,74,0.18); }
    .kmu-badge-warn   { background: var(--warn-bg); color: var(--warn); border-color: rgba(180,83,9,0.18); }
    .kmu-badge-err    { background: var(--err-bg);  color: var(--err);  border-color: rgba(185,28,28,0.18); }
    .kmu-badge-info   { background: var(--info-bg); color: var(--info); border-color: rgba(29,78,216,0.18); }
    .kmu-badge-muted  { background: rgba(15,23,42,0.05); color: var(--muted); border-color: rgba(15,23,42,0.06); }

    /* ========== Candidate cards ========== */
    .cand-row { display: grid; grid-template-columns: minmax(0,2.4fr) minmax(0,1fr) minmax(0,2.4fr) auto; gap: 18px; align-items: center; }
    .cand-id { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .cand-name { color: var(--ink); font-weight: 700; font-size: 15.5px; }
    .cand-role { color: var(--muted); font-size: 13px; margin-top: 2px; }
    .cand-metric { font-family: var(--display); font-weight: 700; font-size: 24px; color: var(--ink); line-height: 1.05; letter-spacing: -0.01em; }
    .cand-metric em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .cand-metric-label { color: var(--soft); font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }
    .cand-skills { display: flex; flex-wrap: wrap; gap: 5px; }

    /* ========== Coverage panel ========== */
    .cov {
      display: flex; align-items: center; gap: 14px;
      background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(20,184,166,0.06));
      border: 1px solid rgba(79,70,229,0.18);
      border-radius: 14px; padding: 14px 18px;
    }
    .cov-num { font-family: var(--display); font-weight: 700; font-size: 28px; color: var(--ink); letter-spacing: -0.02em; line-height: 1; }
    .cov-num em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .cov-label { color: var(--muted); font-size: 12.5px; margin-top: 4px; }
    .cov-bar { flex: 1; height: 6px; background: rgba(15,23,42,0.06); border-radius: 999px; overflow: hidden; }
    .cov-bar > i { display: block; height: 100%; background: var(--grad); border-radius: 999px; transition: width .8s cubic-bezier(.2,.7,.2,1); }

    /* ========== Timeline (audit log) ========== */
    .tl { position: relative; margin: 4px 0; padding-left: 26px; }
    .tl::before { content:""; position: absolute; left: 6px; top: 6px; bottom: 6px; width: 2px; background: linear-gradient(180deg, rgba(79,70,229,0.20), rgba(20,184,166,0.10)); }
    .tl-item { position: relative; padding: 0 0 18px 6px; }
    .tl-dot {
      position: absolute; left: -22px; top: 6px;
      width: 12px; height: 12px; border-radius: 50%;
      background: var(--surface-strong); border: 2px solid var(--indigo);
      box-shadow: 0 0 0 3px rgba(79,70,229,0.10);
    }
    .tl-item.is-warn .tl-dot { border-color: var(--warn); box-shadow: 0 0 0 3px rgba(180,83,9,0.12); }
    .tl-item.is-err .tl-dot  { border-color: var(--err);  box-shadow: 0 0 0 3px rgba(185,28,28,0.12); }
    .tl-item.is-ok .tl-dot   { border-color: var(--ok);   box-shadow: 0 0 0 3px rgba(22,163,74,0.12); }
    .tl-time { color: var(--soft); font-size: 11.5px; font-weight: 600; }
    .tl-action { color: var(--ink); font-weight: 600; font-size: 14px; margin-top: 2px; }
    .tl-target { color: var(--muted); font-size: 13px; }

    /* ========== Question rows ========== */
    .qrow {
      background: var(--surface-strong);
      border: 1px solid var(--line); border-radius: 12px;
      padding: 12px 16px; margin-bottom: 10px;
      color: var(--ink); font-size: 14px;
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
    }
    .qrow.reviewed { border-left: 3px solid var(--ok); background: linear-gradient(90deg, rgba(22,163,74,0.05), var(--surface-strong) 40%); }

    /* ========== Upload tiles ========== */
    .uphead { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .upic {
      width: 44px; height: 44px; border-radius: 12px;
      display: inline-flex; align-items: center; justify-content: center;
      color: var(--indigo); background: rgba(79,70,229,0.10);
    }
    .upic svg { width: 20px; height: 20px; }
    .upic.teal { color: var(--teal); background: rgba(20,184,166,0.10); }
    .uptitle { font-family: var(--display); font-weight: 700; font-size: 16px; color: var(--ink); }
    .updesc { color: var(--muted); font-size: 12.5px; margin-top: 2px; }

    /* ========== HiL note ========== */
    .hil-line {
      position: relative;
      background: linear-gradient(90deg, rgba(79,70,229,0.07), rgba(20,184,166,0.05) 60%, transparent);
      border: 1px solid var(--line);
      border-left: 3px solid var(--indigo);
      border-radius: 12px;
      padding: 12px 18px;
      color: var(--ink-2);
      font-size: 13.5px; font-weight: 500;
      margin: 16px 0;
      display: flex; align-items: center; gap: 10px;
    }
    .hil-line .ic { color: var(--indigo); }
    .hil-line .ic svg { width: 16px; height: 16px; }

    /* ========== Icon block (16/18/22) ========== */
    .ic { display: inline-flex; align-items: center; justify-content: center; vertical-align: middle; }
    .ic svg { width: 100%; height: 100%; }
    .ic-sm { width: 14px; height: 14px; }
    .ic-md { width: 18px; height: 18px; }
    .ic-lg { width: 22px; height: 22px; }

    /* ========== Section title (legacy hooks) ========== */
    .section-title { font-family: var(--display); font-size: 20px; font-weight: 700; color: var(--ink); margin: 6px 0 4px; }
    .section-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 16px; }
    .kmu-card-title { color: var(--ink); font-family: var(--display); font-weight: 700; font-size: 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
    .kmu-card-title small { color: var(--teal); font-weight: 600; font-size: 12.5px; display: inline-flex; align-items: center; gap: 6px; }

    /* Legacy hooks kept for the existing business code paths */
    .coverage-box { background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(20,184,166,0.07)); border: 1px solid rgba(79,70,229,0.20); border-radius: 14px; padding: 14px 18px; text-align: center; }
    .coverage-value { font-family: var(--display); font-size: 26px; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; }
    .coverage-label { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .kmu-list-avatar { width: 38px; height: 38px; border-radius: 50%; background: var(--grad); color: #fff; display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; flex-shrink: 0; box-shadow: inset 0 1px 0 rgba(255,255,255,0.25); }
    .kmu-logo { font-family: var(--display); font-size: 17px; font-weight: 700; color: var(--ink); padding: 4px 8px 12px 8px; letter-spacing: -0.01em; }

    /* ========== Reveal animations ========== */
    @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .reveal { animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
    .reveal.d1 { animation-delay: .04s; }
    .reveal.d2 { animation-delay: .10s; }
    .reveal.d3 { animation-delay: .16s; }
    .reveal.d4 { animation-delay: .22s; }
    .reveal.d5 { animation-delay: .28s; }
    .reveal.d6 { animation-delay: .34s; }

    /* Modern scroll-linked reveal (Chromium 115+, Firefox 129+, Safari 16+ partial) */
    @supports (animation-timeline: view()) {
      .scroll-reveal {
        animation: fadeUp 1ms linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 22%;
      }
    }

    .kpi, .gcard, .wf-step, .feed-row, .app-row, .tl-item, [data-testid="stVerticalBlockBorderWrapper"] { animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
    .kpi-grid > .kpi:nth-child(1){animation-delay:.04s}
    .kpi-grid > .kpi:nth-child(2){animation-delay:.10s}
    .kpi-grid > .kpi:nth-child(3){animation-delay:.16s}
    .kpi-grid > .kpi:nth-child(4){animation-delay:.22s}
    .wf > .wf-step:nth-child(2){animation-delay:.04s}
    .wf > .wf-step:nth-child(3){animation-delay:.08s}
    .wf > .wf-step:nth-child(4){animation-delay:.12s}
    .wf > .wf-step:nth-child(5){animation-delay:.16s}
    .wf > .wf-step:nth-child(6){animation-delay:.20s}

    /* Hide Streamlit default footer/menu for a cleaner SaaS shell */
    [data-testid="stToolbar"] { right: 14px; }

    @media (max-width: 1080px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .hero { padding: 32px 28px 28px; }
      .hero h1 { font-size: 36px; }
      .ph h1 { font-size: 34px; }
      .cand-row { grid-template-columns: 1fr; }
    }

    /* ====================================================================
       LANDING v2 — Spacing scale, type hierarchy, story rhythm
       ==================================================================== */
    :root{
      --s1:8px; --s2:16px; --s3:24px; --s4:32px; --s5:48px; --s6:64px; --s7:96px; --s8:128px;
      --t-hero: clamp(44px, 6.6vw, 72px);
      --t-h2:   clamp(32px, 4.4vw, 48px);
      --t-sub:  24px;
      --t-body: 18px;
      --t-meta: 14px;
    }

    /* marketing nav (kept) */
    .mkt-nav { display: flex; align-items: center; gap: var(--s4); padding: 8px 8px 8px 10px; }
    .mkt-nav .links { display: flex; align-items: center; gap: 6px; }
    .mkt-nav-link { color: var(--muted); font-size: var(--t-meta); font-weight: 600; padding: 8px 14px; border-radius: 10px; text-decoration: none; transition: background .2s ease, color .2s ease; }
    .mkt-nav-link:hover { background: rgba(15,23,42,0.04); color: var(--ink); }

    /* section rhythm + shared type */
    .lp-sec { margin-top: var(--s7); }
    .lp-sec.tight { margin-top: var(--s5); }
    .lp-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: var(--t-meta); font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: var(--indigo); }
    .lp-eyebrow::before { content:""; width: 22px; height: 1.5px; background: linear-gradient(90deg, var(--indigo), var(--teal)); border-radius: 2px; }
    .lp-eyebrow.center { justify-content: center; }
    .lp-head { max-width: 760px; margin: 0 auto; text-align: center; }
    .lp-h2 { font-family: var(--display); font-weight: 700; font-size: var(--t-h2); line-height: 1.06; letter-spacing: -0.025em; color: var(--ink); margin: var(--s2) 0 var(--s2); }
    .lp-h2 em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .lp-sub { font-size: var(--t-sub); line-height: 1.45; color: var(--muted); }
    .lp-body { font-size: var(--t-body); line-height: 1.65; color: var(--body); }
    .lp-meta { font-size: var(--t-meta); color: var(--soft); }
    .lp-head .lp-sub { margin-top: var(--s2); }

    /* hero — asymmetric, oversized headline */
    .lp-hero { display: grid; grid-template-columns: 1.02fr 0.98fr; gap: var(--s6); align-items: center; margin-top: var(--s4); }
    .lp-hero h1 { font-family: var(--display); font-weight: 700; font-size: var(--t-hero); line-height: 1.0; letter-spacing: -0.035em; color: var(--ink); margin: var(--s3) 0 var(--s3); }
    .lp-hero h1 em { font-style: normal; background: var(--grad-text); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
    .lp-hero .lp-sub { max-width: 520px; }
    .lp-hero-meta { display: flex; gap: var(--s4); flex-wrap: wrap; margin-top: var(--s4); }
    .lp-hero-meta span { display: inline-flex; align-items: center; gap: 8px; color: var(--body); font-size: var(--t-meta); font-weight: 600; }
    .lp-hero-meta .ic { color: var(--teal); }

    /* split (text + visual), alternating sides */
    .lp-split { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s6); align-items: center; }
    .lp-split.reverse .lp-visual { order: -1; }
    .lp-split .lp-sub { margin-top: var(--s2); }
    .lp-split .lp-body { margin-top: var(--s3); }
    .lp-list { list-style: none; padding: 0; margin: var(--s3) 0 0; }
    .lp-list li { position: relative; padding: 12px 0 12px 30px; color: var(--ink-2); font-size: var(--t-body); line-height: 1.5; border-bottom: 1px solid var(--line-2); }
    .lp-list li:last-child { border-bottom: none; }
    .lp-list li::before { content:""; position: absolute; left: 0; top: 17px; width: 14px; height: 14px; border-radius: 5px; background: linear-gradient(135deg, var(--indigo), var(--teal)); }

    /* trust strip — low visual weight */
    .lp-strip { display: flex; flex-wrap: wrap; gap: var(--s5); align-items: center; justify-content: space-between; padding: var(--s3) var(--s4); margin-top: var(--s5); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .lp-strip span { display: inline-flex; align-items: center; gap: 10px; color: var(--muted); font-size: var(--t-meta); font-weight: 600; }
    .lp-strip .ic { color: var(--indigo); }

    /* feature grid — numbered cards */
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--s4); }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--s4); }
    .lp-card { position: relative; background: var(--surface); backdrop-filter: blur(22px); border: 1px solid var(--line); border-radius: var(--radius-lg); padding: var(--s4); box-shadow: var(--shadow-2); transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
    .lp-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-3); border-color: rgba(79,70,229,0.22); }
    .lp-card .num { position: absolute; top: 14px; right: 18px; font-family: var(--display); font-weight: 700; font-size: 34px; color: rgba(79,70,229,0.16); }
    .lp-card .ico { width: 46px; height: 46px; border-radius: 13px; display: inline-flex; align-items: center; justify-content: center; color: var(--indigo); background: rgba(79,70,229,0.10); margin-bottom: var(--s2); }
    .lp-card .ico.teal { color: var(--teal); background: rgba(20,184,166,0.12); }
    .lp-card .ico .ic, .lp-card .ico .ic svg { width: 20px; height: 20px; }
    .lp-card h3 { font-family: var(--display); font-size: 19px; font-weight: 700; color: var(--ink); margin: 0 0 8px; }
    .lp-card p { font-size: 15px; line-height: 1.55; color: var(--muted); margin: 0; }

    /* agent cards — horizontal, distinct from grid */
    .lp-agents { display: flex; flex-direction: column; gap: var(--s2); }
    .lp-agent { display: grid; grid-template-columns: auto 1fr auto; gap: var(--s3); align-items: center; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: var(--s3) var(--s4); box-shadow: var(--shadow-1); transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
    .lp-agent:hover { transform: translateX(4px); box-shadow: var(--shadow-2); border-color: rgba(79,70,229,0.22); }
    .lp-agent .ico { width: 44px; height: 44px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; color: #fff; background: var(--grad); }
    .lp-agent .ico .ic, .lp-agent .ico .ic svg { width: 20px; height: 20px; }
    .lp-agent b { font-family: var(--display); font-size: 17px; color: var(--ink); display: block; }
    .lp-agent small { color: var(--muted); font-size: 14px; }
    .lp-agent .step { font-family: var(--display); font-weight: 700; color: var(--soft); font-size: 13px; letter-spacing: .1em; }

    /* workflow timeline — vertical connected */
    .tl-flow { display: flex; flex-direction: column; }
    .tl-flow .node { position: relative; display: grid; grid-template-columns: auto 1fr; gap: var(--s3); padding-bottom: var(--s4); }
    .tl-flow .node:last-child { padding-bottom: 0; }
    .tl-flow .node::before { content:""; position: absolute; left: 21px; top: 48px; bottom: 0; width: 2px; background: linear-gradient(180deg, rgba(79,70,229,0.40), rgba(20,184,166,0.20)); }
    .tl-flow .node:last-child::before { display: none; }
    .tl-flow .dot { width: 44px; height: 44px; border-radius: 13px; display: inline-flex; align-items: center; justify-content: center; color: #fff; background: var(--grad); box-shadow: 0 8px 18px rgba(79,70,229,0.30); z-index: 1; }
    .tl-flow .dot .ic, .tl-flow .dot .ic svg { width: 20px; height: 20px; }
    .tl-flow .node.final .dot { background: linear-gradient(135deg, var(--teal), #0D9488); box-shadow: 0 8px 18px rgba(20,184,166,0.30); }
    .tl-flow b { font-family: var(--display); font-size: 18px; color: var(--ink); display: block; margin-top: 4px; }
    .tl-flow small { color: var(--muted); font-size: 14px; }

    /* product mockup window */
    .mock { background: var(--surface-strong); border: 1px solid var(--line); border-radius: var(--radius-lg); box-shadow: var(--shadow-3); overflow: hidden; }
    .mock-bar { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,0.6); }
    .mock-dots { display: flex; gap: 6px; }
    .mock-dots i { width: 10px; height: 10px; border-radius: 50%; display: block; }
    .mock-dots i:nth-child(1) { background: #F87171; }
    .mock-dots i:nth-child(2) { background: #FBBF24; }
    .mock-dots i:nth-child(3) { background: #34D399; }
    .mock-url { flex: 1; height: 24px; border-radius: 7px; background: rgba(15,23,42,0.05); display: flex; align-items: center; padding: 0 12px; color: var(--soft); font-size: 12px; }
    .mock-body { padding: var(--s4); }

    /* dashboard mockup */
    .mk-kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--s2); margin-bottom: var(--s3); }
    .mk-kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
    .mk-kpi .l { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .mk-kpi .v { font-family: var(--display); font-weight: 700; font-size: 26px; color: var(--ink); margin-top: 4px; }
    .mk-chart { display: flex; align-items: flex-end; gap: 10px; height: 130px; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
    .mk-col { flex: 1; border-radius: 6px 6px 0 0; background: var(--grad); opacity: .85; }
    .mk-col.c1 { height: 38%; } .mk-col.c2 { height: 62%; } .mk-col.c3 { height: 48%; }
    .mk-col.c4 { height: 82%; } .mk-col.c5 { height: 56%; } .mk-col.c6 { height: 94%; } .mk-col.c7 { height: 70%; }
    .mk-col.alt { background: linear-gradient(180deg, var(--teal), #0D9488); opacity: .8; }

    /* candidate mockup */
    .mk-list { display: flex; flex-direction: column; gap: 10px; }
    .mk-cand { display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
    .mk-cand .nm { font-weight: 600; color: var(--ink); font-size: 14px; }
    .mk-cand .rl { color: var(--muted); font-size: 12px; margin-top: 1px; }
    .mk-cand .chips { display: flex; gap: 5px; margin-top: 6px; flex-wrap: wrap; }
    .mk-match { text-align: right; }
    .mk-match b { font-family: var(--display); font-size: 20px; color: var(--ink); }
    .mk-match small { display: block; color: var(--soft); font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }

    /* feed mockup */
    .mk-feed { display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; padding: 11px 2px; border-bottom: 1px solid var(--line-2); }
    .mk-feed:last-child { border-bottom: none; }
    .mk-feed .fi { width: 32px; height: 32px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; color: var(--indigo); background: rgba(79,70,229,0.08); }
    .mk-feed.ok .fi { color: var(--ok); background: rgba(22,163,74,0.10); }
    .mk-feed.warn .fi { color: var(--warn); background: rgba(180,83,9,0.10); }
    .mk-feed .fb b { color: var(--ink); font-size: 13px; font-weight: 600; display: block; }
    .mk-feed .fb small { color: var(--muted); font-size: 11.5px; }
    .mk-feed .ft { color: var(--soft); font-size: 11px; font-weight: 600; }

    /* human-in-the-loop flow (centered) */
    .lp-hil { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: var(--s3); align-items: stretch; margin-top: var(--s4); }
    .lp-hil .node { text-align: center; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-lg); padding: var(--s5) var(--s3); box-shadow: var(--shadow-2); }
    .lp-hil .node.human { border-color: rgba(20,184,166,0.30); background: linear-gradient(180deg, rgba(20,184,166,0.08), var(--surface) 60%); }
    .lp-hil .ico { width: 50px; height: 50px; border-radius: 15px; display: inline-flex; align-items: center; justify-content: center; color: var(--indigo); background: rgba(79,70,229,0.10); margin-bottom: var(--s2); }
    .lp-hil .node.human .ico { color: var(--teal); background: rgba(20,184,166,0.14); }
    .lp-hil .ico .ic, .lp-hil .ico .ic svg { width: 24px; height: 24px; }
    .lp-hil b { font-family: var(--display); font-size: 18px; color: var(--ink); display: block; margin-bottom: 4px; }
    .lp-hil small { color: var(--muted); font-size: 14px; }
    .lp-hil .arr { display: flex; align-items: center; justify-content: center; color: var(--soft); }
    .lp-hil .arr .ic, .lp-hil .arr .ic svg { width: 24px; height: 24px; }

    /* CTA band (centered, tinted) */
    .lp-cta { position: relative; overflow: hidden; text-align: center; border-radius: var(--radius-xl); padding: var(--s7) var(--s5); box-shadow: var(--shadow-3); border: 1px solid var(--line); background: linear-gradient(135deg, rgba(111,110,255,0.10), rgba(37,99,235,0.07) 50%, rgba(20,184,166,0.10)); }
    .lp-cta::before { content:""; position: absolute; right: -120px; top: -120px; width: 360px; height: 360px; border-radius: 50%; background: radial-gradient(closest-side, rgba(111,110,255,0.22), transparent 70%); pointer-events: none; }
    .lp-cta-in { position: relative; z-index: 1; }
    .lp-cta h2 { font-family: var(--display); font-weight: 700; font-size: var(--t-h2); letter-spacing: -0.025em; color: var(--ink); margin: 0 0 var(--s2); }
    .lp-cta p { font-size: var(--t-sub); color: var(--muted); max-width: 600px; margin: 0 auto; }

    /* footer */
    .lp-foot { margin-top: var(--s7); padding: var(--s4) 4px var(--s2); border-top: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: var(--s3); flex-wrap: wrap; }
    .lp-foot .brand { display: flex; align-items: center; gap: 11px; }
    .lp-foot .brand .m { width: 32px; height: 32px; border-radius: 9px; background: var(--grad); color: #fff; display: inline-flex; align-items: center; justify-content: center; }
    .lp-foot .brand .m .ic, .lp-foot .brand .m .ic svg { width: 17px; height: 17px; }
    .lp-foot .brand b { font-family: var(--display); font-size: 16px; color: var(--ink); }
    .lp-foot .links { display: flex; gap: 8px; flex-wrap: wrap; }
    .lp-foot .links a { color: var(--muted); font-size: 14px; font-weight: 500; padding: 6px 12px; border-radius: 8px; }
    .lp-foot .links a:hover { background: rgba(15,23,42,0.04); color: var(--ink); }
    .lp-foot .copy { color: var(--soft); font-size: 12.5px; width: 100%; padding-top: 14px; }

    /* landing reveal — fade-up on render, scroll-linked where supported */
    .lp-hero, .lp-split, .grid-3, .grid-4, .lp-strip, .tl-flow, .lp-agents, .lp-hil, .lp-cta {
      animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both;
    }
    @supports (animation-timeline: view()) {
      .lp-split, .grid-3, .grid-4, .lp-strip, .tl-flow, .lp-agents, .lp-hil, .lp-cta {
        animation: fadeUp 1ms linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 18%;
      }
    }

    @media (max-width: 1080px) {
      .lp-hero { grid-template-columns: 1fr; gap: var(--s4); }
      .lp-split { grid-template-columns: 1fr; gap: var(--s4); }
      .lp-split.reverse .lp-visual { order: 0; }
      .grid-3, .grid-4 { grid-template-columns: 1fr 1fr; }
      .lp-hil { grid-template-columns: 1fr; }
      .lp-hil .arr { transform: rotate(90deg); }
    }
    @media (max-width: 640px) {
      .grid-3, .grid-4 { grid-template-columns: 1fr; }
      .lp-strip { gap: var(--s3); }
    }
    </style>
    <div class="bg-fx" aria-hidden="true">
      <div class="orb a"></div>
      <div class="orb b"></div>
      <div class="orb c"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SVG-Icon-Set (Lucide-inspired, stroke-only) — alle Emojis durch Icons ersetzt
# ---------------------------------------------------------------------------

ICONS: dict[str, str] = {
    "sparkles":   '<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>',
    "bot":        '<rect x="3" y="11" width="18" height="9" rx="2"/><circle cx="12" cy="6" r="2"/><path d="M12 8v3"/><path d="M8 16h.01"/><path d="M16 16h.01"/>',
    "file":       '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>',
    "check":      '<polyline points="20 6 9 17 4 12"/>',
    "check-circle":'<circle cx="12" cy="12" r="10"/><polyline points="16 10 11 15 8 12"/>',
    "search":     '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.6" y2="16.6"/>',
    "message":    '<path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.8-.9L3 21l1.9-5.7A8.38 8.38 0 0 1 4 11.5 8.5 8.5 0 0 1 12.5 3a8.38 8.38 0 0 1 8.5 8.5z"/>',
    "clipboard":  '<rect x="8" y="2" width="8" height="4" rx="1"/><rect x="4" y="6" width="16" height="16" rx="2"/>',
    "paperclip":  '<path d="M21 12.79V7a4 4 0 0 0-8 0v10a2 2 0 0 0 4 0V8"/>',
    "upload":     '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "settings":   '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "help":       '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "user":       '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users":      '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "puzzle":     '<path d="M19.5 14.5a2.5 2.5 0 1 0 0-5H18V8a2 2 0 0 0-2-2h-1.5a2.5 2.5 0 1 0-5 0H8a2 2 0 0 0-2 2v1.5a2.5 2.5 0 1 0 0 5V16a2 2 0 0 0 2 2h1.5a2.5 2.5 0 1 0 5 0H16a2 2 0 0 0 2-2v-1.5z"/>',
    "scale":      '<path d="M12 3v18"/><path d="M5 9l-3 7a4 4 0 0 0 6 0L5 9z"/><path d="M19 9l-3 7a4 4 0 0 0 6 0L19 9z"/><path d="M3 5h18"/>',
    "arrow-right":'<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "chevron-down":'<polyline points="6 9 12 15 18 9"/>',
    "chevron-right":'<polyline points="9 6 15 12 9 18"/>',
    "log":        '<path d="M21 12V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7"/><path d="M9 7h6"/><path d="M9 11h6"/><path d="M9 15h3"/><circle cx="17" cy="17" r="3"/><path d="M17 14v3l2 1"/>',
    "lightning":  '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "shield":     '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "trend-up":   '<polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/>',
    "edit":       '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
    "mail":       '<rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="2 6 12 13 22 6"/>',
    "filter":     '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "clock":      '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "x":          '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "play":       '<polygon points="5 3 19 12 5 21 5 3"/>',
    "more":       '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
    "logout":     '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "globe":      '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
}


def ic(name: str, size: str = "md", cls: str = "") -> str:
    body = ICONS.get(name, "")
    klass = f"ic ic-{size} " + (cls or "")
    return (
        f'<span class="{klass.strip()}">'
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg></span>"
    )


# ---------------------------------------------------------------------------
# Session-State + Router
# ---------------------------------------------------------------------------

if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "job_profile" not in st.session_state:
    st.session_state.job_profile = None
if "job_profile_source" not in st.session_state:
    st.session_state.job_profile_source = ""
if "reviewed_questions" not in st.session_state:
    st.session_state.reviewed_questions = {}
if "_flash" not in st.session_state:
    st.session_state._flash = []
if "_coverage_logged" not in st.session_state:
    st.session_state._coverage_logged = set()
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "home"

NAV_ITEMS = [
    ("dashboard",  "Dashboard",  "lightning"),
    ("recruiting", "Recruiting", "clipboard"),
    ("kandidaten", "Kandidaten", "users"),
    ("audit",      "Audit Log",  "log"),
]
_VALID_PAGES = {pid for pid, _, _ in NAV_ITEMS} | {"home"}


def goto(page: str) -> None:
    if page in _VALID_PAGES:
        st.session_state.nav_page = page
        st.rerun()


def quality_bucket(quality: dict | None) -> str:
    if not quality or quality.get("_error"):
        return "unvollständig"
    missing_n = len(quality.get("missing_information", []))
    unclear_n = len(quality.get("unclear_information", []))
    if missing_n == 0 and unclear_n == 0:
        return "vollständig"
    if missing_n >= 3:
        return "unvollständig"
    return "informationslücken"


def initials_for(name: str) -> str:
    parts = [p for p in (name or "").strip().split() if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _activity_kind(action: str) -> tuple[str, str]:
    """Returns (css_class, icon_name) for a given audit action string."""
    a = (action or "").lower()
    if "fehler" in a:
        return "is-err", "x"
    if "rückfrage" in a or "rueckfrage" in a:
        return "", "message"
    if "stellenprofil" in a:
        return "is-teal", "clipboard"
    if "cv analysiert" in a or "extrahiert" in a or "text extrahiert" in a:
        return "is-success", "file"
    if "informationslücken" in a or "klärungspunkte" in a:
        return "", "puzzle"
    if "anforderungen" in a:
        return "is-teal", "scale"
    if "feedback" in a:
        return "", "edit"
    if "hochgeladen" in a:
        return "", "upload"
    return "", "sparkles"


# ---------------------------------------------------------------------------
# Sidebar — minimal (Nav lebt in der Top-Bar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="kmu-logo">Recruiting&nbsp;AI</div>',
        unsafe_allow_html=True,
    )
    st.caption("AI Workspace für Geschäftsführer ohne eigenes HR-Team.")
    st.divider()
    if st.button(
        "←  Zur Website",
        key="sb_home",
        use_container_width=True,
        type="primary" if st.session_state.nav_page == "home" else "secondary",
    ):
        goto("home")
    st.markdown("**Bereich**")
    for _pid, _lbl, _ico in NAV_ITEMS:
        if st.button(
            ("●  " if st.session_state.nav_page == _pid else "    ") + _lbl,
            key=f"sb_{_pid}",
            use_container_width=True,
            type="primary" if st.session_state.nav_page == _pid else "secondary",
        ):
            goto(_pid)
    st.divider()
    st.button("Einstellungen", key="sb_settings", use_container_width=True)
    st.button("Hilfe & Support", key="sb_help", use_container_width=True)


# ---------------------------------------------------------------------------
# Top-Nav (Brand · Pages · User-Dropdown)
# ---------------------------------------------------------------------------


def render_top_nav() -> None:
    active = st.session_state.nav_page
    cols = st.columns([3.0, 1.05, 1.15, 1.15, 1.05, 1.05, 1.6])
    with cols[0]:
        st.markdown(
            f"""
            <div class="shell-brand">
              <div class="shell-mark">{ic("sparkles", "md")}</div>
              <div class="shell-name">Recruiting&nbsp;AI<small>AI Workspace</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for idx, (pid, lbl, _icn) in enumerate(NAV_ITEMS):
        with cols[1 + idx]:
            wrap_cls = "nav-btn-wrap active" if active == pid else "nav-btn-wrap"
            st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
            if st.button(lbl, key=f"nav_{pid}", use_container_width=True):
                goto(pid)
            st.markdown("</div>", unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<div class="nav-btn-wrap">', unsafe_allow_html=True)
        _has_popover = hasattr(st, "popover")
        _menu_ctx = st.popover("Mehr") if _has_popover else st.expander("Mehr")
        with _menu_ctx:
            st.caption("Schnellzugriff")
            if st.button("Zur Website", key="mn_home", use_container_width=True):
                goto("home")
            if st.button("Einstellungen", key="mn_settings", use_container_width=True):
                pass
            if st.button("Hilfe & Support", key="mn_help", use_container_width=True):
                pass
            st.divider()
            if st.button("Sign out", key="mn_signout", use_container_width=True):
                pass
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[6]:
        st.markdown(
            """
            <div class="user-chip">
              <span class="av">GF</span>
              <span><b>Geschäftsführer</b><small>Pro Plan</small></span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page-Building-Blocks
# ---------------------------------------------------------------------------


def page_header(eyebrow: str, title_html: str, lead: str) -> None:
    st.markdown(
        f'<div class="ph"><span class="eyebrow">{eyebrow}</span>'
        f"<h1>{title_html}</h1><div class=\"lead\">{lead}</div></div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, sub: str = "", right: str = "") -> None:
    right_html = f'<div class="right">{right}</div>' if right else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="sec"><div class="sec-head"><div><h2>{title}</h2>'
        f"{sub_html}</div>{right_html}</div></div>",
        unsafe_allow_html=True,
    )


def render_workflow() -> None:
    steps = [
        ("upload", "Stelle hochladen", "Stellenanzeige einfügen."),
        ("clipboard", "Stellenprofil-Agent", "Prüfbare Anforderungen."),
        ("file", "CV-Agent", "Lebensläufe strukturieren."),
        ("scale", "Anforderungsabgleich", "Gefunden / teilweise / nicht."),
        ("puzzle", "Informationslücken", "Fehlende Angaben."),
        ("message", "Rückfragen", "Höflich vorbereitet."),
    ]
    cells = ""
    for i, (icn, name, desc) in enumerate(steps):
        is_final = "is-final" if i == len(steps) - 1 else ""
        cells += (
            f'<div class="wf-step {is_final}">'
            f'<div class="wf-ic">{ic(icn, "md")}</div>'
            f"<b>{name}</b><small>{desc}</small></div>"
        )
    st.markdown(f'<div class="wf">{cells}</div>', unsafe_allow_html=True)


def hil_note(text: str) -> None:
    st.markdown(
        f'<div class="hil-line"><span class="ic">{ic("shield", "md")}</span>'
        f"<span>{text}</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Render: Dashboard
# ---------------------------------------------------------------------------


def render_dashboard() -> None:
    total_candidates = len(st.session_state.candidates)
    total_gaps = sum(
        len((c.get("quality") or {}).get("missing_information") or [])
        + len((c.get("quality") or {}).get("unclear_information") or [])
        for c in st.session_state.candidates
    )
    total_questions = sum(
        len((c.get("followups") or {}).get("questions") or [])
        for c in st.session_state.candidates
    )
    total_found = 0
    if st.session_state.job_profile:
        for c in st.session_state.candidates:
            total_found += status_counts(
                evaluate_candidate_requirements(
                    st.session_state.job_profile, c["data"]
                )
            )["Gefunden"]

    # Hero
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-in">
            <span class="hero-eye"><span class="sp"></span>AI Workspace · live</span>
            <h1>Bewerbungen verstehen.<br/><em>Menschen</em> entscheiden.</h1>
            <div class="sub">Vier spezialisierte Agenten strukturieren Lebensläufe,
            prüfen fachliche Anforderungen und bereiten Rückfragen vor — ohne
            automatische Personalentscheidung.</div>
            <div class="hero-trust">
              <span><span class="ic">{ic("check-circle", "md")}</span> Multi-Agent-Analyse</span>
              <span><span class="ic">{ic("shield", "md")}</span> Human-in-the-Loop</span>
              <span><span class="ic">{ic("log", "md")}</span> Vollständig auditierbar</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Hero CTAs (real buttons)
    ca, cb, _spacer = st.columns([1.4, 1.4, 4])
    with ca:
        if st.button("Recruiting starten", key="hero_recruiting", type="primary", use_container_width=True):
            goto("recruiting")
    with cb:
        if st.button("Kandidaten ansehen", key="hero_kandidaten", use_container_width=True):
            goto("kandidaten")

    # KPI Strip
    st.markdown('<div class="sec">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-head"><div class="kpi-ic purple">{ic("file", "md")}</div>
              <div class="kpi-label">Bewerbungen</div></div>
            <div class="kpi-value">{total_candidates}</div>
            <div class="kpi-delta">{ic("trend-up","sm")}&nbsp;analysiert</div>
          </div>
          <div class="kpi">
            <div class="kpi-head"><div class="kpi-ic indigo">{ic("check-circle", "md")}</div>
              <div class="kpi-label">Anforderungen gefunden</div></div>
            <div class="kpi-value">{total_found}</div>
            <div class="kpi-delta">über alle Kandidaten</div>
          </div>
          <div class="kpi">
            <div class="kpi-head"><div class="kpi-ic blue">{ic("puzzle", "md")}</div>
              <div class="kpi-label">Klärungsbedarf</div></div>
            <div class="kpi-value">{total_gaps}</div>
            <div class="kpi-delta">offene Informationslücken</div>
          </div>
          <div class="kpi">
            <div class="kpi-head"><div class="kpi-ic teal">{ic("message", "md")}</div>
              <div class="kpi-label">Rückfragen vorbereitet</div></div>
            <div class="kpi-value">{total_questions}</div>
            <div class="kpi-delta">warten auf Freigabe</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Workflow status
    section_header(
        "Workflow",
        "Sechs Schritte. Der letzte gehört dem Menschen.",
        "Multi-Agent",
    )
    render_workflow()

    # Two-column: last applicants | activity feed
    st.markdown('<div class="sec"></div>', unsafe_allow_html=True)
    dash_left, dash_right = st.columns([1.25, 1], gap="large")

    with dash_left:
        with st.container(border=True):
            st.markdown(
                f'<div class="kmu-card-title">Letzte Bewerbungen'
                f'<small><span class="ic">{ic("clock","sm")}</span> Upload-Reihenfolge</small></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.candidates:
                rows = ""
                for c in st.session_state.candidates[-5:][::-1]:
                    cv = c["data"]
                    q = c.get("quality") or {}
                    has_gaps = bool(
                        q.get("missing_information") or q.get("unclear_information")
                    )
                    badge_cls = "badge-warn" if has_gaps else "badge-ok"
                    badge_lbl = "Klärungsbedarf" if has_gaps else "Analyse abgeschlossen"
                    name = cv.get("name") or "(ohne Name)"
                    first_role = ""
                    if cv.get("experience"):
                        first_role = cv["experience"][0].get("role", "")
                    rows += (
                        '<div class="app-row">'
                        f'<div class="app-av">{initials_for(name)}</div>'
                        '<div>'
                        f'<div class="app-name">{name}</div>'
                        f'<div class="app-role">{first_role or c["filename"]}</div>'
                        '</div>'
                        f'<span class="badge {badge_cls}">{badge_lbl}</span>'
                        '</div>'
                    )
                st.markdown(f'<div class="applist">{rows}</div>', unsafe_allow_html=True)
            else:
                st.caption(
                    "Noch keine Bewerbungen analysiert. Wechseln Sie zu "
                    "Recruiting, um Lebensläufe hochzuladen."
                )

    with dash_right:
        with st.container(border=True):
            st.markdown(
                f'<div class="kmu-card-title">AI Activity'
                f'<small><span class="gcard-meta"><span class="live"></span> live</span></small></div>',
                unsafe_allow_html=True,
            )
            _recent_log = load_audit_log()[-8:][::-1]
            if _recent_log:
                rows = ""
                for entry in _recent_log:
                    ts = entry.get("timestamp", "")
                    t_disp = ts.split("T", 1)[1][:5] if "T" in ts else ts
                    action = entry.get("action", "")
                    target = entry.get("target", "")
                    kind, icn = _activity_kind(action)
                    rows += (
                        f'<div class="feed-row {kind}">'
                        f'<div class="feed-ic">{ic(icn, "md")}</div>'
                        '<div class="feed-body">'
                        f'<b>{action}</b><small>{target}</small></div>'
                        f'<span class="feed-time">{t_disp}</span></div>'
                    )
                st.markdown(f'<div class="feed">{rows}</div>', unsafe_allow_html=True)
            else:
                st.caption("Noch keine Agent-Aktivität.")

    hil_note("Der Agent strukturiert Informationen. Die Entscheidung trifft der Geschäftsführer.")


# ---------------------------------------------------------------------------
# Render: Recruiting (Upload + Stellenprofil)
# ---------------------------------------------------------------------------


def render_recruiting() -> None:
    page_header(
        "Recruiting",
        "Stellenprofil <em>analysieren</em>. Bewerbungen aufnehmen.",
        "Geben Sie die Stellenanzeige ein und laden Sie Lebensläufe als PDF hoch. "
        "Der Workspace strukturiert beides — Sie behalten die Kontrolle.",
    )

    # Flash messages (preserved across reruns)
    for _level, _msg in st.session_state._flash:
        getattr(st, _level, st.info)(_msg)
    st.session_state._flash = []

    up_left, up_right = st.columns(2, gap="large")

    with up_left:
        with st.container(border=True):
            st.markdown(
                f'<div class="uphead"><div class="upic">{ic("clipboard", "md")}</div>'
                f'<div><div class="uptitle">Stellenprofil</div>'
                f'<div class="updesc">Stellenanzeige als Text einfügen.</div></div></div>',
                unsafe_allow_html=True,
            )
            job_url = st.text_input(
                "Stellen-URL (optional, derzeit deaktiviert)",
                placeholder="https://… (in dieser Demo nicht aktiv)",
                label_visibility="collapsed",
                key="rec_job_url",
            )
            if job_url.strip():
                st.caption(
                    "Hinweis: URL-Fetch ist in dieser Demo nicht aktiviert. "
                    "Bitte den Stellentext direkt unten einfügen."
                )
            job_text = st.text_area(
                "Stellenprofil",
                height=180,
                placeholder=(
                    "z. B. Wir suchen einen Python-Entwickler mit SQL und "
                    "Deutsch C1 …"
                ),
                label_visibility="collapsed",
                key="rec_job_text",
            )
            if st.button(
                "Stelle analysieren",
                key="rec_analyze_job",
                type="primary",
                disabled=not job_text.strip(),
                use_container_width=True,
            ):
                with st.spinner("Strukturiere Stellenprofil …"):
                    try:
                        profile = analyze_job_profile(job_text)
                        st.session_state.job_profile = profile
                        st.session_state.job_profile_source = job_text.strip()
                        st.session_state._coverage_logged = set()
                        log_audit(
                            action="Stellenprofil analysiert",
                            target=profile.get("role", "(ohne Titel)"),
                            result_type="strukturiertes Stellenprofil",
                        )
                        _all_req = _collect_all_requirements(profile)
                        _check, _soft = filter_objectively_checkable_requirements(_all_req)
                        log_audit(
                            action="fachlich prüfbare Anforderungen extrahiert",
                            target=profile.get("role", "(ohne Titel)"),
                            result_type=f"{len(_check)} prüfbar",
                        )
                        log_audit(
                            action="nicht automatisch prüfbare Anforderungen erkannt",
                            target=profile.get("role", "(ohne Titel)"),
                            result_type=f"{len(_soft)} weich/soft",
                        )
                        st.session_state._flash.append(
                            ("success", "Stellenprofil strukturiert.")
                        )
                    except Exception as e:  # noqa: BLE001
                        log_audit(
                            action="Stellenprofil analysiert",
                            target="(Haupteingabe)",
                            result_type=f"Fehler: {e}",
                        )
                        st.session_state._flash.append(
                            ("error", f"Fehler bei der Analyse des Stellenprofils: {e}")
                        )
                st.rerun()
            if st.session_state.job_profile and st.button(
                "Stellenprofil löschen",
                key="rec_clear_job",
                use_container_width=True,
            ):
                st.session_state.job_profile = None
                st.session_state.job_profile_source = ""
                st.session_state._coverage_logged = set()
                st.rerun()

    with up_right:
        with st.container(border=True):
            st.markdown(
                f'<div class="uphead"><div class="upic teal">{ic("upload", "md")}</div>'
                f'<div><div class="uptitle">Bewerbungen</div>'
                f'<div class="updesc">Mehrere Lebensläufe als PDF.</div></div></div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "PDF-Dateien",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="rec_uploader",
            )
            if st.button(
                "Bewerbungen analysieren",
                key="rec_analyze_cvs",
                type="primary",
                disabled=not uploaded,
                use_container_width=True,
            ):
                prior_feedback = feedback_block(load_feedback())
                new_files = [
                    f for f in uploaded
                    if f.name not in st.session_state.processed_files
                ]
                if not new_files:
                    st.session_state._flash.append(
                        ("warning", "Alle ausgewählten Dateien wurden bereits verarbeitet.")
                    )
                    st.rerun()
                else:
                    progress = st.progress(0.0)
                    ok = 0
                    errors: list[str] = []
                    for i, f in enumerate(new_files, start=1):
                        with st.spinner(f"Verarbeite {f.name} …"):
                            try:
                                log_audit(
                                    action="PDF hochgeladen",
                                    target=f.name,
                                    result_type="Datei akzeptiert",
                                )
                                text = extract_text_from_pdf(f.read())
                                if not text:
                                    errors.append(
                                        f"{f.name}: kein Text extrahierbar "
                                        "(evtl. gescanntes PDF ohne OCR)."
                                    )
                                    log_audit(
                                        action="Text extrahiert",
                                        target=f.name,
                                        result_type="leer / nicht maschinenlesbar",
                                    )
                                    progress.progress(i / len(new_files))
                                    continue
                                log_audit(
                                    action="Text extrahiert",
                                    target=f.name,
                                    result_type=f"{len(text)} Zeichen",
                                )
                                data = extract_cv(text, prior_feedback)
                                log_audit(
                                    action="CV analysiert",
                                    target=data.get("name", "") or f.name,
                                    result_type="strukturierte Felder + Belege",
                                )
                                try:
                                    gaps = analyze_information_gaps(text, data)
                                    log_audit(
                                        action="Informationslücken erkannt",
                                        target=data.get("name", "") or f.name,
                                        result_type=(
                                            f"{len(gaps['missing_information'])} fehlend, "
                                            f"{len(gaps['unclear_information'])} unklar"
                                        ),
                                    )
                                except Exception as qe:  # noqa: BLE001
                                    gaps = {
                                        "missing_information": [],
                                        "unclear_information": [],
                                        "suggested_questions": [],
                                        "_error": str(qe),
                                    }
                                    log_audit(
                                        action="Informationslücken erkannt",
                                        target=f.name,
                                        result_type=f"Fehler: {qe}",
                                    )
                                try:
                                    followups = generate_follow_up_questions(
                                        gaps, st.session_state.job_profile
                                    )
                                    log_audit(
                                        action="Rückfragen erzeugt",
                                        target=data.get("name", "") or f.name,
                                        result_type=(
                                            f"{len(followups['questions'])} Rückfragen "
                                            "(warten auf Prüfung & Freigabe)"
                                        ),
                                    )
                                except Exception as fe:  # noqa: BLE001
                                    followups = {
                                        "questions": gaps.get("suggested_questions", [])
                                    }
                                    log_audit(
                                        action="Rückfragen erzeugt",
                                        target=f.name,
                                        result_type=f"Fehler: {fe}",
                                    )
                                st.session_state.candidates.append(
                                    {
                                        "filename": f.name,
                                        "data": data,
                                        "quality": gaps,
                                        "followups": followups,
                                    }
                                )
                                st.session_state.processed_files.add(f.name)
                                ok += 1
                                if st.session_state.job_profile:
                                    _req = evaluate_candidate_requirements(
                                        st.session_state.job_profile, data
                                    )
                                    _cnt = status_counts(_req)
                                    log_audit(
                                        action="Anforderungen abgeglichen",
                                        target=data.get("name", "") or f.name,
                                        result_type=(
                                            f"{_cnt['Gefunden']} gefunden, "
                                            f"{_cnt['Teilweise gefunden']} teilweise, "
                                            f"{_cnt['Nicht gefunden']} nicht gefunden"
                                        ),
                                    )
                                    _klaer = _cnt["Teilweise gefunden"]
                                    if _klaer:
                                        log_audit(
                                            action="Klärungspunkte erkannt",
                                            target=data.get("name", "") or f.name,
                                            result_type=f"{_klaer} Klärungspunkte",
                                        )
                            except Exception as e:  # noqa: BLE001
                                errors.append(f"Fehler bei {f.name}: {e}")
                                log_audit(
                                    action="CV analysiert",
                                    target=f.name,
                                    result_type=f"Fehler: {e}",
                                )
                        progress.progress(i / len(new_files))
                    if ok:
                        st.session_state._flash.append(
                            ("success", f"{ok} Lebensläufe analysiert.")
                        )
                    for err in errors:
                        st.session_state._flash.append(("error", err))
                    st.rerun()
            if st.session_state.candidates and st.button(
                "Alle Kandidaten löschen",
                key="rec_clear_cands",
                use_container_width=True,
            ):
                st.session_state.candidates = []
                st.session_state.processed_files = set()
                st.session_state.reviewed_questions = {}
                st.session_state._coverage_logged = set()
                st.rerun()

    # Structured job profile (read-only)
    section_header(
        "Strukturiertes Stellenprofil",
        "Strukturierte Anforderungen — keine Bewerber-Bewertung.",
    )
    if st.session_state.job_profile:
        jp = st.session_state.job_profile
        with st.container(border=True):
            st.markdown(f"### {jp.get('role') or NICHT_GEFUNDEN}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Muss-Kriterien**")
                if jp.get("must_criteria"):
                    for x in jp["must_criteria"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
                st.markdown("**Gewünschte Skills**")
                if jp.get("desired_skills"):
                    for x in jp["desired_skills"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
                st.markdown("**Gewünschte Zertifikate**")
                if jp.get("desired_certificates"):
                    for x in jp["desired_certificates"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
            with c2:
                st.markdown("**Kann-Kriterien**")
                if jp.get("nice_criteria"):
                    for x in jp["nice_criteria"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
                st.markdown("**Gewünschte Sprachen**")
                if jp.get("desired_languages"):
                    for x in jp["desired_languages"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
                st.markdown("**Gewünschte Berufserfahrung**")
                if jp.get("desired_experience"):
                    for x in jp["desired_experience"]:
                        st.markdown(f"- {x}")
                else:
                    st.caption(NICHT_GEFUNDEN)
            with st.expander("Rohdaten (JSON)"):
                st.json(jp)
    else:
        st.info("Noch kein Stellenprofil hinterlegt — fügen Sie oben einen Stellentext ein.")

    hil_note("Recruiting AI bewertet niemanden. Es strukturiert — Sie entscheiden.")


# ---------------------------------------------------------------------------
# Render: Kandidaten (Filter, Karten, Detail mit Rückfragen + Feedback)
# ---------------------------------------------------------------------------


def render_kandidaten() -> None:
    page_header(
        "Kandidaten",
        "Strukturierte <em>Profile</em>. Belegte Anforderungen.",
        "Alle eingelesenen Bewerbungen in Upload-Reihenfolge. Kein Ranking, "
        "kein Score, keine Sortierung nach Eignung.",
    )

    if not st.session_state.candidates:
        with st.container(border=True):
            st.info(
                "Noch keine Kandidaten — wechseln Sie zu Recruiting, um Bewerbungen hochzuladen."
            )
            if st.button("Zu Recruiting wechseln", key="kand_empty_goto", type="primary"):
                goto("recruiting")
        return

    # Filter
    with st.container(border=True):
        st.markdown(
            f'<div class="kmu-card-title">Filter'
            f'<small><span class="ic">{ic("filter","sm")}</span> objektive Felder</small></div>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            skills_q = st.text_input("Skills (Komma-getrennt)", "", key="kand_f_skills")
        with col2:
            languages_q = st.text_input("Sprachen", "", key="kand_f_langs")
        with col3:
            certs_q = st.text_input("Zertifikate", "", key="kand_f_certs")

    filtered = filter_candidates(
        st.session_state.candidates, skills_q, languages_q, certs_q
    )
    section_header(
        "Übersicht",
        f"{len(filtered)} von {len(st.session_state.candidates)} Kandidaten",
        "Upload-Reihenfolge",
    )
    job_profile_state = st.session_state.job_profile
    if job_profile_state:
        for _c in filtered:
            _cov = calculate_requirement_coverage(job_profile_state, _c["data"])
            log_coverage_once(_c, _cov, job_profile_state)

    if not filtered:
        st.write("Keine Bewerber entsprechen den Filtern.")
        return

    fnames = [c["filename"] for c in filtered]
    if st.session_state.get("_open_cand") not in fnames:
        st.session_state._open_cand = fnames[0]

    for _c in filtered:
        _d = _c["data"]
        _name = _d.get("name") or "(ohne Name)"
        _role = ""
        if _d.get("experience"):
            _role = _d["experience"][0].get("role", "")
        _q = _c.get("quality") or {}
        _has_gaps = bool(
            _q.get("missing_information") or _q.get("unclear_information")
        )
        if job_profile_state:
            _rc = status_counts(
                evaluate_candidate_requirements(job_profile_state, _d)
            )
            _tot = (
                _rc["Gefunden"]
                + _rc["Teilweise gefunden"]
                + _rc["Nicht gefunden"]
            )
            _metric_html = (
                f'<div class="cand-metric"><em>{_rc["Gefunden"]}</em>'
                f"<span> / {_tot}</span></div>"
                f'<div class="cand-metric-label">Anforderungen gefunden</div>'
            ) if _tot else (
                '<div class="cand-metric">—</div>'
                '<div class="cand-metric-label">Kein Profil hinterlegt</div>'
            )
        else:
            _metric_html = (
                '<div class="cand-metric">—</div>'
                '<div class="cand-metric-label">Kein Profil hinterlegt</div>'
            )
        _skills = _d.get("skills") or []
        _skill_badges = "".join(
            f'<span class="badge badge-info">{s}</span>'
            for s in _skills[:4]
        ) + (
            f'<span class="badge badge-muted">+{len(_skills) - 4}</span>'
            if len(_skills) > 4 else ""
        )
        _klaer_n = len(
            _q.get("missing_information") or []
        ) + len(_q.get("unclear_information") or [])
        _status_badge = (
            f'<span class="badge badge-warn">Klärungsbedarf · {_klaer_n}</span>'
            if _klaer_n
            else '<span class="badge badge-ok">Analyse abgeschlossen</span>'
        )
        _is_open = _c["filename"] == st.session_state._open_cand
        with st.container(border=True):
            cc = st.columns([2.6, 1.1, 2.6, 1.3])
            with cc[0]:
                st.markdown(
                    f'<div class="cand-id">'
                    f'<div class="app-av">{initials_for(_name)}</div>'
                    "<div>"
                    f'<div class="cand-name">{_name}</div>'
                    f'<div class="cand-role">{_role or _c["filename"]}</div>'
                    "</div></div>",
                    unsafe_allow_html=True,
                )
            with cc[1]:
                st.markdown(_metric_html, unsafe_allow_html=True)
            with cc[2]:
                _skills_html = _skill_badges or (
                    '<span class="badge badge-muted">keine Skills erfasst</span>'
                )
                st.markdown(
                    f'<div class="cand-skills">{_skills_html}</div>'
                    f'<div class="mt-2">{_status_badge}</div>',
                    unsafe_allow_html=True,
                )
            with cc[3]:
                if st.button(
                    "Profil geöffnet" if _is_open else "Profil öffnen",
                    key=f"open_{_c['filename']}",
                    disabled=_is_open,
                    use_container_width=True,
                    type="primary" if not _is_open else "secondary",
                ):
                    st.session_state._open_cand = _c["filename"]
                    st.rerun()

    # ---- Detail ----
    selected = next(
        c for c in filtered if c["filename"] == st.session_state._open_cand
    )
    cv_data = selected["data"]
    followups_q = (selected.get("followups") or {}).get("questions") or []
    req_rows = evaluate_candidate_requirements(
        st.session_state.job_profile, cv_data
    )
    req_counts = status_counts(req_rows)
    req_total = len(req_rows)
    klaerung_items = [r for r in req_rows if r["status"] == "Teilweise gefunden"]

    section_header(
        "Kandidatenprofil",
        cv_data.get("name") or "(ohne Name)",
        "Belege im Lebenslauf",
    )
    with st.container(border=True):
        head_l, head_r = st.columns([3, 2])
        with head_l:
            st.markdown(f"### {cv_data.get('name') or NICHT_GEFUNDEN}")
            if cv_data.get("experience"):
                first = cv_data["experience"][0]
                sub = (
                    f"{first.get('role', '')} · {first.get('company', '')}"
                ).strip(" ·")
                if sub:
                    st.caption(sub)
        with head_r:
            if req_total:
                pct = int(round((req_counts["Gefunden"] / req_total) * 100))
                st.markdown(
                    f"""
                    <div class="cov">
                      <div>
                        <div class="cov-num"><em>{req_counts['Gefunden']}</em> / {req_total}</div>
                        <div class="cov-label">Anforderungen gefunden</div>
                      </div>
                      <div class="cov-bar"><i style="width:{pct}%"></i></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Kein Stellenprofil hinterlegt.")

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Berufserfahrung**")
            exp = cv_data.get("experience") or []
            if exp:
                for e in exp[:5]:
                    line = f"- {e.get('role') or '?'} @ {e.get('company') or '?'}"
                    if e.get("period"):
                        line += f" · {e['period']}"
                    st.markdown(line)
            else:
                st.caption(NICHT_GEFUNDEN)
            st.markdown("**Ausbildung**")
            edu = cv_data.get("education") or []
            if edu:
                for e in edu[:3]:
                    st.markdown(
                        f"- {e.get('degree') or '?'}, "
                        f"{e.get('institution') or '?'}"
                        + (f" · {e['period']}" if e.get('period') else "")
                    )
            else:
                st.caption(NICHT_GEFUNDEN)
        with col_right:
            skills = cv_data.get("skills") or []
            if skills:
                badges = "".join(
                    f'<span class="badge badge-info">{s}</span>'
                    for s in skills[:14]
                )
                more = (
                    f'<span class="badge badge-info">+{len(skills) - 14}</span>'
                    if len(skills) > 14 else ""
                )
                st.markdown(
                    f"**Skills**<br/><div class=\"cand-skills mt-2\">{badges}{more}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**Skills:** {NICHT_GEFUNDEN}")
            st.markdown(f"**Sprachen:** {_short_languages(cv_data)}")
            certs = cv_data.get("certificates") or []
            st.markdown(
                f"**Zertifikate:** "
                f"{', '.join(certs) if certs else NICHT_GEFUNDEN}"
            )

        STATUS_BADGE_MAP = {
            "Gefunden": "badge-ok",
            "Teilweise gefunden": "badge-warn",
            "Nicht gefunden": "badge-err",
        }
        if req_rows:
            st.markdown("### Gefundene Anforderungen")
            st.caption(
                f"Gefunden: **{req_counts['Gefunden']} von {req_total}** · "
                f"teilweise {req_counts['Teilweise gefunden']} · "
                f"nicht gefunden {req_counts['Nicht gefunden']}"
            )
            for r in req_rows:
                cls = STATUS_BADGE_MAP.get(r["status"], "badge-muted")
                hinweis = r.get("reason") or ""
                with st.expander(f"{r['requirement']} — {r['status']} · {hinweis}"):
                    st.markdown(
                        f'<span class="badge {cls}">{r["status"]}</span>',
                        unsafe_allow_html=True,
                    )
                    src = r.get("evidence") or find_source_excerpt(
                        r["requirement"], cv_data
                    )
                    if src:
                        st.markdown("**Fundstelle im Lebenslauf:**")
                        st.markdown(f"> {src}")
                    else:
                        st.caption("Keine konkrete Fundstelle erfasst.")

    # ---- Rückfragen (Human-in-the-Loop) ----
    section_header(
        "Rückfragen",
        "Vom Rückfragen-Agenten vorbereitet — niemals automatisch versendet.",
        "Human-in-the-Loop",
    )
    if not followups_q:
        with st.container(border=True):
            st.info("Keine offenen Rückfragevorschläge für diesen Kandidaten.")
    else:
        with st.container(border=True):
            fname = selected["filename"]
            reviewed_set = st.session_state.reviewed_questions.setdefault(fname, set())
            for i, q in enumerate(followups_q):
                is_reviewed = i in reviewed_set
                tag = (
                    '<span class="badge badge-ok">geprüft</span>'
                    if is_reviewed
                    else '<span class="badge badge-info">offen</span>'
                )
                cls = "qrow reviewed" if is_reviewed else "qrow"
                st.markdown(
                    f'<div class="{cls}"><span>{q}</span>{tag}</div>',
                    unsafe_allow_html=True,
                )
                bc1, bc2, _bc3 = st.columns([1.3, 1.5, 3])
                with bc1:
                    if not is_reviewed and st.button(
                        "Rückfrage prüfen",
                        key=f"check_{fname}_{i}",
                        use_container_width=True,
                    ):
                        log_audit(
                            action="Rückfrage geprüft",
                            target=cv_data.get("name", "") or fname,
                            result_type=q[:80],
                        )
                        st.toast("Rückfrage markiert als geprüft.", icon="✅")
                with bc2:
                    if not is_reviewed and st.button(
                        "Als geprüft markieren",
                        key=f"mark_{fname}_{i}",
                        type="primary",
                        use_container_width=True,
                    ):
                        reviewed_set.add(i)
                        log_audit(
                            action="Rückfrage als geprüft markiert",
                            target=cv_data.get("name", "") or fname,
                            result_type=q[:80],
                        )
                        st.rerun()

    # ---- Inline Feedback + JSON ----
    with st.expander("Feedback zu diesem Kandidaten geben"):
        st.caption("Feedback verbessert nur die Extraktion, nicht die Auswahl.")
        with st.form("feedback_form", clear_on_submit=True):
            category = st.selectbox("Art der Korrektur", FEEDBACK_CATEGORIES)
            note = st.text_area(
                "Anmerkung",
                placeholder=(
                    "z. B. „SQL wurde übersehen“ oder "
                    "„Sprache Französisch falsch erkannt“"
                ),
            )
            submitted = st.form_submit_button("Feedback speichern", type="primary")
            if submitted:
                if not note.strip():
                    st.warning("Bitte eine Anmerkung eingeben.")
                else:
                    add_feedback(cv_data.get("name", ""), category, note)
                    log_audit(
                        action="Feedback gespeichert",
                        target=cv_data.get("name", "") or selected["filename"],
                        result_type=category,
                    )
                    st.success("Feedback gespeichert.")

    with st.expander("Technische Details (JSON)"):
        st.json(cv_data)


# ---------------------------------------------------------------------------
# Render: Audit Log
# ---------------------------------------------------------------------------


def render_audit_log() -> None:
    page_header(
        "Audit Log",
        "Jeder Schritt <em>nachvollziehbar</em>.",
        "Vollständige Timeline der Agentenaktivität, plus gesammeltes Feedback "
        "und Tabellenansicht zum Export.",
    )

    log_entries = load_audit_log()
    tab_tl, tab_fb, tab_tab = st.tabs(["Timeline", "Feedback", "Tabellenansicht"])

    with tab_tl:
        st.caption(f"Autonomie-Stufe: {AUTONOMY_LEVEL}")
        if not log_entries:
            st.info("Audit-Log ist leer.")
        else:
            tl_items = []
            for entry in log_entries[-50:][::-1]:
                ts = entry.get("timestamp", "")
                t_disp = ts.split("T", 1)[1][:5] if "T" in ts else ts
                action = entry.get("action", "")
                target = entry.get("target", "")
                result = entry.get("result_type", "")
                sub = " · ".join(x for x in [target, result] if x)
                kind = ""
                a_l = action.lower()
                r_l = (result or "").lower()
                if "fehler" in a_l or "fehler" in r_l:
                    kind = "is-err"
                elif "erkannt" in a_l or "extrahiert" in a_l or "abgeglichen" in a_l:
                    kind = "is-ok"
                elif "klärung" in a_l or "rückfrage" in a_l:
                    kind = "is-warn"
                tl_items.append(
                    f'<div class="tl-item {kind}"><div class="tl-dot"></div>'
                    f'<div class="tl-time">{t_disp}</div>'
                    f'<div class="tl-action">{action}</div>'
                    f'<div class="tl-target">{sub}</div></div>'
                )
            st.markdown(
                f'<div class="tl">{"".join(tl_items)}</div>',
                unsafe_allow_html=True,
            )

    with tab_fb:
        st.caption("Feedback verbessert nur die Extraktion, nicht die Auswahl.")
        entries = load_feedback()
        if entries:
            st.dataframe(
                pd.DataFrame(entries),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(entries)} Einträge.")
        else:
            st.info("Noch kein Feedback vorhanden.")

    with tab_tab:
        if log_entries:
            st.dataframe(
                pd.DataFrame(log_entries),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Audit-Log ist leer.")


# ---------------------------------------------------------------------------
# Render: Marketing / Landing Page
# ---------------------------------------------------------------------------


def render_marketing_nav() -> None:
    cols = st.columns([3.6, 1.0, 1.25])
    with cols[0]:
        st.markdown(
            f"""
            <div class="shell-nav mkt-nav">
              <div class="shell-brand">
                <div class="shell-mark">{ic("sparkles", "md")}</div>
                <div class="shell-name">Recruiting&nbsp;AI<small>AI Recruiting Platform</small></div>
              </div>
              <div class="links">
                <a class="mkt-nav-link" href="#produkt">Produkt</a>
                <a class="mkt-nav-link" href="#workflow">Workflow</a>
                <a class="mkt-nav-link" href="#features">Features</a>
                <a class="mkt-nav-link" href="#sicherheit">Sicherheit</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("Dashboard", key="mkt_nav_dash", use_container_width=True):
            goto("dashboard")
    with cols[2]:
        if st.button("Demo starten", key="mkt_nav_demo", type="primary", use_container_width=True):
            goto("recruiting")


def _lp_head(eyebrow: str, title_html: str, sub: str) -> None:
    st.markdown(
        f'<div class="lp-sec"><div class="lp-head">'
        f'<span class="lp-eyebrow center">{eyebrow}</span>'
        f'<h2 class="lp-h2">{title_html}</h2>'
        f'<div class="lp-sub">{sub}</div></div></div>',
        unsafe_allow_html=True,
    )


# ---- Produkt-Mockups (visuelle Anker) --------------------------------------


def _mock_dashboard_html() -> str:
    tc = len(st.session_state.candidates)
    tg = sum(
        len((c.get("quality") or {}).get("missing_information") or [])
        + len((c.get("quality") or {}).get("unclear_information") or [])
        for c in st.session_state.candidates
    )
    tq = sum(
        len((c.get("followups") or {}).get("questions") or [])
        for c in st.session_state.candidates
    )
    return (
        '<div class="mock"><div class="mock-bar"><div class="mock-dots"><i></i><i></i><i></i></div>'
        '<div class="mock-url">recruiting-ai.app / dashboard</div></div>'
        '<div class="mock-body"><div class="mk-kpis">'
        f'<div class="mk-kpi"><div class="l">Bewerbungen</div><div class="v">{tc}</div></div>'
        f'<div class="mk-kpi"><div class="l">Klärungsbedarf</div><div class="v">{tg}</div></div>'
        f'<div class="mk-kpi"><div class="l">Rückfragen</div><div class="v">{tq}</div></div>'
        '</div><div class="mk-chart">'
        '<div class="mk-col c1"></div><div class="mk-col c2"></div><div class="mk-col c3 alt"></div>'
        '<div class="mk-col c4"></div><div class="mk-col c5 alt"></div><div class="mk-col c6"></div>'
        '<div class="mk-col c7"></div></div></div></div>'
    )


def _mock_feed_html() -> str:
    log = load_audit_log()[-5:][::-1]
    if not log:
        log = [
            {"action": "Stellenprofil analysiert", "target": "Senior Python Entwickler", "timestamp": "T09:41"},
            {"action": "CV analysiert", "target": "Anna Becker", "timestamp": "T09:41"},
            {"action": "Anforderungen abgeglichen", "target": "6 gefunden", "timestamp": "T09:42"},
            {"action": "Informationslücken erkannt", "target": "2 offen", "timestamp": "T09:42"},
            {"action": "Rückfragen erzeugt", "target": "3 vorbereitet", "timestamp": "T09:43"},
        ]
    rows = ""
    for e in log:
        ts = e.get("timestamp", "")
        t = ts.split("T", 1)[1][:5] if "T" in ts else ts
        action = e.get("action", "")
        target = e.get("target", "")
        kind, icn = _activity_kind(action)
        cls = "ok" if kind == "is-success" else ("warn" if kind == "is-warn" else "")
        rows += (
            f'<div class="mk-feed {cls}"><div class="fi">{ic(icn, "sm")}</div>'
            f'<div class="fb"><b>{action}</b><small>{target}</small></div>'
            f'<div class="ft">{t}</div></div>'
        )
    return (
        '<div class="mock"><div class="mock-bar"><div class="mock-dots"><i></i><i></i><i></i></div>'
        '<div class="mock-url">recruiting-ai.app / activity</div></div>'
        f'<div class="mock-body">{rows}</div></div>'
    )


def _mock_candidates_html() -> str:
    data = st.session_state.candidates[:3]
    rows = ""
    if not data:
        sample = [
            ("Anna Becker", "Frontend-Entwicklerin", ["React", "TypeScript", "CSS"], "8 / 9"),
            ("Mehmet Yılmaz", "DevOps Engineer", ["AWS", "Docker", "Python"], "6 / 9"),
            ("Clara Wolf", "Data Analyst", ["SQL", "Python", "BI"], "7 / 9"),
        ]
        for nm, rl, skills, m in sample:
            chips = "".join(f'<span class="badge badge-info">{s}</span>' for s in skills)
            rows += (
                f'<div class="mk-cand"><div class="app-av">{initials_for(nm)}</div>'
                f'<div><div class="nm">{nm}</div><div class="rl">{rl}</div>'
                f'<div class="chips">{chips}</div></div>'
                f'<div class="mk-match"><b>{m}</b><small>gefunden</small></div></div>'
            )
    else:
        jp = st.session_state.job_profile
        for c in data:
            d = c["data"]
            nm = d.get("name") or "(ohne Name)"
            rl = d["experience"][0].get("role", "") if d.get("experience") else c["filename"]
            skills = (d.get("skills") or [])[:3]
            chips = "".join(
                f'<span class="badge badge-info">{s}</span>' for s in skills
            ) or '<span class="badge badge-muted">—</span>'
            if jp:
                rc = status_counts(evaluate_candidate_requirements(jp, d))
                tot = rc["Gefunden"] + rc["Teilweise gefunden"] + rc["Nicht gefunden"]
                m = f'{rc["Gefunden"]} / {tot}' if tot else "—"
            else:
                m = "—"
            rows += (
                f'<div class="mk-cand"><div class="app-av">{initials_for(nm)}</div>'
                f'<div><div class="nm">{nm}</div><div class="rl">{rl}</div>'
                f'<div class="chips">{chips}</div></div>'
                f'<div class="mk-match"><b>{m}</b><small>gefunden</small></div></div>'
            )
    return (
        '<div class="mock"><div class="mock-bar"><div class="mock-dots"><i></i><i></i><i></i></div>'
        '<div class="mock-url">recruiting-ai.app / kandidaten</div></div>'
        f'<div class="mock-body"><div class="mk-list">{rows}</div></div></div>'
    )


def _mock_audit_html() -> str:
    log = load_audit_log()[-5:][::-1]
    if not log:
        log = [
            {"action": "Stellenprofil analysiert", "target": "Senior Python Entwickler", "timestamp": "T09:41"},
            {"action": "CV analysiert", "target": "Anna Becker", "timestamp": "T09:41"},
            {"action": "Anforderungen abgeglichen", "target": "6 gefunden, 2 teilweise", "timestamp": "T09:42"},
            {"action": "Klärungspunkte erkannt", "target": "2 Klärungspunkte", "timestamp": "T09:42"},
            {"action": "Rückfragen erzeugt", "target": "3 vorbereitet", "timestamp": "T09:43"},
        ]
    items = ""
    for e in log:
        ts = e.get("timestamp", "")
        t = ts.split("T", 1)[1][:5] if "T" in ts else ts
        action = e.get("action", "")
        target = e.get("target", "")
        a_l = action.lower()
        kind = "is-err" if "fehler" in a_l else (
            "is-warn" if ("klärung" in a_l or "rückfrage" in a_l) else "is-ok"
        )
        items += (
            f'<div class="tl-item {kind}"><div class="tl-dot"></div>'
            f'<div class="tl-time">{t}</div><div class="tl-action">{action}</div>'
            f'<div class="tl-target">{target}</div></div>'
        )
    return (
        '<div class="mock"><div class="mock-bar"><div class="mock-dots"><i></i><i></i><i></i></div>'
        '<div class="mock-url">recruiting-ai.app / audit</div></div>'
        f'<div class="mock-body"><div class="tl">{items}</div></div></div>'
    )


def _workflow_timeline_html() -> str:
    steps = [
        ("upload", "Stelle hochladen", "Stellenanzeige einfügen."),
        ("file", "CV analysieren", "Lebenslauf strukturieren."),
        ("scale", "Qualifikationen prüfen", "Anforderungen abgleichen."),
        ("puzzle", "Informationslücken erkennen", "Fehlende Angaben sichtbar machen."),
        ("message", "Rückfragen vorbereiten", "Höflich vorformuliert."),
        ("check-circle", "Entscheidung treffen", "Der Mensch entscheidet."),
    ]
    nodes = ""
    for i, (icn, name, desc) in enumerate(steps):
        final = "final" if i == len(steps) - 1 else ""
        nodes += (
            f'<div class="node {final}"><div class="dot">{ic(icn, "md")}</div>'
            f"<div><b>{name}</b><small>{desc}</small></div></div>"
        )
    return f'<div class="tl-flow">{nodes}</div>'


# ---- Immersive Intro (100vh Splash, scroll to explore) ---------------------

_INTRO_STYLE = """
<style>
[data-testid="stMain"] { overflow-x: clip; }
.intro {
  position: relative;
  width: 100vw; left: 50%; margin-left: -50vw; margin-right: -50vw;
  min-height: 100vh; margin-top: -1.4rem;
  display: flex; align-items: center; justify-content: center;
}
.intro-bg { position: absolute; inset: 0; z-index: 0; overflow: hidden; }
.intro-grad {
  position: absolute; inset: -2px; z-index: 0;
  background: linear-gradient(125deg, #EEF0FF 0%, #E9EDFF 22%, #E5FBF6 50%, #F1ECFF 74%, #EEF0FF 100%);
  background-size: 300% 300%; animation: introShift 22s ease infinite;
}
@keyframes introShift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.intro-glow { position: absolute; border-radius: 50%; filter: blur(74px); will-change: transform; }
.intro-glow.g1 { width: 540px; height: 540px; left: 6%; top: 10%; background: radial-gradient(closest-side, rgba(111,110,255,0.55), transparent 70%); animation: introF1 17s ease-in-out infinite alternate; }
.intro-glow.g2 { width: 580px; height: 580px; right: 4%; top: 16%; background: radial-gradient(closest-side, rgba(20,184,166,0.42), transparent 70%); animation: introF2 21s ease-in-out infinite alternate; }
.intro-glow.g3 { width: 500px; height: 500px; left: 38%; bottom: 0%; background: radial-gradient(closest-side, rgba(37,99,235,0.40), transparent 70%); animation: introF3 25s ease-in-out infinite alternate; }
@keyframes introF1 { from { transform: translate3d(0,0,0); } to { transform: translate3d(60px,40px,0); } }
@keyframes introF2 { from { transform: translate3d(0,0,0); } to { transform: translate3d(-52px,28px,0); } }
@keyframes introF3 { from { transform: translate3d(0,0,0); } to { transform: translate3d(28px,-42px,0); } }
.intro-particles { position: absolute; inset: 0; pointer-events: none; }
.intro-particles .pt { position: absolute; border-radius: 50%; background: rgba(79,70,229,0.55); box-shadow: 0 0 10px rgba(79,70,229,0.55); opacity: .5; animation: ptFloat 9s ease-in-out infinite; }
.intro-particles .p1 { width: 7px; height: 7px; left: 18%; top: 30%; animation-delay: 0s; }
.intro-particles .p2 { width: 5px; height: 5px; left: 30%; top: 64%; background: rgba(20,184,166,0.6); box-shadow: 0 0 10px rgba(20,184,166,0.6); animation-delay: 1.2s; }
.intro-particles .p3 { width: 9px; height: 9px; left: 72%; top: 26%; animation-delay: 2.1s; }
.intro-particles .p4 { width: 6px; height: 6px; left: 82%; top: 58%; background: rgba(37,99,235,0.6); box-shadow: 0 0 10px rgba(37,99,235,0.6); animation-delay: 3.0s; }
.intro-particles .p5 { width: 4px; height: 4px; left: 50%; top: 18%; animation-delay: 1.7s; }
.intro-particles .p6 { width: 6px; height: 6px; left: 60%; top: 74%; animation-delay: 2.6s; }
.intro-particles .p7 { width: 5px; height: 5px; left: 12%; top: 72%; background: rgba(20,184,166,0.6); box-shadow: 0 0 10px rgba(20,184,166,0.6); animation-delay: 0.6s; }
@keyframes ptFloat { 0%,100% { transform: translateY(-14px); opacity: .35; } 50% { transform: translateY(14px); opacity: .75; } }

.intro-inner { position: relative; z-index: 2; text-align: center; padding: 0 24px; }
.intro-card {
  display: inline-flex; flex-direction: column; align-items: center;
  padding: 60px 72px;
  background: rgba(255,255,255,0.42);
  backdrop-filter: blur(26px) saturate(140%); -webkit-backdrop-filter: blur(26px) saturate(140%);
  border: 1px solid rgba(255,255,255,0.6);
  border-radius: 36px;
  box-shadow: 0 34px 90px rgba(31,41,55,0.14), inset 0 1px 0 rgba(255,255,255,0.7);
}
.intro-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; letter-spacing: .06em; color: #4F46E5;
  background: rgba(79,70,229,0.08); border: 1px solid rgba(79,70,229,0.18);
  padding: 7px 16px; border-radius: 999px; margin-bottom: 26px;
}
.intro-eyebrow .ic, .intro-eyebrow .ic svg { width: 15px; height: 15px; }
.intro-title {
  font-family: var(--display); font-weight: 800;
  font-size: clamp(56px, 11vw, 128px); line-height: 0.92; letter-spacing: -0.045em;
  margin: 0 0 18px;
  background: linear-gradient(120deg, #4F46E5 0%, #6F6EFF 42%, #14B8A6 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.intro-sub { font-size: clamp(18px, 2.4vw, 26px); font-weight: 500; color: #33414E; max-width: 600px; margin: 0 auto; line-height: 1.4; }

.intro-scroll {
  position: absolute; left: 50%; bottom: 5vh; transform: translateX(-50%); z-index: 2;
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  color: #64748B; font-size: 12px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
}
.intro-mouse { width: 26px; height: 42px; border: 2px solid rgba(15,23,42,0.28); border-radius: 14px; position: relative; }
.intro-mouse i { position: absolute; left: 50%; top: 8px; width: 4px; height: 8px; margin-left: -2px; border-radius: 2px; background: #4F46E5; animation: introWheel 1.7s ease-in-out infinite; }
@keyframes introWheel { 0% { transform: translateY(0); opacity: 1; } 70% { transform: translateY(13px); opacity: 0; } 100% { opacity: 0; } }
.intro-chev { display: flex; align-items: center; color: #94A3B8; animation: introHint 2.2s ease-in-out infinite; }
.intro-chev .ic, .intro-chev .ic svg { width: 18px; height: 18px; }
@keyframes introHint { 0%,100% { transform: translateY(0); opacity: .6; } 50% { transform: translateY(6px); opacity: 1; } }

/* Scroll-reveal: hero exits upward + shrinks while the site fades in below */
@supports (animation-timeline: view()) {
  .intro-card {
    animation: introExit linear both;
    animation-timeline: view();
    animation-range: exit 0% exit 100%;
  }
  .intro-scroll {
    animation: introFade linear both;
    animation-timeline: view();
    animation-range: exit 0% exit 35%;
  }
}
@keyframes introExit { to { transform: translateY(-72px) scale(0.9); opacity: 0; } }
@keyframes introFade { to { opacity: 0; } }

@media (max-width: 640px) { .intro-card { padding: 40px 28px; border-radius: 28px; } }
@media (prefers-reduced-motion: reduce) {
  .intro-grad, .intro-glow, .intro-particles .pt, .intro-mouse i, .intro-chev, .intro-scroll, .intro-card { animation: none !important; }
}
</style>
"""


def _intro_markup() -> str:
    return (
        '<div class="intro"><div class="intro-bg">'
        '<div class="intro-grad"></div>'
        '<div class="intro-glow g1"></div><div class="intro-glow g2"></div>'
        '<div class="intro-glow g3"></div>'
        '<div class="intro-particles">'
        '<span class="pt p1"></span><span class="pt p2"></span><span class="pt p3"></span>'
        '<span class="pt p4"></span><span class="pt p5"></span><span class="pt p6"></span>'
        '<span class="pt p7"></span></div></div>'
        '<div class="intro-inner"><div class="intro-card">'
        f'<span class="intro-eyebrow">{ic("sparkles", "sm")} AI Recruiting Platform</span>'
        '<h1 class="intro-title">Recruiting&nbsp;AI</h1>'
        '<div class="intro-sub">Recruiting ohne stundenlanges Lebenslauflesen.</div>'
        "</div></div>"
        '<div class="intro-scroll"><span>Scroll to explore</span>'
        '<span class="intro-mouse"><i></i></span>'
        f'<span class="intro-chev">{ic("chevron-down", "sm")}</span></div>'
        "</div>"
    )


# ---- Hero v3: CSS-Gradient + Glows, Spline rechts (transparent) ------------

_HERO2_STYLE = """
<style>
/* Full-bleed gradient backdrop + soft glows, painted behind the hero */
.hero-bg { position: relative; z-index: -1; height: 0; }
.hero-bg::before {
  content: ""; position: absolute; pointer-events: none;
  left: 50%; transform: translateX(-50%); top: -32px;
  width: 100vw; height: 660px;
  background:
    radial-gradient(440px 340px at 14% 94%, rgba(37,99,235,0.22), transparent 70%),
    radial-gradient(480px 360px at 86% 6%, rgba(124,108,255,0.24), transparent 70%),
    linear-gradient(135deg, #F8FCFF 0%, #F7F5FF 50%, #EEF2FF 100%);
  -webkit-mask-image: linear-gradient(to bottom, #000 0%, #000 68%, transparent 100%);
  mask-image: linear-gradient(to bottom, #000 0%, #000 68%, transparent 100%);
}
.hero2-text { padding: 24px 8px 0 4px; animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both; }
.hero2-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; letter-spacing: .04em; color: #4F46E5;
  background: rgba(255,255,255,0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(79,70,229,0.18); padding: 7px 15px; border-radius: 999px;
  box-shadow: 0 8px 22px rgba(79,70,229,0.12);
}
.hero2-eyebrow .ic, .hero2-eyebrow .ic svg { width: 15px; height: 15px; }
.hero2-title {
  font-family: var(--display); font-weight: 800;
  font-size: clamp(40px, 4.4vw, 60px); line-height: 1.03; letter-spacing: -0.032em;
  color: #0B1020; margin: 24px 0 18px; max-width: 560px;
}
.hero2-title em {
  font-style: normal;
  background: linear-gradient(120deg, #4F46E5 0%, #6F6EFF 45%, #14B8A6 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero2-sub { font-size: clamp(17px, 1.4vw, 20px); line-height: 1.6; color: #475569; max-width: 490px; margin-bottom: 28px; }
.hero2-meta { display: flex; flex-wrap: wrap; gap: 10px; }
.hero2-meta span {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; color: #334155;
  background: rgba(255,255,255,0.55); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.7); padding: 8px 14px; border-radius: 12px;
  box-shadow: 0 8px 18px rgba(15,23,42,0.06);
}
.hero2-meta .ic { color: #14B8A6; }
.hero2-meta .ic, .hero2-meta .ic svg { width: 15px; height: 15px; }
@media (max-width: 1080px) {
  .hero-bg::before { height: 1180px; }
  .hero2-text { padding-top: 6px; }
  .hero2-title { font-size: clamp(34px, 7vw, 46px); }
}
</style>
"""

_SPLINE_HERO_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: transparent; overflow: hidden; }
  #wrap { position: relative; width: 100%; height: 100%; background: transparent; }
  spline-viewer { width: 100%; height: 100%; display: block; background: transparent; }
  spline-viewer::part(logo) { display: none !important; }
</style>
<script type="module" src="https://unpkg.com/@splinetool/viewer@1/build/spline-viewer.js"></script>
</head>
<body>
  <div id="wrap">
    <spline-viewer
      url="https://prod.spline.design/Ji0hiX2hb-mU5zX1/scene.splinecode"
      loading-anim-type="none">
    </spline-viewer>
  </div>
</body></html>
"""


# ---- Merged Hero (Intro + Hero): 100vh full-bleed, Spline als BG ----------

_HERO_MERGED_STYLE = """
<style>
.hero-merged-marker { display: none; }

/* The hero container: full-bleed 100vh with CSS gradient + soft glows */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) {
  position: relative;
  width: 100vw; left: 50%; margin-left: -50vw;
  min-height: 100vh; margin-top: -1.4rem;
  padding: 64px clamp(28px, 5vw, 110px) 120px;
  background: linear-gradient(135deg, #F8FCFF 0%, #F7F5FF 50%, #EEF2FF 100%);
  overflow: hidden;
  display: flex; flex-direction: column; justify-content: center; gap: 22px;
}
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker)::before {
  content: ""; position: absolute; pointer-events: none;
  right: -160px; top: -180px; width: 560px; height: 560px;
  background: radial-gradient(closest-side, rgba(124,108,255,0.34), transparent 70%);
  filter: blur(60px); z-index: 0;
}
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker)::after {
  content: ""; position: absolute; pointer-events: none;
  left: -160px; bottom: -180px; width: 580px; height: 580px;
  background: radial-gradient(closest-side, rgba(37,99,235,0.30), transparent 70%);
  filter: blur(60px); z-index: 0;
}

/* Marker takes no flow space */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) > div:first-child {
  position: absolute; opacity: 0; pointer-events: none;
  height: 0; min-height: 0; margin: 0; padding: 0;
}

/* Spline embed (the iframe + its wrappers) becomes the absolute background */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="element-container"]:has([data-testid="stIFrame"]),
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="element-container"]:has([data-testid="stCustomComponentV1"]) {
  position: absolute !important; inset: 0 !important;
  width: 100% !important; height: 100% !important;
  z-index: 1 !important; margin: 0 !important; padding: 0 !important;
}
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="stIFrame"],
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="stCustomComponentV1"] {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  width: 100% !important; height: 100% !important;
}
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) iframe {
  width: 100% !important; height: 100% !important;
  background: transparent !important; border: 0 !important; box-shadow: none !important; display: block;
}

/* Foreground content layers above Spline */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) > div:not(:first-child):not(:nth-child(2)) {
  position: relative; z-index: 3;
}

/* Soft left-side readability veil so the headline is always crisp */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) > div:nth-child(2)::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 2;
  background: linear-gradient(90deg, rgba(248,252,255,0.72) 0%, rgba(248,252,255,0.30) 28%, transparent 52%);
}

/* Text + CTAs constrained left column width */
.hero-merged-text {
  max-width: 620px;
  animation: fadeUp .8s cubic-bezier(.2,.7,.2,1) both;
}
.hero-merged-text .hero2-title { margin-top: 26px; margin-bottom: 22px; }
.hero-merged-text .hero2-sub { margin-bottom: 0; max-width: 540px; }

.hero-merged-meta {
  display: flex; flex-wrap: wrap; gap: 10px; max-width: 620px;
  animation: fadeUp .8s cubic-bezier(.2,.7,.2,1) both; animation-delay: .28s;
}
.hero-merged-meta span {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; color: #334155;
  background: rgba(255,255,255,0.65); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,0.7); padding: 8px 14px; border-radius: 12px;
  box-shadow: 0 8px 18px rgba(15,23,42,0.06);
}
.hero-merged-meta .ic { color: #14B8A6; }
.hero-merged-meta .ic, .hero-merged-meta .ic svg { width: 15px; height: 15px; }

/* CTA columns row width */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="stHorizontalBlock"] {
  max-width: 620px;
}

/* Scroll-to-explore floats absolutely at bottom center of the hero */
[data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) [data-testid="element-container"]:has(.hero-merged-scroll) {
  position: absolute !important;
  left: 0; right: 0; bottom: 36px;
  z-index: 4 !important;
  margin: 0 !important;
  display: flex !important; justify-content: center !important;
  pointer-events: none;
}
.hero-merged-scroll {
  display: inline-flex; flex-direction: column; align-items: center; gap: 10px;
  color: #64748B; font-size: 12px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
}
.hero-merged-scroll .intro-mouse { width: 26px; height: 42px; border: 2px solid rgba(15,23,42,0.28); border-radius: 14px; position: relative; }
.hero-merged-scroll .intro-mouse i { position: absolute; left: 50%; top: 8px; width: 4px; height: 8px; margin-left: -2px; border-radius: 2px; background: #4F46E5; animation: introWheel 1.7s ease-in-out infinite; }
.hero-merged-scroll .intro-chev { display: flex; align-items: center; color: #94A3B8; animation: introHint 2.2s ease-in-out infinite; }
.hero-merged-scroll .intro-chev .ic, .hero-merged-scroll .intro-chev .ic svg { width: 18px; height: 18px; }

@keyframes introWheel { 0% { transform: translateY(0); opacity: 1; } 70% { transform: translateY(13px); opacity: 0; } 100% { opacity: 0; } }
@keyframes introHint { 0%,100% { transform: translateY(0); opacity: .6; } 50% { transform: translateY(6px); opacity: 1; } }
@keyframes introExit { to { transform: translateY(-72px) scale(0.94); opacity: 0; } }
@keyframes introFade { to { opacity: 0; } }

/* Scroll-linked exit (Apple/Linear feel) — content drifts up + fades as
   the hero scrolls past, scroll hint vanishes first */
@supports (animation-timeline: view()) {
  .hero-merged-text, .hero-merged-meta {
    animation: introExit linear both;
    animation-timeline: view();
    animation-range: exit 0% exit 100%;
  }
  .hero-merged-scroll {
    animation: introFade linear both;
    animation-timeline: view();
    animation-range: exit 0% exit 35%;
  }
}

@media (max-width: 1080px) {
  [data-testid="stVerticalBlock"]:has(> div:first-child .hero-merged-marker) {
    padding: 48px 24px 120px;
  }
}
@media (prefers-reduced-motion: reduce) {
  .hero-merged-scroll .intro-mouse i, .hero-merged-scroll .intro-chev,
  .hero-merged-text, .hero-merged-meta { animation: none !important; }
}
</style>
"""


# ---- Landing Page (Story-Layout, wechselnder Aufbau) -----------------------


def render_home() -> None:
    # HERO = Intro/Entrance — the only hero section. 100vh full-bleed,
    # CSS gradient + soft glows, transparent Spline scene as integrated
    # background (no white box / iframe-like container), headline + sub +
    # CTAs + meta + scroll-to-explore on top. Animations and scroll-reveal
    # are preserved.
    st.markdown(_HERO_MERGED_STYLE, unsafe_allow_html=True)
    with st.container():
        st.markdown(
            '<div id="produkt" class="hero-merged-marker"></div>',
            unsafe_allow_html=True,
        )
        components.html(_SPLINE_HERO_HTML, height=720)
        st.markdown(
            f'<div class="hero-merged-text">'
            f'<span class="hero2-eyebrow">{ic("sparkles","sm")} AI Recruiting · Multi-Agent</span>'
            f'<h1 class="hero2-title">Recruiting ohne <em>stundenlanges</em> Lebenslauflesen.</h1>'
            f'<div class="hero2-sub">Recruiting AI analysiert Bewerbungen, prüft Qualifikationen '
            f"und erkennt Informationslücken &ndash; ohne automatische Personalentscheidung.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        hc1, hc2, _hsp = st.columns([1.3, 1.3, 4])
        with hc1:
            if st.button("Demo starten", key="hero_demo", type="primary", use_container_width=True):
                goto("recruiting")
        with hc2:
            if st.button("Mehr erfahren", key="hero_more", use_container_width=True):
                goto("dashboard")
        st.markdown(
            f'<div class="hero-merged-meta">'
            f'<span><span class="ic">{ic("bot","sm")}</span> Multi-Agent</span>'
            f'<span><span class="ic">{ic("shield","sm")}</span> Human-in-the-Loop</span>'
            f'<span><span class="ic">{ic("log","sm")}</span> Auditierbar</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hero-merged-scroll"><span>Scroll to explore</span>'
            f'<span class="intro-mouse"><i></i></span>'
            f'<span class="intro-chev">{ic("chevron-down","sm")}</span></div>',
            unsafe_allow_html=True,
        )

    render_marketing_nav()

    # 2 — TRUST STRIP (dünnes Band)
    st.markdown(
        f'<div class="lp-strip">'
        f'<span><span class="ic">{ic("bot", "sm")}</span> Multi-Agent Workflow</span>'
        f'<span><span class="ic">{ic("shield", "sm")}</span> Human-in-the-Loop</span>'
        f'<span><span class="ic">{ic("log", "sm")}</span> Audit Log</span>'
        f'<span><span class="ic">{ic("check-circle", "sm")}</span> Keine automatische Entscheidung</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # 3 — PROBLEM: Feature-Grid (nummeriert)
    _lp_head(
        "Das Problem",
        "Warum Recruiting heute <em>Zeit</em> kostet",
        "Kleine Teams ohne HR-Abteilung verlieren Stunden mit manueller Sichtung.",
    )
    st.markdown(
        f'<div class="grid-4">'
        f'<div class="lp-card"><div class="num">01</div><div class="ico">{ic("clock", "md")}</div>'
        f"<h3>5–10 Stunden Sichtung</h3><p>Bewerbungsstapel manuell zu lesen bindet wertvolle "
        f"Zeit der Geschäftsführung.</p></div>"
        f'<div class="lp-card"><div class="num">02</div><div class="ico">{ic("file", "md")}</div>'
        f"<h3>Verschiedene CV-Formate</h3><p>Jeder Lebenslauf ist anders aufgebaut — "
        f"Vergleichbarkeit muss erst entstehen.</p></div>"
        f'<div class="lp-card"><div class="num">03</div><div class="ico">{ic("puzzle", "md")}</div>'
        f"<h3>Fehlende Informationen</h3><p>Wichtige Angaben fehlen oft — und fallen erst "
        f"spät im Prozess auf.</p></div>"
        f'<div class="lp-card"><div class="num">04</div><div class="ico">{ic("users", "md")}</div>'
        f"<h3>Keine HR-Abteilung</h3><p>Ohne eigenes Recruiting-Team bleibt die Last bei "
        f"wenigen Personen.</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 4 — LÖSUNG: Text links / Workflow-Timeline rechts
    st.markdown(
        f'<div id="workflow" class="lp-sec"><div class="lp-split">'
        f'<div><span class="lp-eyebrow">Der Ablauf</span>'
        f'<h2 class="lp-h2">So arbeitet <em>Recruiting AI</em></h2>'
        f'<div class="lp-sub">Sechs transparente Schritte vom Upload bis zur Entscheidung.</div>'
        f'<ul class="lp-list">'
        f"<li>Stellenprofil &amp; Lebensläufe werden getrennt strukturiert</li>"
        f"<li>Jede Anforderung erhält eine Belegstelle im Lebenslauf</li>"
        f"<li>Rückfragen werden vorbereitet, nie automatisch versendet</li>"
        f"</ul></div>"
        f'<div class="lp-visual">{_workflow_timeline_html()}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # 5 — WORKSPACE ENTRY: zentriertes Band
    st.markdown(
        '<div class="lp-sec"><div class="lp-cta"><div class="lp-cta-in">'
        "<h2>Bereit für die Analyse?</h2>"
        "<p>Öffnen Sie den Recruiting Workspace und starten Sie mit Stellenprofil "
        "und Lebensläufen.</p></div></div></div>",
        unsafe_allow_html=True,
    )
    we1, we2, we3 = st.columns([1.6, 1.7, 1.6])
    with we2:
        if st.button("Recruiting Workspace öffnen", key="ws_entry", type="primary", use_container_width=True):
            goto("recruiting")

    # 6 — DASHBOARD-VORSCHAU: Activity-Feed-Mockup links / Text rechts (reverse)
    st.markdown(
        f'<div class="lp-sec"><div class="lp-split reverse">'
        f'<div><span class="lp-eyebrow">Transparenz</span>'
        f'<h2 class="lp-h2">Volle Transparenz über <em>jeden Schritt</em></h2>'
        f'<div class="lp-sub">Ein Live-Activity-Feed zeigt jede Agentenaktion — '
        f"nachvollziehbar statt Black Box.</div>"
        f'<ul class="lp-list">'
        f"<li>Live-Feed jeder Agentenaktion in Echtzeit</li>"
        f"<li>Kennzahlen zu Bewerbungen, Lücken und Rückfragen</li>"
        f"<li>Jeder Schritt landet revisionssicher im Audit-Log</li>"
        f"</ul></div>"
        f'<div class="lp-visual">{_mock_feed_html()}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    dp1, dp2, _dp3 = st.columns([1.6, 1.2, 4])
    with dp1:
        if st.button("Zum Dashboard", key="prev_dash", use_container_width=True):
            goto("dashboard")

    # 7 — AGENTEN: horizontale Liste (anderer Aufbau als Grid)
    _lp_head(
        "Die Agenten",
        "Mehrere Agenten. <em>Ein</em> Workflow.",
        "Spezialisierte Agenten arbeiten zusammen — keiner trifft eine Entscheidung.",
    )
    agents = [
        ("clipboard", "Stellenprofil-Agent", "Extrahiert objektiv prüfbare Anforderungen aus der Stellenanzeige."),
        ("file", "CV-Agent", "Strukturiert jeden Lebenslauf in vergleichbare Felder mit Belegen."),
        ("scale", "Matching-Agent", "Gleicht Qualifikationen ab: gefunden, teilweise, nicht gefunden."),
        ("puzzle", "Informationslücken-Agent", "Erkennt fehlende und unklare Angaben statt sie zu raten."),
        ("message", "Rückfragen-Agent", "Bereitet höfliche Rückfragen vor — versendet wird nichts automatisch."),
    ]
    _arows = ""
    for _i, (_icn, _title, _desc) in enumerate(agents, start=1):
        _arows += (
            f'<div class="lp-agent"><div class="ico">{ic(_icn, "md")}</div>'
            f"<div><b>{_title}</b><small>{_desc}</small></div>"
            f'<div class="step">0{_i}</div></div>'
        )
    st.markdown(f'<div class="lp-sec tight"><div id="features" class="lp-agents">{_arows}</div></div>', unsafe_allow_html=True)

    # 8 — KANDIDATEN: Text links / Candidate-Mockup rechts
    st.markdown(
        f'<div class="lp-sec"><div class="lp-split">'
        f'<div><span class="lp-eyebrow">Kandidaten</span>'
        f'<h2 class="lp-h2">Bewerbungen auf <em>einen Blick</em> vergleichen</h2>'
        f'<div class="lp-sub">Strukturierte Kandidatenkarten statt unübersichtlicher Tabellen.</div>'
        f'<ul class="lp-list">'
        f"<li>Einheitliche Karten statt wirrer Tabellen</li>"
        f"<li>Skills, Sprachen und Zertifikate auf einen Blick</li>"
        f"<li>Kein Ranking, kein Score — nur Struktur</li>"
        f"</ul></div>"
        f'<div class="lp-visual">{_mock_candidates_html()}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    kc1, kc2, _kc3 = st.columns([1.6, 1.2, 4])
    with kc1:
        if st.button("Kandidaten ansehen", key="prev_kand", use_container_width=True):
            goto("kandidaten")

    # 9 — HUMAN-IN-THE-LOOP: zentrierter Flow
    _lp_head(
        "Vertrauen",
        "Der Mensch <em>entscheidet</em>.",
        "Der Agent liefert Daten. Die Verantwortung bleibt bei Ihnen.",
    )
    st.markdown(
        f'<div class="lp-sec tight"><div class="lp-hil">'
        f'<div class="node"><div class="ico">{ic("bot", "lg")}</div>'
        f"<b>Agent</b><small>Strukturiert &amp; prüft Informationen</small></div>"
        f'<div class="arr">{ic("arrow-right", "md")}</div>'
        f'<div class="node human"><div class="ico">{ic("user", "lg")}</div>'
        f"<b>Geschäftsführer</b><small>Sichtet die aufbereiteten Daten</small></div>"
        f'<div class="arr">{ic("arrow-right", "md")}</div>'
        f'<div class="node human"><div class="ico">{ic("check-circle", "lg")}</div>'
        f"<b>Entscheidung</b><small>Die Auswahl trifft der Mensch</small></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # 10 — AUDIT LOG: Timeline-Mockup links / Text rechts (reverse)
    st.markdown(
        f'<div class="lp-sec"><div class="lp-split reverse">'
        f'<div><span class="lp-eyebrow">Audit Log</span>'
        f'<h2 class="lp-h2">Jeder Schritt <em>nachvollziehbar</em></h2>'
        f'<div class="lp-sub">Eine vollständige Timeline jeder Agentenaktivität — '
        f"mit Zeitstempel und Ergebnis.</div>"
        f'<ul class="lp-list">'
        f"<li>Lückenlose Timeline jeder Analyse</li>"
        f"<li>Zeitstempel und Ergebnis pro Schritt</li>"
        f"<li>Exportierbar für Fairness und Compliance</li>"
        f"</ul></div>"
        f'<div class="lp-visual">{_mock_audit_html()}</div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    al1, al2, _al3 = st.columns([1.6, 1.2, 4])
    with al1:
        if st.button("Audit Log öffnen", key="prev_audit", use_container_width=True):
            goto("audit")

    # 11 — SICHERHEIT: Feature-Grid
    _lp_head(
        "Sicherheit",
        "Transparenz statt <em>Black Box</em>",
        "Kontrolle, Nachvollziehbarkeit und Fairness sind eingebaut.",
    )
    st.markdown(
        f'<div class="grid-4">'
        f'<div class="lp-card"><div class="ico teal">{ic("check-circle", "md")}</div>'
        f"<h3>Keine automatische Entscheidung</h3><p>Recruiting AI bewertet niemanden "
        f"und erstellt kein Ranking.</p></div>"
        f'<div class="lp-card"><div class="ico teal">{ic("log", "md")}</div>'
        f"<h3>Auditierbar</h3><p>Jeder Agentenschritt wird lückenlos protokolliert.</p></div>"
        f'<div class="lp-card"><div class="ico teal">{ic("search", "md")}</div>'
        f"<h3>Nachvollziehbar</h3><p>Jede Aussage verweist auf eine Belegstelle im "
        f"Lebenslauf.</p></div>"
        f'<div class="lp-card"><div class="ico teal">{ic("shield", "md")}</div>'
        f"<h3>Kontrolle beim Nutzer</h3><p>Sie behalten jederzeit die volle "
        f"Entscheidungshoheit.</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 12 — FINAL CTA
    st.markdown(
        '<div class="lp-sec"><div class="lp-cta"><div class="lp-cta-in">'
        "<h2>Starten Sie Ihre erste Bewerbungsanalyse.</h2>"
        "<p>In unter einer Minute zum strukturierten Überblick — ohne Setup, "
        "mit voller Kontrolle.</p></div></div></div>",
        unsafe_allow_html=True,
    )
    fc1, fc2, fc3 = st.columns([1.6, 1.8, 1.6])
    with fc2:
        if st.button("Recruiting Workspace öffnen", key="final_cta", type="primary", use_container_width=True):
            goto("recruiting")

    # 13 — FOOTER
    st.markdown(
        f'<div class="lp-foot">'
        f'<div class="brand"><span class="m">{ic("sparkles", "md")}</span><b>Recruiting&nbsp;AI</b></div>'
        f'<div class="links"><a href="#produkt">Impressum</a><a href="#produkt">Datenschutz</a>'
        f'<a href="#produkt">Kontakt</a></div>'
        f'<div class="copy">© 2026 Recruiting&nbsp;AI · Human-in-the-Loop · '
        f"Keine automatische Personalentscheidung.</div></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# App-Shell: Seite routen (Marketing-Landing oder funktionaler Workspace)
# ---------------------------------------------------------------------------

_page = st.session_state.nav_page

if _page == "home":
    render_home()
else:
    st.markdown('<div class="shell-nav-wrap">', unsafe_allow_html=True)
    render_top_nav()
    st.markdown("</div>", unsafe_allow_html=True)

    if _page == "dashboard":
        render_dashboard()
    elif _page == "recruiting":
        render_recruiting()
    elif _page == "kandidaten":
        render_kandidaten()
    elif _page == "audit":
        render_audit_log()
    else:
        render_dashboard()

    st.caption(
        "Der Mensch entscheidet. Recruiting AI strukturiert Informationen — "
        "die Entscheidung trifft der Geschäftsführer."
    )
