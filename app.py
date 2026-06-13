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
# Custom CSS — Dark Dashboard Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=Familjen+Grotesk:wght@400;500;600;700&display=swap');

    :root{
        --paper:#F4F1EA; --paper-2:#ECE6D9; --card:#FFFDF8;
        --ink:#17202A; --ink-2:#33414E; --muted:#6E6A5F; --soft:#8C8675;
        --line:#E5DECF; --line-2:#EFE9DC;
        --accent:#BE5B2A; --accent-d:#A14A1F; --accent-soft:#F4E7DA;
        --display:'Fraunces',Georgia,'Times New Roman',serif;
        --sans:'Familjen Grotesk',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        --ok:#1F7A4D; --warn:#B45309; --err:#B3261E; --info:#1D4ED8;
    }

    html, body, [class*="css"], .stApp, p, span, label, li, div, input, textarea, button {
        font-family: var(--sans);
    }
    .stApp { color: var(--ink); }

    /* ---- Background: warm paper with very soft grain ---- */
    [data-testid="stAppViewContainer"] {
        position: relative;
        background-color: var(--paper);
        background-image:
            radial-gradient(900px 520px at 100% -6%, rgba(190,91,42,0.05), transparent 60%),
            radial-gradient(820px 520px at -6% 4%, rgba(23,32,42,0.045), transparent 62%);
        background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"]::before {
        content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
        opacity: 0.05;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }
    [data-testid="stMain"] { background: transparent; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.4rem; max-width: 1180px; position: relative; z-index: 1; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: var(--ink); }

    /* ---- Headings / text ---- */
    h1, h2, h3, h4, h5, h6 { color: var(--ink) !important; font-family: var(--sans); letter-spacing: -0.012em; font-weight: 700; }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
    p, span, label, li { color: var(--ink-2); }
    a { color: var(--accent); text-decoration: none; }

    /* ---- Inputs ---- */
    .stTextInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        background: var(--card) !important; color: var(--ink) !important;
        border: 1px solid var(--line) !important; border-radius: 10px !important;
        font-family: var(--sans) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(190,91,42,0.14) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--paper-2) !important; border: 1.4px dashed #C9BFA9 !important;
        color: var(--muted) !important; border-radius: 12px !important;
    }

    /* ---- Buttons: default = ghost ink, primary = solid ink ---- */
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        font-family: var(--sans) !important; font-weight: 600 !important;
        border-radius: 10px !important; padding: 9px 18px !important;
        transition: all .16s ease; letter-spacing: .01em;
    }
    .stButton > button {
        background: transparent !important; color: var(--ink) !important;
        border: 1.4px solid var(--ink) !important; box-shadow: none !important;
    }
    .stButton > button:hover { background: var(--ink) !important; color: var(--paper) !important; transform: translateY(-1px); }
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stFormSubmitButton"] > button,
    .stDownloadButton > button {
        background: var(--ink) !important; color: #F7F2E8 !important;
        border: 1.4px solid var(--ink) !important;
    }
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important;
    }
    .stButton > button:disabled { opacity: .45 !important; transform: none !important; }

    /* ---- Tabs ---- */
    [data-testid="stTabs"] [role="tablist"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 6px; gap: 4px; flex-wrap: wrap;
    }
    [data-testid="stTabs"] [role="tab"] {
        color: var(--muted) !important; background: transparent !important;
        border-radius: 8px !important; padding: 8px 16px !important; font-weight: 600 !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: var(--accent-soft) !important; color: var(--accent-d) !important;
    }

    /* ---- Cards / expanders / dataframe ---- */
    [data-testid="stExpander"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 14px; box-shadow: 0 6px 22px rgba(23,32,42,0.05);
    }
    [data-testid="stExpander"] summary { color: var(--ink) !important; }
    [data-testid="stDataFrame"] {
        background: var(--card); border: 1px solid var(--line);
        border-radius: 14px; padding: 6px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card) !important; border: 1px solid var(--line) !important;
        border-radius: 18px; box-shadow: 0 10px 30px rgba(23,32,42,0.06);
    }
    [data-testid="stSpinner"] { color: var(--accent); }

    /* ---- Generic cards / titles ---- */
    .kmu-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 18px;
        padding: 22px 24px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(23,32,42,0.06);
    }
    .kmu-card-title {
        color: var(--ink); font-weight: 700; font-size: 18px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .kmu-card-title small { color: var(--accent); font-weight: 600; font-size: 13px; }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 18px;
        padding: 20px 22px; position: relative; overflow: hidden; min-height: 132px;
        box-shadow: 0 10px 30px rgba(23,32,42,0.06);
    }
    .kpi-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; }
    .kpi-icon {
        width: 40px; height: 40px; border-radius: 11px; display: inline-flex;
        align-items: center; justify-content: center; font-size: 19px; margin-bottom: 12px;
    }
    .kpi-card.teal::before { background: var(--ink); }
    .kpi-card.orange::before { background: var(--warn); }
    .kpi-card.blue::before { background: var(--info); }
    .kpi-card.green::before { background: var(--ok); }
    .kpi-card.red::before { background: var(--err); }
    .kpi-card.teal .kpi-icon { background: #E9EAEC; }
    .kpi-card.orange .kpi-icon { background: #FBEAD3; }
    .kpi-card.blue .kpi-icon { background: #E2E8FB; }
    .kpi-card.green .kpi-icon { background: #DCEFE4; }
    .kpi-value { font-size: 34px; font-weight: 800; line-height: 1.1; color: var(--ink); font-family: var(--display); }
    .kpi-value.teal { color: var(--ink); }
    .kpi-value.blue { color: var(--info); }
    .kpi-value.orange { color: var(--warn); }
    .kpi-value.green { color: var(--ok); }
    .kpi-value.red { color: var(--err); }
    .kpi-label { color: var(--muted); font-size: 13px; margin-top: 8px; line-height: 1.4; font-weight: 500; }

    /* ---- Dashboard header ---- */
    .dashboard-header { display: flex; justify-content: space-between; align-items: flex-end; margin: 6px 0 18px; }
    .dashboard-header h1 { font-size: 30px !important; margin: 0 !important; font-weight: 800; font-family: var(--display); letter-spacing: -0.01em; }
    .dashboard-header .subtitle { color: var(--muted); font-size: 14px; margin-top: 4px; }
    .kmu-avatar, .app-avatar {
        background: var(--ink); color: var(--paper); width: 34px; height: 34px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;
    }
    .kmu-topbar { display: none; }

    /* ---- Badges ---- */
    .kmu-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; margin: 1px 0; }
    .kmu-badge-ok { background: #DCEFE4; color: var(--ok); }
    .kmu-badge-warn { background: #FBEAD3; color: var(--warn); }
    .kmu-badge-err { background: #F7DEDC; color: var(--err); }
    .kmu-badge-info { background: var(--accent-soft); color: var(--accent-d); }
    .kmu-badge-muted { background: #ECE6D9; color: var(--muted); }

    /* ---- Lists ---- */
    .kmu-list-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid var(--line-2); gap: 12px; }
    .kmu-list-item:last-child { border-bottom: none; }
    .kmu-list-avatar { width: 38px; height: 38px; border-radius: 50%; background: var(--ink); color: var(--paper); display: inline-flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; flex-shrink: 0; }
    .kmu-list-name { font-weight: 600; color: var(--ink); font-size: 14px; }
    .kmu-list-file { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .kmu-logo { font-size: 17px; font-weight: 800; color: var(--ink); padding: 4px 8px 12px 8px; font-family: var(--display); }

    /* ---- Donut legend ---- */
    .donut-wrap { display: flex; align-items: center; gap: 24px; }
    .donut-legend { font-size: 13px; flex: 1; }
    .donut-legend-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; color: var(--ink-2); }
    .donut-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
    .donut-count { margin-left: auto; color: var(--muted); font-weight: 600; }

    /* ---- Question rows ---- */
    .question-row { background: var(--paper-2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; color: var(--ink); font-size: 14px; }
    .question-row.reviewed { border-left: 4px solid var(--ok); }

    /* ---- HiL line ---- */
    .hil-line { color: var(--ink-2); font-size: 13px; padding: 10px 16px; margin: 4px 0 14px 0; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 8px; font-weight: 500; }

    /* ---- Coverage box ---- */
    .coverage-box { background: var(--accent-soft); border: 1px solid #EBD8C6; border-radius: 12px; padding: 14px 18px; margin: 6px 0 4px 0; text-align: center; }
    .coverage-value { font-size: 26px; font-weight: 800; color: var(--accent-d); font-family: var(--display); }
    .coverage-label { color: var(--muted); font-size: 12px; margin-top: 4px; }

    /* ---- Timeline (Audit) ---- */
    .timeline { position: relative; margin: 6px 0 0 6px; padding-left: 22px; }
    .timeline::before { content: ""; position: absolute; left: 5px; top: 4px; bottom: 4px; width: 2px; background: var(--line); }
    .tl-item { position: relative; padding: 0 0 18px 4px; }
    .tl-dot { position: absolute; left: -22px; top: 3px; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); border: 2px solid var(--card); box-shadow: 0 0 0 2px var(--accent-soft); }
    .tl-time { color: var(--soft); font-size: 12px; font-weight: 600; }
    .tl-action { color: var(--ink); font-weight: 600; font-size: 14px; }
    .tl-target { color: var(--muted); font-size: 13px; }
    .section-title { font-size: 20px; font-weight: 800; color: var(--ink); margin: 6px 0 2px 0; font-family: var(--display); }
    .section-sub { color: var(--muted); font-size: 14px; margin-bottom: 16px; }

    /* ---- Upload tiles ---- */
    .upload-head { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
    .upload-ic { width: 42px; height: 42px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 20px; background: var(--accent-soft); }
    .upload-ic.green { background: #DCEFE4; }
    .upload-title { font-weight: 700; font-size: 16px; color: var(--ink); }
    .upload-desc { color: var(--muted); font-size: 13px; margin-bottom: 8px; }

    /* ---- Candidate cards ---- */
    .cand-name { font-weight: 700; font-size: 16px; color: var(--ink); }
    .cand-role { color: var(--muted); font-size: 13px; margin-top: 2px; }
    .cand-metric { font-size: 22px; font-weight: 800; color: var(--accent-d); font-family: var(--display); }
    .cand-metric-label { font-size: 11px; color: var(--soft); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }

    /* ---- Activity feed ---- */
    .feed-item { display: flex; align-items: center; gap: 11px; padding: 9px 2px; border-bottom: 1px solid var(--line-2); animation: fadeUpG .5s ease both; }
    .feed-item:last-child { border-bottom: none; }
    .feed-ic { width: 30px; height: 30px; border-radius: 9px; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; font-size: 14px; background: var(--accent-soft); }
    .feed-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
    .feed-body b { color: var(--ink); font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .feed-body small { color: var(--soft); font-size: 11.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .feed-time { color: var(--soft); font-size: 11.5px; font-weight: 600; flex-shrink: 0; }

    /* ---- Microinteractions ---- */
    @keyframes fadeUpG { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .kmu-card, .kpi-card, [data-testid="stVerticalBlockBorderWrapper"] { transition: transform .2s ease, box-shadow .2s ease; }
    .kpi-card:hover, .kmu-card:hover { transform: translateY(-4px); box-shadow: 0 18px 42px rgba(23,32,42,0.12); }
    .kmu-badge { animation: fadeUpG .45s ease both; }

    /* =====================================================================
       TOP NAVIGATION (masthead) + MARKETING PAGES
       ===================================================================== */
    .nav-brand { display: flex; align-items: center; gap: 11px; padding-top: 4px; }
    .nav-mark {
        width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
        background: var(--ink); color: var(--paper); display: inline-flex;
        align-items: center; justify-content: center; font-size: 19px;
        font-family: var(--display); font-weight: 700;
    }
    .nav-name { font-family: var(--display); font-weight: 700; font-size: 19px; color: var(--ink); line-height: 1; letter-spacing: -0.01em; }
    .nav-name small { display: block; font-family: var(--sans); font-weight: 500; font-size: 11px; color: var(--muted); letter-spacing: .14em; text-transform: uppercase; margin-top: 3px; }
    .nav-rule { border: none; border-top: 1px solid var(--line); margin: 4px 0 6px; }

    .eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: var(--accent-d); }
    .eyebrow::before { content: ""; width: 26px; height: 1.5px; background: var(--accent); display: inline-block; }

    /* ---- Marketing hero (stock image, Ken-Burns) ---- */
    .mk-hero { position: relative; border-radius: 26px; overflow: hidden; margin: 8px 0 18px; min-height: 520px; display: flex; align-items: center; box-shadow: 0 24px 60px rgba(23,32,42,0.20); border: 1px solid rgba(23,32,42,0.10); }
    .mk-hero-bg { position: absolute; inset: 0; background-size: cover; background-position: center; transform: scale(1.06); animation: kenburns 22s ease-in-out infinite alternate; z-index: 0; }
    @keyframes kenburns { from { transform: scale(1.04) translate(0,0); } to { transform: scale(1.14) translate(-2%, -2%); } }
    .mk-hero-ov { position: absolute; inset: 0; z-index: 1; background: linear-gradient(105deg, rgba(18,22,28,0.90) 0%, rgba(18,22,28,0.72) 42%, rgba(18,22,28,0.22) 78%, rgba(18,22,28,0.05) 100%); }
    .mk-hero-in { position: relative; z-index: 2; padding: 56px 56px; max-width: 760px; }
    .mk-eyebrow { color: #E8B58C; }
    .mk-eyebrow::before { background: #E8B58C; }
    .mk-h1 { font-family: var(--display); font-weight: 600; font-size: 56px; line-height: 1.04; letter-spacing: -0.02em; color: #FBF7EF; margin: 16px 0 18px; }
    .mk-h1 em { font-style: italic; color: #F0C49B; }
    .mk-sub { color: #E7E2D7; font-size: 18px; line-height: 1.6; max-width: 560px; font-weight: 400; }
    .mk-meta { display: flex; gap: 26px; flex-wrap: wrap; margin-top: 26px; }
    .mk-meta span { color: #EDE7DA; font-size: 13.5px; font-weight: 500; display: inline-flex; align-items: center; gap: 8px; }
    .mk-meta i { color: #F0C49B; font-style: normal; font-weight: 800; }

    /* ---- Section scaffolding ---- */
    .mk-sec { margin: 40px 0; }
    .mk-sec-head { max-width: 680px; margin-bottom: 26px; }
    .mk-h2 { font-family: var(--display); font-weight: 600; font-size: 38px; line-height: 1.1; letter-spacing: -0.015em; color: var(--ink); margin: 12px 0 10px; }
    .mk-lead { color: var(--muted); font-size: 17px; line-height: 1.6; }

    /* ---- Logos strip ---- */
    .logos { display: flex; flex-wrap: wrap; gap: 16px 34px; align-items: center; padding: 18px 24px; background: var(--card); border: 1px solid var(--line); border-radius: 16px; }
    .logos span { font-family: var(--display); font-weight: 600; font-size: 19px; color: #9A9282; letter-spacing: .02em; }
    .logos .lab { font-family: var(--sans); font-weight: 600; font-size: 12px; letter-spacing: .16em; text-transform: uppercase; color: var(--soft); margin-right: 8px; }

    /* ---- Stats band ---- */
    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
    .stat { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 26px 22px; box-shadow: 0 8px 24px rgba(23,32,42,0.05); transition: transform .2s ease, box-shadow .2s ease; }
    .stat:hover { transform: translateY(-4px); box-shadow: 0 16px 38px rgba(23,32,42,0.12); }
    .stat-n { font-family: var(--display); font-weight: 700; font-size: 40px; color: var(--ink); line-height: 1; }
    .stat-n em { color: var(--accent); font-style: normal; }
    .stat-l { color: var(--muted); font-size: 14px; margin-top: 8px; line-height: 1.45; }

    /* ---- Feature grid ---- */
    .feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
    .feat-card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: 0 10px 28px rgba(23,32,42,0.06); transition: transform .22s ease, box-shadow .22s ease; display: flex; flex-direction: column; }
    .feat-card:hover { transform: translateY(-6px); box-shadow: 0 22px 48px rgba(23,32,42,0.14); }
    .feat-img { height: 168px; background-size: cover; background-position: center; position: relative; }
    .feat-img::after { content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(18,22,28,0) 40%, rgba(18,22,28,0.28) 100%); }
    .feat-body { padding: 20px 22px 24px; }
    .feat-k { font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent-d); }
    .feat-h { font-family: var(--display); font-weight: 600; font-size: 21px; color: var(--ink); margin: 8px 0 8px; line-height: 1.2; }
    .feat-p { color: var(--muted); font-size: 14.5px; line-height: 1.6; }

    /* ---- Showreel (cross-fade slideshow) ---- */
    .reel { position: relative; height: 420px; border-radius: 22px; overflow: hidden; border: 1px solid rgba(23,32,42,0.10); box-shadow: 0 22px 54px rgba(23,32,42,0.18); }
    .reel .slide { position: absolute; inset: 0; background-size: cover; background-position: center; opacity: 0; animation: reelFade 18s infinite; }
    .reel .slide:nth-child(1){ animation-delay: 0s; }
    .reel .slide:nth-child(2){ animation-delay: 6s; }
    .reel .slide:nth-child(3){ animation-delay: 12s; }
    @keyframes reelFade { 0%{opacity:0;transform:scale(1.05);} 6%{opacity:1;} 30%{opacity:1;transform:scale(1.1);} 36%{opacity:0;} 100%{opacity:0;} }
    .reel-ov { position: absolute; inset: 0; z-index: 2; background: linear-gradient(0deg, rgba(18,22,28,0.66), rgba(18,22,28,0.05) 60%); display: flex; align-items: flex-end; padding: 30px 34px; }
    .reel-ov h3 { font-family: var(--display); font-weight: 600; font-size: 26px; color: #FBF7EF; margin: 0; }
    .reel-ov p { color: #E7E2D7; margin: 6px 0 0; font-size: 15px; }
    .reel-dot { position: absolute; z-index: 3; top: 20px; left: 22px; display: inline-flex; align-items: center; gap: 8px; color: #FBF7EF; font-size: 12px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; }
    .reel-dot::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: #E5484D; box-shadow: 0 0 0 0 rgba(229,72,77,0.6); animation: pulseDot 1.8s infinite; }
    @keyframes pulseDot { 0%{box-shadow:0 0 0 0 rgba(229,72,77,0.5);} 70%{box-shadow:0 0 0 10px rgba(229,72,77,0);} 100%{box-shadow:0 0 0 0 rgba(229,72,77,0);} }

    /* ---- Split (image + text) ---- */
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: center; }
    .split-img { height: 380px; border-radius: 20px; background-size: cover; background-position: center; box-shadow: 0 18px 44px rgba(23,32,42,0.16); border: 1px solid rgba(23,32,42,0.08); }
    .split ul { list-style: none; padding: 0; margin: 16px 0 0; }
    .split li { color: var(--ink-2); font-size: 15.5px; line-height: 1.5; padding: 10px 0 10px 30px; position: relative; border-bottom: 1px solid var(--line-2); }
    .split li::before { content: "→"; position: absolute; left: 0; top: 10px; color: var(--accent); font-weight: 800; }

    /* ---- Testimonials ---- */
    .quote-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
    .quote { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 26px 24px; box-shadow: 0 10px 28px rgba(23,32,42,0.06); display: flex; flex-direction: column; }
    .quote .mark { font-family: var(--display); font-size: 46px; line-height: .4; color: var(--accent); height: 24px; }
    .quote p { color: var(--ink-2); font-size: 15.5px; line-height: 1.6; font-style: italic; font-family: var(--display); font-weight: 400; }
    .quote-by { display: flex; align-items: center; gap: 12px; margin-top: 18px; }
    .quote-av { width: 44px; height: 44px; border-radius: 50%; background-size: cover; background-position: center; flex-shrink: 0; border: 2px solid var(--accent-soft); }
    .quote-by b { color: var(--ink); font-size: 14px; display: block; }
    .quote-by small { color: var(--muted); font-size: 12.5px; }

    /* ---- Pricing ---- */
    .price-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; align-items: stretch; }
    .price-card { background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 28px 26px; display: flex; flex-direction: column; box-shadow: 0 10px 28px rgba(23,32,42,0.06); }
    .price-card.featured { border: 1.6px solid var(--ink); box-shadow: 0 20px 48px rgba(23,32,42,0.16); position: relative; }
    .price-tag { position: absolute; top: 18px; right: 18px; background: var(--accent); color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; padding: 4px 12px; border-radius: 999px; }
    .price-k { font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--accent-d); }
    .price-amt { font-family: var(--display); font-weight: 700; font-size: 44px; color: var(--ink); margin: 10px 0 2px; }
    .price-amt small { font-family: var(--sans); font-size: 15px; font-weight: 500; color: var(--muted); }
    .price-li { list-style: none; padding: 0; margin: 16px 0; flex: 1; }
    .price-li li { color: var(--ink-2); font-size: 14.5px; padding: 8px 0 8px 26px; position: relative; border-bottom: 1px solid var(--line-2); }
    .price-li li::before { content: "✓"; position: absolute; left: 0; color: var(--ok); font-weight: 800; }

    /* ---- CTA band ---- */
    .cta-band { position: relative; border-radius: 24px; overflow: hidden; padding: 50px 48px; margin: 8px 0; background: linear-gradient(120deg, #14181E 0%, #1E2731 60%, #2A3845 100%); box-shadow: 0 24px 56px rgba(23,32,42,0.22); }
    .cta-band::after { content: ""; position: absolute; right: -60px; top: -60px; width: 320px; height: 320px; border-radius: 50%; background: radial-gradient(closest-side, rgba(190,91,42,0.34), transparent 70%); }
    .cta-h { font-family: var(--display); font-weight: 600; font-size: 34px; color: #FBF7EF; line-height: 1.12; margin: 0 0 10px; position: relative; z-index: 1; max-width: 620px; }
    .cta-p { color: #D8D2C6; font-size: 16px; line-height: 1.6; max-width: 560px; position: relative; z-index: 1; }

    /* ---- Contact ---- */
    .contact-card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 24px 26px; box-shadow: 0 10px 28px rgba(23,32,42,0.06); }
    .contact-row { display: flex; align-items: center; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--line-2); }
    .contact-row:last-child { border-bottom: none; }
    .contact-ic { width: 40px; height: 40px; border-radius: 11px; background: var(--accent-soft); display: inline-flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
    .contact-row b { color: var(--ink); font-size: 14px; display: block; }
    .contact-row small { color: var(--muted); font-size: 13px; }

    /* ---- Footer ---- */
    .mk-foot { border-top: 1px solid var(--line); margin-top: 44px; padding: 30px 4px 12px; display: grid; grid-template-columns: 1.6fr 1fr 1fr 1fr; gap: 24px; }
    .mk-foot h4 { font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--soft); margin: 0 0 12px; }
    .mk-foot a, .mk-foot p { color: var(--ink-2); font-size: 14px; line-height: 1.9; display: block; }
    .mk-foot .brandline { font-family: var(--display); font-weight: 700; font-size: 20px; color: var(--ink); margin-bottom: 8px; }
    .mk-foot .muted { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .foot-legal { color: var(--soft); font-size: 12.5px; text-align: center; padding: 18px 0 6px; border-top: 1px solid var(--line-2); margin-top: 24px; }

    @media (max-width: 1000px){
        .feat-grid, .quote-grid, .price-grid, .stats { grid-template-columns: 1fr 1fr; }
        .split { grid-template-columns: 1fr; }
        .mk-h1 { font-size: 40px; }
        .mk-hero-in { padding: 36px 28px; }
        .mk-foot { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 640px){
        .feat-grid, .quote-grid, .price-grid, .stats { grid-template-columns: 1fr; }
        .mk-h1 { font-size: 32px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Dashboard-Hilfsfunktionen (rein optisch — keine Bewertung der Bewerber)
# ---------------------------------------------------------------------------


def quality_bucket(quality: dict | None) -> str:
    """Klassifiziert die DATENQUALITÄT in drei Buckets (kein Eignungsurteil)."""
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
if "reviewed_questions" not in st.session_state:
    # Map filename -> set of question-indices, die als geprüft markiert wurden
    st.session_state.reviewed_questions = {}
if "_flash" not in st.session_state:
    # Transiente Statusmeldungen, die einen st.rerun() überleben
    st.session_state._flash = []
if "_coverage_logged" not in st.session_state:
    st.session_state._coverage_logged = set()


# ---------------------------------------------------------------------------
# Sidebar — Logo, Navigation, Eingaben, Hilfe
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-Page-Router (Session-State) — reine Navigation, keine Geschäftslogik
# ---------------------------------------------------------------------------

NAV_PAGES = [
    ("home", "Start"),
    ("produkt", "Produkt"),
    ("ablauf", "So funktioniert's"),
    ("referenzen", "Referenzen"),
    ("preise", "Preise"),
]
NAV_MORE = [
    ("ueber", "Über uns"),
    ("kontakt", "Kontakt"),
]
_ALL_PAGE_IDS = {p for p, _ in NAV_PAGES + NAV_MORE} | {"workspace"}

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "home"


def goto(page: str) -> None:
    """Wechselt die sichtbare Seite (nur Navigation)."""
    if page in _ALL_PAGE_IDS:
        st.session_state.nav_page = page
        st.rerun()


with st.sidebar:
    st.markdown(
        '<div class="kmu-logo">Recruiting&nbsp;AI</div>',
        unsafe_allow_html=True,
    )
    st.caption("Der KI-Workspace für Geschäftsführer ohne eigenes HR-Team.")
    st.divider()
    st.markdown("**Navigation**")
    for _pid, _lbl in NAV_PAGES + NAV_MORE:
        _active = st.session_state.nav_page == _pid
        if st.button(
            ("●  " if _active else "") + _lbl,
            key=f"sb_{_pid}",
            use_container_width=True,
            type="primary" if _active else "secondary",
        ):
            goto(_pid)
    st.divider()
    if st.button(
        "Workspace öffnen  →",
        key="sb_ws",
        type="primary",
        use_container_width=True,
    ):
        goto("workspace")
    st.caption("Der Mensch entscheidet — der Agent strukturiert.")


# ---------------------------------------------------------------------------
# Top-Navigation, Marketing-Seiten & Seiten-Routing
# (rein optisch — die bestehende App-Logik bleibt unverändert)
# ---------------------------------------------------------------------------


def _img(pid: str, w: int = 1200) -> str:
    return f"https://images.unsplash.com/{pid}?auto=format&fit=crop&w={w}&q=80"


IMG_HERO = _img("photo-1521737604893-d14cc237f11d", 1700)
IMG_REEL1 = _img("photo-1600880292203-757bb62b4baf", 1400)
IMG_REEL2 = _img("photo-1497366216548-37526070297c", 1400)
IMG_REEL3 = _img("photo-1522071820081-009f0129c71c", 1400)
IMG_SPLIT1 = _img("photo-1573164713988-8665fc963095", 1200)
IMG_SPLIT2 = _img("photo-1551836022-d5d88e9218df", 1200)
IMG_ABOUT = _img("photo-1542744173-8e7e53415bb0", 1300)

FEATURES = [
    {
        "img": _img("photo-1450101499163-c8848c66ca85"),
        "k": "Schritt 01",
        "h": "Stellenprofil verstehen",
        "p": "Fügen Sie die Stellenanzeige ein. Der Agent extrahiert ausschließlich "
        "objektiv prüfbare Anforderungen — Skills, Sprachen, Zertifikate, Erfahrung.",
    },
    {
        "img": _img("photo-1486312338219-ce68d2c6f44d"),
        "k": "Schritt 02",
        "h": "Lebensläufe strukturieren",
        "p": "Mehrere PDF-Bewerbungen werden gleichzeitig eingelesen und in saubere, "
        "vergleichbare Felder überführt — inklusive Belegstellen im Original.",
    },
    {
        "img": _img("photo-1460925895917-afdab827c52f"),
        "k": "Schritt 03",
        "h": "Anforderungen abgleichen",
        "p": "Jede Anforderung wird transparent als gefunden, teilweise gefunden oder "
        "nicht gefunden markiert — mit nachvollziehbarer Begründung. Kein Score.",
    },
    {
        "img": _img("photo-1454165804606-c3d57bc86b40"),
        "k": "Schritt 04",
        "h": "Informationslücken erkennen",
        "p": "Fehlende oder unklare Angaben werden sichtbar gemacht, statt sie zu "
        "raten — die Grundlage für faire, fundierte Gespräche.",
    },
    {
        "img": _img("photo-1521791136064-7986c2920216"),
        "k": "Schritt 05",
        "h": "Rückfragen vorbereiten",
        "p": "Höfliche, präzise Rückfragen werden vorformuliert. Sie prüfen und geben "
        "frei — es wird nichts automatisch versendet.",
    },
    {
        "img": _img("photo-1600880292089-90a7e086ee0c"),
        "k": "Schritt 06",
        "h": "Der Mensch entscheidet",
        "p": "Recruiting AI bewertet niemanden und erstellt kein Ranking. Die Auswahl "
        "treffen Sie — vollständig auditierbar und DSGVO-konform gedacht.",
    },
]

TESTIMONIALS = [
    {
        "av": _img("photo-1500648767791-00dcc994a43e", 200),
        "t": "Wir bekommen täglich 40 Bewerbungen. Recruiting AI gibt uns in Minuten "
        "eine saubere, vergleichbare Übersicht — die Entscheidung treffen wir trotzdem selbst.",
        "by": "Markus Reinhardt",
        "role": "Geschäftsführer, Reinhardt Bau GmbH",
    },
    {
        "av": _img("photo-1494790108377-be9c29b29330", 200),
        "t": "Endlich kein stundenlanges Lebenslauflesen mehr. Besonders die "
        "Informationslücken-Erkennung spart uns peinliche Nachfragen im Gespräch.",
        "by": "Sandra Kühn",
        "role": "Inhaberin, Kühn Digital Studio",
    },
    {
        "av": _img("photo-1472099645785-5658abf4ff4e", 200),
        "t": "Transparent und nachvollziehbar. Jede Aussage hat eine Belegstelle — "
        "das schafft Vertrauen bei uns und im Team.",
        "by": "Daniel Vogt",
        "role": "CEO, Vogt Logistik",
    },
    {
        "av": _img("photo-1519085360753-af0119f7cbe7", 200),
        "t": "Wir sind ein 12-Personen-Team ohne HR-Abteilung. Der Workspace fühlt "
        "sich an wie eine zusätzliche, sehr gründliche Kollegin.",
        "by": "Lena Brandt",
        "role": "Gründerin, Brandt Manufaktur",
    },
    {
        "av": _img("photo-1507003211169-0a1dd7228f2d", 200),
        "t": "Die Audit-Funktion ist Gold wert. Wir können jeden Schritt der Analyse "
        "lückenlos nachvollziehen — wichtig für Fairness und Compliance.",
        "by": "Thomas Eder",
        "role": "Geschäftsführer, Eder & Partner",
    },
    {
        "av": _img("photo-1438761681033-6461ffad8d80", 200),
        "t": "Keine Black Box, keine automatische Ablehnung. Genau das wollten wir: "
        "ein Werkzeug, das uns zuarbeitet, statt für uns zu urteilen.",
        "by": "Julia Hofmann",
        "role": "Inhaberin, Hofmann Praxisklinik",
    },
]


def render_top_nav() -> None:
    active = st.session_state.nav_page
    cols = st.columns([2.5, 0.9, 1.0, 1.55, 1.35, 1.15, 1.05, 1.65])
    with cols[0]:
        st.markdown(
            '<div class="nav-brand"><span class="nav-mark">R</span>'
            '<span class="nav-name">Recruiting&nbsp;AI'
            "<small>Recruiting Intelligence</small></span></div>",
            unsafe_allow_html=True,
        )
    for idx, (pid, lbl) in enumerate(NAV_PAGES):
        with cols[1 + idx]:
            if st.button(
                ("●  " if active == pid else "") + lbl,
                key=f"nav_{pid}",
                use_container_width=True,
            ):
                goto(pid)
    with cols[6]:
        _menu = st.popover("Mehr  ▾") if hasattr(st, "popover") else st.expander("Mehr  ▾")
        with _menu:
            st.caption("Weitere Seiten")
            for pid, lbl in NAV_MORE:
                if st.button(lbl, key=f"more_{pid}", use_container_width=True):
                    goto(pid)
            st.divider()
            if st.button("→ Workspace", key="more_ws", use_container_width=True):
                goto("workspace")
    with cols[7]:
        if st.button(
            "Workspace öffnen  →",
            key="nav_ws",
            type="primary",
            use_container_width=True,
        ):
            goto("workspace")
    st.markdown('<hr class="nav-rule">', unsafe_allow_html=True)


# ---- Wiederverwendbare Bausteine -------------------------------------------


def _hero(img, eyebrow, title_html, sub, meta_html):
    st.markdown(
        f"""
        <div class="mk-hero">
          <div class="mk-hero-bg" style="background-image:url('{img}')"></div>
          <div class="mk-hero-ov"></div>
          <div class="mk-hero-in">
            <span class="eyebrow mk-eyebrow">{eyebrow}</span>
            <h1 class="mk-h1">{title_html}</h1>
            <div class="mk-sub">{sub}</div>
            <div class="mk-meta">{meta_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_header(eyebrow, title, lead):
    st.markdown(
        f'<div style="margin:14px 0 30px">'
        f'<span class="eyebrow">{eyebrow}</span>'
        f'<div class="mk-h2" style="font-size:46px;margin-top:14px">{title}</div>'
        f'<div class="mk-lead" style="max-width:720px;margin-top:8px">{lead}</div></div>',
        unsafe_allow_html=True,
    )


def _sec_head(eyebrow, title, lead):
    st.markdown(
        f'<div class="mk-sec-head"><span class="eyebrow">{eyebrow}</span>'
        f'<div class="mk-h2">{title}</div>'
        f'<div class="mk-lead">{lead}</div></div>',
        unsafe_allow_html=True,
    )


def _logos():
    st.markdown(
        '<div class="logos"><span class="lab">Vertraut von Unternehmen ohne HR-Team</span>'
        "<span>Reinhardt&nbsp;Bau</span><span>Kühn&nbsp;Studio</span>"
        "<span>Vogt&nbsp;Logistik</span><span>Brandt&nbsp;Manufaktur</span>"
        "<span>Eder&nbsp;&amp;&nbsp;Partner</span><span>Hofmann&nbsp;Klinik</span></div>",
        unsafe_allow_html=True,
    )


def _stats():
    st.markdown(
        """
        <div class="stats">
          <div class="stat"><div class="stat-n">4<em>×</em></div>
            <div class="stat-l">spezialisierte Agenten in einem Workflow</div></div>
          <div class="stat"><div class="stat-n">~8&nbsp;<em>Min</em></div>
            <div class="stat-l">statt Stunden pro Bewerbungsstapel</div></div>
          <div class="stat"><div class="stat-n">100<em>%</em></div>
            <div class="stat-l">der Schritte auditierbar dokumentiert</div></div>
          <div class="stat"><div class="stat-n">0</div>
            <div class="stat-l">automatische Personalentscheidungen</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_grid(items):
    cards = ""
    for it in items:
        cards += (
            '<div class="feat-card">'
            f"<div class=\"feat-img\" style=\"background-image:url('{it['img']}')\"></div>"
            f'<div class="feat-body"><div class="feat-k">{it["k"]}</div>'
            f'<div class="feat-h">{it["h"]}</div>'
            f'<div class="feat-p">{it["p"]}</div></div></div>'
        )
    st.markdown(f'<div class="feat-grid">{cards}</div>', unsafe_allow_html=True)


def _testimonials(items):
    cards = ""
    for it in items:
        cards += (
            '<div class="quote"><div class="mark">&ldquo;</div>'
            f'<p>{it["t"]}</p><div class="quote-by">'
            f"<div class=\"quote-av\" style=\"background-image:url('{it['av']}')\"></div>"
            f'<div><b>{it["by"]}</b><small>{it["role"]}</small></div></div></div>'
        )
    st.markdown(f'<div class="quote-grid">{cards}</div>', unsafe_allow_html=True)


def _split(img, eyebrow, title, lead, items, img_right=True):
    li = "".join(f"<li>{x}</li>" for x in items)
    img_html = f"<div class=\"split-img\" style=\"background-image:url('{img}')\"></div>"
    txt_html = (
        f'<div><span class="eyebrow">{eyebrow}</span>'
        f'<div class="mk-h2">{title}</div><div class="mk-lead">{lead}</div>'
        f"<ul>{li}</ul></div>"
    )
    inner = (txt_html + img_html) if img_right else (img_html + txt_html)
    st.markdown(
        f'<div class="mk-sec"><div class="split">{inner}</div></div>',
        unsafe_allow_html=True,
    )


def _reel():
    st.markdown(
        f"""
        <div class="reel">
          <span class="reel-dot">Im Einsatz</span>
          <div class="slide" style="background-image:url('{IMG_REEL1}')"></div>
          <div class="slide" style="background-image:url('{IMG_REEL2}')"></div>
          <div class="slide" style="background-image:url('{IMG_REEL3}')"></div>
          <div class="reel-ov"><div>
            <h3>Vom PDF zur Entscheidungsgrundlage</h3>
            <p>Vier Agenten arbeiten zusammen — Sie behalten die Kontrolle.</p>
          </div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cta_band(title, text, primary_label, primary_target, key):
    st.markdown(
        f'<div class="cta-band"><div class="cta-h">{title}</div>'
        f'<div class="cta-p">{text}</div></div>',
        unsafe_allow_html=True,
    )
    cc = st.columns([1.5, 1.5, 3])
    with cc[0]:
        if st.button(
            primary_label, key=key, type="primary", use_container_width=True
        ):
            goto(primary_target)
    with cc[1]:
        if st.button(
            "Kontakt aufnehmen", key=key + "_k", use_container_width=True
        ):
            goto("kontakt")


def _footer():
    st.markdown(
        """
        <div class="mk-foot">
          <div>
            <div class="brandline">Recruiting&nbsp;AI</div>
            <p class="muted">Der KI-Workspace für Geschäftsführer ohne eigenes
            HR-Team. Wir strukturieren Bewerbungen — die Entscheidung bleibt
            immer bei Ihnen.</p>
          </div>
          <div><h4>Produkt</h4><a>Funktionen</a><a>So funktioniert's</a>
            <a>Preise</a><a>Workspace</a></div>
          <div><h4>Unternehmen</h4><a>Über uns</a><a>Referenzen</a>
            <a>Kontakt</a><a>Karriere</a></div>
          <div><h4>Rechtliches</h4><a>Datenschutz</a><a>Impressum</a>
            <a>AGB</a><a>DSGVO</a></div>
        </div>
        <div class="foot-legal">© 2026 Recruiting&nbsp;AI · Human-in-the-Loop ·
        Keine automatische Personalentscheidung · Made in Germany</div>
        """,
        unsafe_allow_html=True,
    )


# ---- Seiten ----------------------------------------------------------------


def render_home():
    _hero(
        IMG_HERO,
        "KI-Recruiting-Workspace",
        "Bewerbungen verstehen.<br><em>Menschen</em> entscheiden.",
        "Recruiting AI strukturiert eingehende Lebensläufe, prüft fachliche "
        "Anforderungen und erkennt Informationslücken — transparent, "
        "nachvollziehbar und ohne automatische Personalentscheidung.",
        "<span><i>✓</i> Multi-Agent-Analyse</span>"
        "<span><i>✓</i> Human-in-the-Loop</span>"
        "<span><i>✓</i> Vollständig auditierbar</span>",
    )
    hc = st.columns([1.5, 1.6, 3])
    with hc[0]:
        if st.button(
            "Kostenlos testen  →", key="home_cta1", type="primary",
            use_container_width=True,
        ):
            goto("workspace")
    with hc[1]:
        if st.button(
            "So funktioniert's", key="home_cta2", use_container_width=True
        ):
            goto("ablauf")

    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    _logos()

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _stats()

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Was der Workspace leistet",
        "Vier Agenten. Ein Ziel: Klarheit ohne Urteil.",
        "Jeder Schritt ist nachvollziehbar belegt. Bewertet wird nicht — "
        "strukturiert schon.",
    )
    _feature_grid(FEATURES[:3])

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Der Ablauf",
        "Vom Stellenprofil zur Entscheidungsgrundlage.",
        "Sieben transparente Schritte — der letzte gehört immer dem Menschen.",
    )
    st.markdown(_WORKFLOW_HTML, unsafe_allow_html=True)

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Stimmen aus der Praxis",
        "Gemacht für Unternehmen ohne HR-Abteilung.",
        "Geschäftsführerinnen und Geschäftsführer, die ihre Zeit zurückbekommen.",
    )
    _testimonials(TESTIMONIALS[:3])

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _cta_band(
        "Bereit, den Bewerbungsstapel in Klarheit zu verwandeln?",
        "Starten Sie in unter einer Minute — kein Setup, keine Kreditkarte. "
        "Sie behalten jederzeit die volle Entscheidungshoheit.",
        "Workspace kostenlos öffnen  →",
        "workspace",
        "home_cta_band",
    )


def render_produkt():
    _page_header(
        "Produkt",
        "Ein Workspace, der zuarbeitet — nicht urteilt.",
        "Recruiting AI verbindet vier spezialisierte Agenten zu einem "
        "durchgängigen, auditierbaren Analyse-Workflow. Sie behalten die "
        "Kontrolle über jede Entscheidung.",
    )
    _feature_grid(FEATURES)

    _split(
        IMG_SPLIT1,
        "Transparenz by Design",
        "Jede Aussage hat eine Belegstelle.",
        "Keine Black Box: Recruiting AI zeigt für jede strukturierte Information "
        "die Fundstelle im Original-Lebenslauf. So bleibt jede Analyse prüfbar.",
        [
            "Anforderungen klar als gefunden / teilweise / nicht gefunden markiert",
            "Belegstellen direkt aus dem Lebenslauf zitiert",
            "Nur objektiv prüfbare Kriterien — keine Soft-Skill-Bewertung",
            "Lückenloses Audit-Log über jeden Agentenschritt",
        ],
        img_right=True,
    )
    _split(
        IMG_SPLIT2,
        "Mensch im Mittelpunkt",
        "Human-in-the-Loop, kompromisslos.",
        "Der Agent liefert Daten — die Entscheidung treffen Sie. Es gibt kein "
        "Ranking, keinen Score und keine automatische Ablehnung.",
        [
            "Keine automatische Personalentscheidung",
            "Rückfragen werden vorbereitet, aber nie automatisch versendet",
            "Volle Kontrolle über jeden Verarbeitungsschritt",
            "DSGVO-konform gedacht, auf Fairness ausgelegt",
        ],
        img_right=False,
    )
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _cta_band(
        "Sehen Sie Recruiting AI an Ihren eigenen Bewerbungen.",
        "Laden Sie ein Stellenprofil und ein paar Lebensläufe hoch — in Minuten "
        "haben Sie eine strukturierte, belegte Übersicht.",
        "Jetzt ausprobieren  →",
        "workspace",
        "prod_cta_band",
    )


def render_ablauf():
    _page_header(
        "So funktioniert's",
        "Vier Agenten, sieben Schritte, ein Mensch am Ende.",
        "Recruiting AI nimmt Ihnen die Fleißarbeit ab und macht jeden Schritt "
        "sichtbar — damit Sie schneller und sicherer entscheiden.",
    )
    st.markdown(_WORKFLOW_HTML, unsafe_allow_html=True)

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Live im Einsatz",
        "Sehen, wie aus Unterlagen Klarheit wird.",
        "Ein dynamischer Blick in den Workspace — vom Upload bis zur Entscheidung.",
    )
    _reel()

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Vernetzte Analyse",
        "Jeder Datenpunkt ist verbunden — und nachvollziehbar.",
        "Die Agenten teilen Kontext, ohne Entscheidungen zu treffen.",
    )
    components.html(_GLOBE_HTML, height=340)

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _cta_band(
        "Probieren Sie den kompletten Ablauf selbst aus.",
        "Vom Stellenprofil bis zu vorbereiteten Rückfragen — in einem Durchgang.",
        "Workspace öffnen  →",
        "workspace",
        "ablauf_cta_band",
    )


def render_referenzen():
    _page_header(
        "Referenzen",
        "Unternehmen, die ihre Zeit zurückbekommen haben.",
        "Vom Handwerksbetrieb bis zur Praxisklinik: Recruiting AI hilft kleinen "
        "und mittleren Teams, Bewerbungen souverän zu strukturieren.",
    )
    _stats()
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _testimonials(TESTIMONIALS)
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _logos()
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _cta_band(
        "Werden Sie das nächste Team, das Stunden spart.",
        "Testen Sie Recruiting AI mit Ihren eigenen Unterlagen — unverbindlich.",
        "Kostenlos starten  →",
        "workspace",
        "ref_cta_band",
    )


def render_preise():
    _page_header(
        "Preise",
        "Faire Preise für Teams ohne HR-Abteilung.",
        "Transparent, monatlich kündbar, ohne versteckte Kosten. Starten Sie "
        "kostenlos und wachsen Sie, wenn Sie mehr brauchen.",
    )
    st.markdown(
        """
        <div class="price-grid">
          <div class="price-card">
            <div class="price-k">Starter</div>
            <div class="price-amt">0&nbsp;€<small>/ Monat</small></div>
            <div class="mk-lead" style="font-size:14px">Zum Ausprobieren.</div>
            <ul class="price-li">
              <li>Bis zu 10 Bewerbungen / Monat</li>
              <li>1 Stellenprofil aktiv</li>
              <li>Anforderungsabgleich & Belege</li>
              <li>Audit-Log</li>
            </ul>
          </div>
          <div class="price-card featured">
            <span class="price-tag">Beliebt</span>
            <div class="price-k">Business</div>
            <div class="price-amt">49&nbsp;€<small>/ Monat</small></div>
            <div class="mk-lead" style="font-size:14px">Für aktive Recruiter.</div>
            <ul class="price-li">
              <li>Unbegrenzte Bewerbungen</li>
              <li>Mehrere Stellenprofile parallel</li>
              <li>Informationslücken & Rückfragen</li>
              <li>Feedback-Lernschleife</li>
              <li>Priorisierter Support</li>
            </ul>
          </div>
          <div class="price-card">
            <div class="price-k">Enterprise</div>
            <div class="price-amt">Individuell</div>
            <div class="mk-lead" style="font-size:14px">Für größere Teams.</div>
            <ul class="price-li">
              <li>Alles aus Business</li>
              <li>SSO & Rollen</li>
              <li>Eigene Datenhaltung</li>
              <li>Dedizierter Ansprechpartner</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pc = st.columns(3)
    with pc[0]:
        if st.button("Starter wählen", key="price_starter", use_container_width=True):
            goto("workspace")
    with pc[1]:
        if st.button(
            "Business starten  →", key="price_business", type="primary",
            use_container_width=True,
        ):
            goto("workspace")
    with pc[2]:
        if st.button("Vertrieb kontaktieren", key="price_ent", use_container_width=True):
            goto("kontakt")

    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _sec_head(
        "Häufige Fragen",
        "Alles, was Geschäftsführer wissen wollen.",
        "Noch eine Frage offen? Schreiben Sie uns über die Kontaktseite.",
    )
    with st.expander("Trifft Recruiting AI eine Auswahl oder lehnt Bewerber ab?"):
        st.write(
            "Nein. Recruiting AI bewertet niemanden, erstellt kein Ranking und "
            "lehnt niemanden ab. Es strukturiert Informationen — die Entscheidung "
            "treffen ausschließlich Sie."
        )
    with st.expander("Werden automatisch E-Mails an Bewerber versendet?"):
        st.write(
            "Nein. Rückfragen werden lediglich vorbereitet. Versendet wird nichts "
            "ohne Ihre ausdrückliche Freigabe."
        )
    with st.expander("Wie nachvollziehbar ist die Analyse?"):
        st.write(
            "Jeder Agentenschritt wird im Audit-Log protokolliert, und jede "
            "strukturierte Aussage verweist auf eine Belegstelle im Lebenslauf."
        )
    with st.expander("Kann ich monatlich kündigen?"):
        st.write("Ja. Alle Pläne sind monatlich kündbar, ohne Mindestlaufzeit.")


def render_ueber():
    _page_header(
        "Über uns",
        "Wir bauen Werkzeuge, die zuarbeiten — nicht entscheiden.",
        "Recruiting AI entstand aus einer einfachen Überzeugung: Software sollte "
        "Geschäftsführern Zeit und Klarheit geben, ohne ihnen die Verantwortung "
        "abzunehmen.",
    )
    _split(
        IMG_ABOUT,
        "Unsere Haltung",
        "Technologie mit Augenmaß.",
        "Wir glauben an menschliche Entscheidungen, unterstützt durch ehrliche, "
        "transparente KI. Kein Hype, keine Black Box — nur ein Werkzeug, das "
        "seinen Job sauberer macht.",
        [
            "Human-in-the-Loop als Grundprinzip",
            "Transparenz und Belegbarkeit vor Bequemlichkeit",
            "Fairness und DSGVO von Anfang an mitgedacht",
            "Gebaut für KMU, nicht für Konzerne",
        ],
        img_right=False,
    )
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _stats()
    st.markdown('<div class="mk-sec"></div>', unsafe_allow_html=True)
    _cta_band(
        "Lernen Sie den Workspace kennen.",
        "Am besten verstehen Sie Recruiting AI, indem Sie es ausprobieren.",
        "Workspace öffnen  →",
        "workspace",
        "ueber_cta_band",
    )


def render_kontakt():
    _page_header(
        "Kontakt",
        "Sprechen wir über Ihr Recruiting.",
        "Ob Demo, Frage oder Enterprise-Anfrage — wir melden uns in der Regel "
        "innerhalb eines Werktags.",
    )
    kc = st.columns([1, 1.2], gap="large")
    with kc[0]:
        st.markdown(
            """
            <div class="contact-card">
              <div class="contact-row"><span class="contact-ic">✉️</span>
                <div><b>E-Mail</b><small>hallo@recruiting-ai.de</small></div></div>
              <div class="contact-row"><span class="contact-ic">📞</span>
                <div><b>Telefon</b><small>+49&nbsp;30&nbsp;1234&nbsp;5678</small></div></div>
              <div class="contact-row"><span class="contact-ic">📍</span>
                <div><b>Standort</b><small>Berlin, Deutschland</small></div></div>
              <div class="contact-row"><span class="contact-ic">🕑</span>
                <div><b>Erreichbarkeit</b><small>Mo–Fr, 9–18&nbsp;Uhr</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kc[1]:
        with st.container(border=True):
            st.markdown(
                '<div class="kmu-card-title">Nachricht senden</div>',
                unsafe_allow_html=True,
            )
            with st.form("contact_form", clear_on_submit=True):
                cf1, cf2 = st.columns(2)
                with cf1:
                    _name = st.text_input("Name")
                with cf2:
                    _company = st.text_input("Unternehmen")
                _email = st.text_input("E-Mail")
                _msg = st.text_area("Ihre Nachricht", height=120)
                _sent = st.form_submit_button("Nachricht senden", type="primary")
                if _sent:
                    if not (_name.strip() and _email.strip() and _msg.strip()):
                        st.warning("Bitte Name, E-Mail und Nachricht ausfüllen.")
                    else:
                        st.success(
                            "Danke! Ihre Nachricht ist eingegangen — "
                            "wir melden uns zeitnah."
                        )
            st.caption(
                "Diese Demo versendet keine echten Nachrichten und speichert "
                "keine Eingaben."
            )


# ---- Workflow- & Globus-Visual (für die Marketing-Seiten) ------------------

_WORKFLOW_HTML = """
<style>
  .wf { margin:6px 0 8px; font-family:'Familjen Grotesk',-apple-system,sans-serif; }
  .wf-row { display:grid; grid-template-columns:repeat(7, 1fr); gap:10px;
    align-items:stretch; }
  .wf-step { position:relative; background:#FFFDF8;
    border:1px solid #E5DECF; border-radius:16px;
    padding:18px 12px 16px; text-align:center;
    box-shadow:0 8px 24px rgba(23,32,42,0.06);
    transition:transform .2s ease, box-shadow .2s ease;
    animation:fadeUpG .6s ease both; }
  .wf-step:nth-child(2){animation-delay:.05s} .wf-step:nth-child(3){animation-delay:.1s}
  .wf-step:nth-child(4){animation-delay:.15s} .wf-step:nth-child(5){animation-delay:.2s}
  .wf-step:nth-child(6){animation-delay:.25s} .wf-step:nth-child(7){animation-delay:.3s}
  .wf-step:hover { transform:translateY(-4px);
    box-shadow:0 16px 36px rgba(23,32,42,0.14); }
  .wf-step::after { content:"→"; position:absolute; right:-13px; top:50%;
    transform:translateY(-50%); color:#BE5B2A; font-weight:800; font-size:16px;
    z-index:2; animation:pulseArrow 2.4s ease-in-out infinite; }
  .wf-step:last-child::after { content:""; }
  @keyframes pulseArrow { 0%,100%{opacity:.4} 50%{opacity:1} }
  .wf-ic { width:40px; height:40px; border-radius:12px; margin:0 auto 10px;
    display:flex; align-items:center; justify-content:center; font-size:19px;
    background:#F4E7DA; }
  .wf-step.gz .wf-ic { background:#DCEFE4; }
  .wf-name { font-size:13px; font-weight:700; color:#17202A; margin-bottom:4px; }
  .wf-desc { font-size:11.5px; color:#6E6A5F; line-height:1.45; }
  @media (max-width:1100px){ .wf-row { grid-template-columns:repeat(4,1fr); }
    .wf-step:nth-child(4)::after{content:""} }
  @media (max-width:700px){ .wf-row { grid-template-columns:repeat(2,1fr); }
    .wf-step::after{content:""} }
</style>
<div class="wf">
  <div class="wf-row">
    <div class="wf-step"><div class="wf-ic">📤</div>
      <div class="wf-name">Stelle hochladen</div>
      <div class="wf-desc">Stellenanzeige einfügen.</div></div>
    <div class="wf-step"><div class="wf-ic">📋</div>
      <div class="wf-name">Stellenprofil-Agent</div>
      <div class="wf-desc">Extrahiert prüfbare Anforderungen.</div></div>
    <div class="wf-step"><div class="wf-ic">📄</div>
      <div class="wf-name">CV-Agent</div>
      <div class="wf-desc">Strukturiert jeden Lebenslauf.</div></div>
    <div class="wf-step"><div class="wf-ic">🔍</div>
      <div class="wf-name">Anforderungsabgleich</div>
      <div class="wf-desc">Gefunden, teilweise, nicht gefunden.</div></div>
    <div class="wf-step"><div class="wf-ic">🧩</div>
      <div class="wf-name">Informationslücken</div>
      <div class="wf-desc">Erkennt fehlende Angaben.</div></div>
    <div class="wf-step"><div class="wf-ic">💬</div>
      <div class="wf-name">Rückfragen</div>
      <div class="wf-desc">Formuliert höfliche Rückfragen.</div></div>
    <div class="wf-step gz"><div class="wf-ic">👤</div>
      <div class="wf-name">Geschäftsführer entscheidet</div>
      <div class="wf-desc">Der Mensch entscheidet.</div></div>
  </div>
</div>
"""

_GLOBE_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { width:100%; height:100%; overflow:hidden;
    font-family:'Familjen Grotesk',-apple-system,Segoe UI,Roboto,sans-serif; }
  .band { position:relative; width:100%; height:100%; border-radius:22px; overflow:hidden;
    background:
      radial-gradient(720px 420px at 80% 50%, rgba(190,91,42,0.20), transparent 60%),
      linear-gradient(135deg, #15181C 0%, #1E242B 55%, #2A333D 100%);
    border:1px solid rgba(255,255,255,0.10);
    box-shadow:0 16px 44px rgba(23,32,42,0.20); }
  #c { position:absolute; inset:0; z-index:0; display:block; }
  .copy { position:absolute; z-index:2; left:48px; top:50%;
    transform:translateY(-50%); max-width:460px; }
  .copy h2 { font-family:'Fraunces',Georgia,serif; font-size:30px; font-weight:600;
    line-height:1.15; letter-spacing:-0.01em; color:#FBF7EF; margin-bottom:12px; }
  .copy p { color:#D8D2C6; font-size:15px; line-height:1.6; margin-bottom:4px; }
  @media (max-width:760px){ .copy{left:24px;max-width:60%} .copy h2{font-size:22px} }
</style></head>
<body>
  <div class="band">
    <canvas id="c"></canvas>
    <div class="copy">
      <h2>Transparente Analyse &ndash; in Sekunden.</h2>
      <p>Recruiting AI strukturiert Bewerbungsunterlagen automatisch:
        nachvollziehbar, fair und ohne automatische Personalentscheidung.</p>
    </div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
  (function(){
    if(!window.THREE){return;}
    var canvas=document.getElementById('c'); var band=canvas.parentElement;
    var renderer=new THREE.WebGLRenderer({canvas:canvas,alpha:true,antialias:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    var scene=new THREE.Scene();
    var camera=new THREE.PerspectiveCamera(45,1,0.1,100); camera.position.set(0,0,6.2);
    var group=new THREE.Group(); group.position.x=1.7; group.rotation.z=0.32; group.rotation.x=0.16;
    scene.add(group);
    var R=1.65;
    var wire=new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.SphereGeometry(R,30,20)),
      new THREE.LineBasicMaterial({color:0x9AA3AD,transparent:true,opacity:0.30}));
    group.add(wire);
    var N=440, pos=[], ga=Math.PI*(3-Math.sqrt(5)), i;
    for(i=0;i<N;i++){var y=1-(i/(N-1))*2; var rr=Math.sqrt(1-y*y); var th=ga*i;
      pos.push(Math.cos(th)*rr*R, y*R, Math.sin(th)*rr*R);}
    var dg=new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
    var dots=new THREE.Points(dg, new THREE.PointsMaterial(
      {color:0xE7E2D7,size:0.045,transparent:true,opacity:0.9}));
    group.add(dots);
    function sp(){var u=Math.random(),v=Math.random();var th=2*Math.PI*u;var ph=Math.acos(2*v-1);
      return new THREE.Vector3(R*Math.sin(ph)*Math.cos(th),R*Math.cos(ph),R*Math.sin(ph)*Math.sin(th));}
    for(var k=0;k<16;k++){var a=sp(),b=sp();
      var mid=a.clone().add(b).multiplyScalar(0.5).setLength(R*1.42);
      var curve=new THREE.QuadraticBezierCurve3(a,mid,b);
      var g=new THREE.BufferGeometry().setFromPoints(curve.getPoints(42));
      group.add(new THREE.Line(g, new THREE.LineBasicMaterial(
        {color:0xBE5B2A,transparent:true,opacity:0.55})));}
    function ring(r0,r1,op){var rg=new THREE.RingGeometry(r0,r1,90);
      var m=new THREE.MeshBasicMaterial({color:0xC9A37A,side:THREE.DoubleSide,
        transparent:true,opacity:op});
      var mesh=new THREE.Mesh(rg,m); mesh.rotation.x=Math.PI/2; return mesh;}
    var rings=new THREE.Group();
    rings.add(ring(2.25,2.38,0.50)); rings.add(ring(2.55,2.62,0.30));
    rings.add(ring(2.80,2.84,0.18));
    rings.rotation.x=0.52; group.add(rings);
    function size(){var w=band.clientWidth,h=band.clientHeight;
      renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();}
    size(); window.addEventListener('resize',size);
    function loop(){requestAnimationFrame(loop);
      group.rotation.y+=0.0016; dots.rotation.y-=0.0006;
      renderer.render(scene,camera);}
    loop();
  })();
  </script>
</body></html>
"""


# ---- Navigation rendern + Seiten-Routing -----------------------------------

render_top_nav()

_page = st.session_state.nav_page
if _page == "home":
    render_home()
elif _page == "produkt":
    render_produkt()
elif _page == "ablauf":
    render_ablauf()
elif _page == "referenzen":
    render_referenzen()
elif _page == "preise":
    render_preise()
elif _page == "ueber":
    render_ueber()
elif _page == "kontakt":
    render_kontakt()

if _page != "workspace":
    _footer()
    st.stop()

# --- Ab hier: Workspace (bestehende App-Logik, unverändert) ---

# ---- KPI-Karten (echte Daten aus der App) ----

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
# Gefundene Anforderungen (Summe über alle Kandidaten, bestehende Funktion)
total_found = 0
if st.session_state.job_profile:
    for c in st.session_state.candidates:
        total_found += status_counts(
            evaluate_candidate_requirements(
                st.session_state.job_profile, c["data"]
            )
        )["Gefunden"]

st.markdown(
    '<div class="dashboard-header"><div><h1>Dashboard</h1>'
    '<div class="subtitle">Übersicht Ihrer Bewerbungen und Analysen</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card teal">
            <div class="kpi-icon">📄</div>
            <div class="kpi-value teal">{total_candidates}</div>
            <div class="kpi-label">Analysierte Bewerbungen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card blue">
            <div class="kpi-icon">✅</div>
            <div class="kpi-value blue">{total_found}</div>
            <div class="kpi-label">Gefundene Anforderungen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card orange">
            <div class="kpi-icon">🔍</div>
            <div class="kpi-value orange">{total_gaps}</div>
            <div class="kpi-label">Klärungsbedarf</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card green">
            <div class="kpi-icon">💬</div>
            <div class="kpi-value green">{total_questions}</div>
            <div class="kpi-label">Vorbereitete Rückfragen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Dashboard-Grid: Letzte Bewerbungen | AI Activity Feed ----
# Der Activity Feed ist eine rein optische Zusammenfassung der letzten
# Audit-Log-Einträge — das Audit Log selbst bleibt unverändert.

_ACTIVITY_ICONS = [
    ("Stellenprofil analysiert", "📋"),
    ("fachlich prüfbare", "✅"),
    ("nicht automatisch prüfbare", "🧩"),
    ("PDF hochgeladen", "📤"),
    ("Text extrahiert", "📄"),
    ("CV analysiert", "🤖"),
    ("Informationslücken", "🔍"),
    ("Rückfragen erzeugt", "💬"),
    ("Rückfrage", "💬"),
    ("Anforderungen abgeglichen", "⚖️"),
    ("Klärungspunkte", "🧩"),
    ("Abdeckungsgrad", "⚖️"),
    ("Feedback", "📝"),
]


def _activity_icon(action: str) -> str:
    for key, icon in _ACTIVITY_ICONS:
        if key.lower() in (action or "").lower():
            return icon
    return "•"


dash_left, dash_right = st.columns([1.25, 1], gap="medium")

with dash_left:
    if st.session_state.candidates:
        with st.container(border=True):
            st.markdown(
                '<div class="kmu-card-title">Letzte Bewerbungen</div>',
                unsafe_allow_html=True,
            )
            for c in st.session_state.candidates[-5:][::-1]:
                cv = c["data"]
                q = c.get("quality") or {}
                has_gaps = bool(
                    q.get("missing_information") or q.get("unclear_information")
                )
                badge_cls = "kmu-badge-warn" if has_gaps else "kmu-badge-ok"
                badge_lbl = (
                    "Klärungsbedarf" if has_gaps else "Analyse abgeschlossen"
                )
                name = cv.get("name") or "(ohne Name)"
                first_role = ""
                if cv.get("experience"):
                    first_role = cv["experience"][0].get("role", "")
                st.markdown(
                    f"""
                    <div class="kmu-list-item">
                        <div class="kmu-list-avatar">{initials_for(name)}</div>
                        <div style="flex:1">
                            <div class="kmu-list-name">{name}</div>
                            <div class="kmu-list-file">{first_role or c['filename']}</div>
                        </div>
                        <span class="kmu-badge {badge_cls}">● {badge_lbl}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption(
            "Noch keine Bewerbungen analysiert. Lade unten Lebensläufe hoch."
        )

with dash_right:
    _recent_log = load_audit_log()[-7:][::-1]
    with st.container(border=True):
        st.markdown(
            '<div class="kmu-card-title">AI Activity '
            '<small>● live</small></div>',
            unsafe_allow_html=True,
        )
        if _recent_log:
            feed_rows = []
            for i, entry in enumerate(_recent_log):
                ts = entry.get("timestamp", "")
                t_disp = ts.split("T", 1)[1][:5] if "T" in ts else ts
                action = entry.get("action", "")
                target = entry.get("target", "")
                feed_rows.append(
                    f'<div class="feed-item" style="animation-delay:{i * 0.06}s">'
                    f'<span class="feed-ic">{_activity_icon(action)}</span>'
                    f'<span class="feed-body"><b>{action}</b>'
                    f'<small>{target}</small></span>'
                    f'<span class="feed-time">{t_disp}</span></div>'
                )
            st.markdown("".join(feed_rows), unsafe_allow_html=True)
        else:
            st.caption("Noch keine Agent-Aktivität.")


# ---------------------------------------------------------------------------
# Zentrale Upload-/Start-Card (verschoben aus der Sidebar)
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown(
        '<div class="kmu-card-title">Stellenprofil &amp; Bewerbungen</div>',
        unsafe_allow_html=True,
    )

    # Transiente Statusmeldungen, die einen st.rerun() überleben
    for _level, _msg in st.session_state._flash:
        getattr(st, _level, st.info)(_msg)
    st.session_state._flash = []

    up_left, up_right = st.columns(2)

    with up_left:
        st.markdown(
            '<div class="upload-head"><span class="upload-ic">📋</span>'
            '<div><div class="upload-title">Stellenprofil</div>'
            '<div class="upload-desc">Laden Sie die Stellenanzeige hoch '
            "oder fügen Sie sie ein.</div></div></div>",
            unsafe_allow_html=True,
        )
        job_url = st.text_input(
            "Stellen-URL (optional, derzeit deaktiviert)",
            placeholder="https://… (in dieser Demo nicht aktiv)",
            label_visibility="collapsed",
        )
        if job_url.strip():
            st.caption(
                "Hinweis: URL-Fetch ist in dieser Demo nicht aktiviert. "
                "Bitte den Stellentext direkt unten einfügen."
            )
        job_text = st.text_area(
            "Stellenprofil",
            height=160,
            placeholder=(
                "z. B. Wir suchen einen Python-Entwickler mit SQL und "
                "Deutsch C1 …"
            ),
            label_visibility="collapsed",
        )
        if st.button(
            "Stelle analysieren", type="primary",
            disabled=not job_text.strip(),
        ):
            with st.spinner("Strukturiere Stellenprofil ..."):
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
                    # Aufteilung in prüfbar / nicht prüfbar protokollieren
                    _all_req = _collect_all_requirements(profile)
                    _check, _soft = filter_objectively_checkable_requirements(
                        _all_req
                    )
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

        if st.session_state.job_profile and st.button("Stellenprofil löschen"):
            st.session_state.job_profile = None
            st.session_state.job_profile_source = ""
            st.session_state._coverage_logged = set()
            st.rerun()

    with up_right:
        st.markdown(
            '<div class="upload-head"><span class="upload-ic green">📎</span>'
            '<div><div class="upload-title">Bewerbungen</div>'
            '<div class="upload-desc">Laden Sie mehrere Lebensläufe '
            "als PDF hoch.</div></div></div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "PDF-Dateien",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button(
            "Bewerbungen analysieren", type="primary", disabled=not uploaded
        ):
            prior_feedback = feedback_block(load_feedback())
            new_files = [
                f
                for f in uploaded
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
                    with st.spinner(f"Verarbeite {f.name} ..."):
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
                            # Informationslücken-Agent
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
                            # Rückfragen-Agent
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
                            # Abgleich der prüfbaren Anforderungen protokollieren
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

        if st.session_state.candidates and st.button("Alle Kandidaten löschen"):
            st.session_state.candidates = []
            st.session_state.processed_files = set()
            st.session_state.reviewed_questions = {}
            st.session_state._coverage_logged = set()
            st.rerun()


# ---------------------------------------------------------------------------
# Horizontale Funktionsnavigation (Tabs) + dezente Human-in-the-Loop-Zeile
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hil-line">Der Agent strukturiert Informationen. '
    "Der Mensch entscheidet."
    "</div>",
    unsafe_allow_html=True,
)

(
    main_tab_profile,
    main_tab_candidates,
    main_tab_questions,
    main_tab_feedback,
    main_tab_audit,
) = st.tabs(
    [
        "Stellenprofil",
        "Kandidatenübersicht",
        "Rückfragen",
        "Feedback",
        "Audit Log",
    ]
)

# ---- Tab: Stellenprofil ----

with main_tab_profile:
    st.markdown(
        '<div class="kmu-card-title">Strukturiertes Stellenprofil</div>',
        unsafe_allow_html=True,
    )
    st.caption("Strukturierte Anforderungen — keine Bewerber-Bewertung.")
    if st.session_state.job_profile:
        jp = st.session_state.job_profile
        st.markdown(f"**Rolle:** {jp.get('role') or NICHT_GEFUNDEN}")
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
            "Alle Kandidaten in Upload-Reihenfolge. Keine Bewertung, "
            "keine Sortierung nach %."
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
        job_profile_state = st.session_state.job_profile
        # Abdeckungsgrad pro Kandidat einmalig protokollieren (kein Spam).
        if job_profile_state:
            for _c in filtered:
                _cov = calculate_requirement_coverage(
                    job_profile_state, _c["data"]
                )
                log_coverage_once(_c, _cov, job_profile_state)
        # Moderne Kandidaten-Karten (statt Tabelle)
        options = filtered
        if options:
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
                _badge_cls = "kmu-badge-warn" if _has_gaps else "kmu-badge-ok"
                _badge_lbl = (
                    "Klärungsbedarf" if _has_gaps else "Analyse abgeschlossen"
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
                    _metric = f"{_rc['Gefunden']} / {_tot}" if _tot else "—"
                else:
                    _metric = "—"
                _skills = _d.get("skills") or []
                _skill_badges = " ".join(
                    f'<span class="kmu-badge kmu-badge-info">{s}</span>'
                    for s in _skills[:4]
                ) + (
                    f' <span class="kmu-badge kmu-badge-muted">+{len(_skills) - 4}</span>'
                    if len(_skills) > 4
                    else ""
                )
                _klaer_n = len(
                    _q.get("missing_information") or []
                ) + len(_q.get("unclear_information") or [])
                _is_open = _c["filename"] == st.session_state._open_cand
                with st.container(border=True):
                    cc = st.columns([3, 1.3, 3, 1.5])
                    with cc[0]:
                        st.markdown(
                            f'<div class="kmu-list-avatar" style="float:left;'
                            f'margin-right:10px">{initials_for(_name)}</div>'
                            f'<div class="cand-name">{_name}</div>'
                            f'<div class="cand-role">{_role or _c["filename"]}</div>',
                            unsafe_allow_html=True,
                        )
                    with cc[1]:
                        st.markdown(
                            f'<div class="cand-metric">{_metric}</div>'
                            '<div class="cand-metric-label">Anforderungen gefunden</div>',
                            unsafe_allow_html=True,
                        )
                    with cc[2]:
                        _status_badge = (
                            f'<span class="kmu-badge kmu-badge-warn">'
                            f"🧩 Klärungsbedarf: {_klaer_n}</span>"
                            if _klaer_n
                            else f'<span class="kmu-badge {_badge_cls}">{_badge_lbl}</span>'
                        )
                        st.markdown(
                            f'<div style="margin:2px 0 6px">{_skill_badges or "—"}</div>'
                            f"{_status_badge}",
                            unsafe_allow_html=True,
                        )
                    with cc[3]:
                        if st.button(
                            "Profil geöffnet" if _is_open else "Profil öffnen",
                            key=f"open_{_c['filename']}",
                            disabled=_is_open,
                            use_container_width=True,
                        ):
                            st.session_state._open_cand = _c["filename"]
                            st.rerun()

            st.markdown(
                '<div class="section-title">Kandidatenprofil</div>',
                unsafe_allow_html=True,
            )
            selected = next(
                c
                for c in filtered
                if c["filename"] == st.session_state._open_cand
            )
            cv_data = selected["data"]
            quality = selected.get("quality") or {}
            followups_q = (selected.get("followups") or {}).get(
                "questions"
            ) or []
            # Neue Kern-Logik: Anforderungen abgleichen (ohne Prozent).
            req_rows = evaluate_candidate_requirements(
                st.session_state.job_profile, cv_data
            )
            req_counts = status_counts(req_rows)
            req_total = len(req_rows)
            not_checkable = get_not_checkable_requirements(
                st.session_state.job_profile
            )
            klaerung_items = [
                r for r in req_rows if r["status"] == "Teilweise gefunden"
            ]

            # ---- Detailfenster (Card, kompakt, mit Badges) ----
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                with head_l:
                    st.markdown(
                        f"### {cv_data.get('name') or NICHT_GEFUNDEN}"
                    )
                    if cv_data.get("experience"):
                        first = cv_data["experience"][0]
                        sub = (
                            f"{first.get('role', '')} @ "
                            f"{first.get('company', '')}"
                        ).strip(" @")
                        if sub:
                            st.caption(sub)
                with head_r:
                    if req_total:
                        st.markdown(
                            f"""
                            <div class="coverage-box">
                                <div class="coverage-value">{req_counts['Gefunden']} / {req_total}</div>
                                <div class="coverage-label">
                                    Anforderungen gefunden
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Kein Stellenprofil hinterlegt.")

                st.divider()
                col_left, col_right = st.columns(2)

                with col_left:
                    # Berufserfahrung
                    st.markdown("**Berufserfahrung**")
                    exp = cv_data.get("experience") or []
                    if exp:
                        for e in exp[:5]:
                            line = (
                                f"- {e.get('role') or '?'} @ "
                                f"{e.get('company') or '?'}"
                            )
                            if e.get("period"):
                                line += f" · {e['period']}"
                            st.markdown(line)
                    else:
                        st.caption(NICHT_GEFUNDEN)
                    # Ausbildung
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
                    # Skills als Badges
                    skills = cv_data.get("skills") or []
                    if skills:
                        badges = " ".join(
                            f'<span class="kmu-badge kmu-badge-info">{s}</span>'
                            for s in skills[:12]
                        )
                        more = (
                            f' <span class="kmu-badge kmu-badge-info">+{len(skills) - 12}</span>'
                            if len(skills) > 12
                            else ""
                        )
                        st.markdown(
                            f"**Skills**<br>{badges}{more}",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**Skills:** {NICHT_GEFUNDEN}")
                    # Sprachen
                    st.markdown(f"**Sprachen:** {_short_languages(cv_data)}")
                    # Zertifikate
                    certs = cv_data.get("certificates") or []
                    st.markdown(
                        f"**Zertifikate:** "
                        f"{', '.join(certs) if certs else NICHT_GEFUNDEN}"
                    )

                # ---- Gefundene Anforderungen (fachlich prüfbar) ----
                STATUS_ICON_MAP = {
                    "Gefunden": "✅",
                    "Teilweise gefunden": "⚠️",
                    "Nicht gefunden": "❌",
                }
                if req_rows:
                    st.markdown("### Gefundene Anforderungen")
                    st.caption(
                        f"Gefunden: **{req_counts['Gefunden']} von "
                        f"{req_total}** fachlich prüfbaren Anforderungen "
                        f"·  ✅ {req_counts['Gefunden']}  "
                        f"⚠️ {req_counts['Teilweise gefunden']}  "
                        f"❌ {req_counts['Nicht gefunden']}"
                    )
                    for r in req_rows:
                        icon = STATUS_ICON_MAP.get(r["status"], "•")
                        hinweis = r.get("reason") or ""
                        with st.expander(
                            f"{icon} **{r['requirement']}** — "
                            f"{r['status']} · {hinweis}"
                        ):
                            src = r.get("evidence") or find_source_excerpt(
                                r["requirement"], cv_data
                            )
                            if src:
                                st.markdown("**Fundstelle im Lebenslauf:**")
                                st.markdown(f"> {src}")
                            else:
                                st.caption("Keine konkrete Fundstelle erfasst.")

                # ---- Inline Feedback + Technische Details ----
                with st.expander("Feedback geben"):
                    st.caption(
                        "Feedback verbessert nur die Extraktion, nicht die Auswahl."
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

                with st.expander("Technische Details (JSON)"):
                    st.json(cv_data)
        else:
            st.write("Keine Bewerber entsprechen den Filtern.")

# ---- Tab: Rückfragen (Human-in-the-Loop) ----

with main_tab_questions:
    st.markdown(
        '<div class="kmu-card-title">Vorgeschlagene Rückfragen</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Vorschläge des Rückfragen-Agenten. "
        "**Es wird keine E-Mail automatisch versendet.**"
    )
    any_questions = False
    for c in st.session_state.candidates:
        questions = (c.get("followups") or {}).get("questions") or []
        if not questions:
            continue
        any_questions = True
        fname = c["filename"]
        reviewed_set = st.session_state.reviewed_questions.setdefault(
            fname, set()
        )
        with st.expander(
            f"{c['data'].get('name') or '(ohne Name)'} — "
            f"{len(questions)} Rückfrage(n)"
        ):
            for i, q in enumerate(questions):
                is_reviewed = i in reviewed_set
                row_class = "question-row reviewed" if is_reviewed else "question-row"
                tag = (
                    '<span class="kmu-badge kmu-badge-ok">geprüft</span>'
                    if is_reviewed
                    else '<span class="kmu-badge kmu-badge-info">offen</span>'
                )
                st.markdown(
                    f'<div class="{row_class}">{q} &nbsp; {tag}</div>',
                    unsafe_allow_html=True,
                )
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if not is_reviewed and st.button(
                        "Rückfrage prüfen",
                        key=f"check_{fname}_{i}",
                    ):
                        log_audit(
                            action="Rückfrage geprüft",
                            target=c["data"].get("name", "") or fname,
                            result_type=q[:80],
                        )
                        st.toast(
                            "Rückfrage markiert als geprüft.", icon="✅"
                        )
                with btn_col2:
                    if not is_reviewed and st.button(
                        "Als geprüft markieren",
                        key=f"mark_{fname}_{i}",
                    ):
                        reviewed_set.add(i)
                        log_audit(
                            action="Rückfrage als geprüft markiert",
                            target=c["data"].get("name", "") or fname,
                            result_type=q[:80],
                        )
                        st.rerun()
    if not any_questions:
        st.info("Keine offenen Rückfragevorschläge.")

# ---- Tab: Feedback (Gesamtübersicht) ----

with main_tab_feedback:
    st.markdown(
        '<div class="kmu-card-title">Feedback-Übersicht</div>',
        unsafe_allow_html=True,
    )
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

# ---- Tab: Audit Log ----

with main_tab_audit:
    st.markdown(
        '<div class="kmu-card-title">Audit-Log</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Autonomie-Stufe: {AUTONOMY_LEVEL}")
    log_entries = load_audit_log()
    if log_entries:
        # Timeline / Activity-Feed (neueste zuerst)
        tl_items = []
        for entry in log_entries[-40:][::-1]:
            ts = entry.get("timestamp", "")
            # Uhrzeit aus ISO-Timestamp extrahieren (HH:MM), Fallback ts
            t_disp = ts
            if "T" in ts:
                t_disp = ts.split("T", 1)[1][:5]
            action = entry.get("action", "")
            target = entry.get("target", "")
            result = entry.get("result_type", "")
            sub = " · ".join(x for x in [target, result] if x)
            tl_items.append(
                f'<div class="tl-item"><div class="tl-dot"></div>'
                f'<div class="tl-time">{t_disp}</div>'
                f'<div class="tl-action">{action}</div>'
                f'<div class="tl-target">{sub}</div></div>'
            )
        st.markdown(
            '<div class="timeline">' + "".join(tl_items) + "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Tabellenansicht"):
            st.dataframe(
                pd.DataFrame(log_entries),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Audit-Log ist leer.")

st.caption(
    "Der Mensch entscheidet. Recruiting AI strukturiert Informationen — "
    "die Entscheidung trifft der Geschäftsführer."
)
