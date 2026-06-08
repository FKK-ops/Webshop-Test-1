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
        flex-wrap: wrap;
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
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        min-height: 130px;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .kpi-card.teal::before { background: #14B8A6; }
    .kpi-card.orange::before { background: #F59E0B; }
    .kpi-card.blue::before { background: #3B82F6; }
    .kpi-card.green::before { background: #10B981; }
    .kpi-card.red::before { background: #EF4444; }
    .kpi-value {
        font-size: 34px;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 4px;
    }
    .kpi-value.teal { color: #14B8A6; }
    .kpi-value.orange { color: #F59E0B; }
    .kpi-value.blue { color: #3B82F6; }
    .kpi-value.green { color: #10B981; }
    .kpi-value.red { color: #EF4444; }
    .kpi-label {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 10px;
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
        display: flex; align-items: center; gap: 18px; color: #94A3B8;
    }
    .kmu-topbar-icon { position: relative; font-size: 18px; cursor: default; }
    .kmu-topbar-icon .badge {
        position: absolute; top: -8px; right: -8px;
        background: #EF4444; color: white;
        border-radius: 50%;
        width: 16px; height: 16px;
        font-size: 10px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
    }
    .kmu-avatar {
        background: #14B8A6; color: #07172A;
        width: 32px; height: 32px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 13px;
    }
    .dashboard-header {
        display: flex; justify-content: space-between; align-items: flex-end;
        margin-bottom: 20px;
    }
    .dashboard-header h1 { font-size: 30px !important; margin: 0 !important; font-weight: 700; }
    .dashboard-header .subtitle { color: #94A3B8; font-size: 14px; margin-top: 4px; }
    .kmu-quote {
        background: #0F1F35;
        border: 1px solid #1E3A5F;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        display: flex; gap: 18px; align-items: flex-start;
    }
    .kmu-quote-mark {
        color: #14B8A6; font-size: 44px; line-height: 0.8;
        font-family: Georgia, serif;
    }
    .kmu-quote-body { flex: 1; }
    .kmu-quote-text { font-style: italic; color: #F8FAFC; font-size: 16px; line-height: 1.5; }
    .kmu-quote-source { color: #14B8A6; font-size: 13px; margin-top: 14px; }
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
    .kmu-badge-info { background: rgba(59, 130, 246, 0.16); color: #3B82F6; }
    .kmu-list-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #1E3A5F;
        gap: 12px;
    }
    .kmu-list-item:last-child { border-bottom: none; }
    .kmu-list-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        background: #0F9181; color: #07172A;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 13px; flex-shrink: 0;
    }
    .kmu-list-name { font-weight: 600; color: #F8FAFC; font-size: 14px; }
    .kmu-list-file { color: #94A3B8; font-size: 12px; margin-top: 2px; }
    .kmu-logo {
        font-size: 17px; font-weight: 700; color: #14B8A6;
        padding: 4px 8px 18px 8px;
    }
    .kmu-nav-section {
        color: #94A3B8; font-size: 11px; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin: 16px 8px 6px 8px;
    }
    .kmu-nav-item {
        display: flex; align-items: center; gap: 10px;
        padding: 9px 12px; border-radius: 8px;
        color: #94A3B8; font-size: 14px; margin: 2px 4px;
    }
    .kmu-nav-item.active {
        background: #0F9181; color: #F8FAFC; font-weight: 600;
    }
    .kmu-nav-item .badge-num {
        margin-left: auto;
        background: #14B8A6; color: #07172A;
        padding: 1px 7px; border-radius: 10px;
        font-size: 11px; font-weight: 700;
    }
    .donut-wrap { display: flex; align-items: center; gap: 24px; }
    .donut-legend { font-size: 13px; flex: 1; }
    .donut-legend-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 0; color: #F8FAFC;
    }
    .donut-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
    .donut-count { margin-left: auto; color: #94A3B8; font-weight: 600; }
    .question-row {
        background: #10233A;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #F8FAFC;
        font-size: 14px;
    }
    .question-row.reviewed { border-left: 4px solid #10B981; }
    /* Zentrale Upload-/Start-Card (st.container(border=True)) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #0F1F35;
        border: 1px solid #1E3A5F !important;
        border-radius: 14px;
    }
    /* Dezente Human-in-the-Loop-Infozeile an der Navigation */
    .hil-line {
        color: #94A3B8;
        font-size: 12px;
        padding: 8px 14px;
        margin: 4px 0 10px 0;
        border-left: 3px solid #14B8A6;
        background: rgba(20, 184, 166, 0.06);
        border-radius: 6px;
    }
    /* Abdeckungsgrad-Anzeige (neutral, keine Bewertung) */
    .coverage-box {
        background: #10233A;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0 4px 0;
    }
    .coverage-value { font-size: 26px; font-weight: 700; color: #14B8A6; }
    .coverage-label { color: #94A3B8; font-size: 12px; margin-top: 4px; }
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

with st.sidebar:
    st.markdown(
        '<div class="kmu-logo">👥 KMU Recruiting Agent</div>',
        unsafe_allow_html=True,
    )
    st.caption("Demo-Version · KMU Recruiting Agent")
    st.divider()
    st.button("⚙️ Einstellungen", use_container_width=True)
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
        help="Stellenprofil & Lebensläufe lädst du unten im Bereich „Stellenprofil & Bewerbungen“ hoch.",
    )

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
buckets = {"vollständig": 0, "informationslücken": 0, "unvollständig": 0}
for c in st.session_state.candidates:
    buckets[quality_bucket(c.get("quality"))] += 1
complete_count = buckets["vollständig"]
gap_count = buckets["informationslücken"] + buckets["unvollständig"]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card teal">
            <div class="kpi-value teal">{total_candidates}</div>
            <div class="kpi-label">analysierte Bewerbungen</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card orange">
            <div class="kpi-value orange">{total_gaps}</div>
            <div class="kpi-label">erkannte Informationslücken (Datenqualität)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card blue">
            <div class="kpi-value blue">{total_questions}</div>
            <div class="kpi-label">vorbereitete Rückfragen (warten auf Freigabe)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card green">
            <div class="kpi-value green">{complete_count} / {gap_count}</div>
            <div class="kpi-label">vollständig / mit Informationslücken</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Letzte Bewerbungen (kompakte Liste, echte Daten oder Platzhalter-Hinweis) ----

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
                "Informationslücken" if has_gaps else "Analyse abgeschlossen"
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
        "Noch keine Bewerbungen analysiert. Lade oben Lebensläufe hoch."
    )


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
        st.markdown("**Stellenprofil eingeben**")
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
            "Stellenprofil analysieren", disabled=not job_text.strip()
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
        st.markdown("**Lebensläufe hochladen (PDF)**")
        uploaded = st.file_uploader(
            "PDF-Dateien",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button(
            "Extraktion starten", type="primary", disabled=not uploaded
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
    '<div class="hil-line">Der Agent bewertet keine Bewerber. '
    "Er zeigt nur nachweisbare Informationen. "
    "Die Entscheidung trifft der Geschäftsführer."
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
        df = candidates_to_dataframe(filtered, job_profile_state)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if job_profile_state:
            st.caption("„Abgleich“ = nur fachliche Anforderungen, keine Sortierung.")

        options = [
            f"{c['data'].get('name') or '(ohne Name)'} – {c['filename']}"
            for c in filtered
        ]
        if options:
            idx = st.selectbox(
                "Kandidat öffnen",
                range(len(options)),
                format_func=lambda i: options[i],
            )
            selected = filtered[idx]
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
                                    fachlich gefunden
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Kein Stellenprofil hinterlegt.")

                # Kurze Badge-Reihen
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
                        f"**Skills** {badges}{more}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**Skills:** {NICHT_GEFUNDEN}")

                lang_text = _short_languages(cv_data)
                st.markdown(f"**Sprachen:** {lang_text}")

                certs = cv_data.get("certificates") or []
                st.markdown(
                    f"**Zertifikate:** {', '.join(certs) if certs else NICHT_GEFUNDEN}"
                )

                # Berufserfahrung + Ausbildung als kurze Stichpunkte
                exp = cv_data.get("experience") or []
                if exp:
                    st.markdown("**Berufserfahrung**")
                    for e in exp[:5]:
                        line = (
                            f"- {e.get('role') or '?'} @ "
                            f"{e.get('company') or '?'}"
                        )
                        if e.get("period"):
                            line += f" · {e['period']}"
                        st.markdown(line)
                edu = cv_data.get("education") or []
                if edu:
                    st.markdown("**Ausbildung**")
                    for e in edu[:3]:
                        st.markdown(
                            f"- {e.get('degree') or '?'}, "
                            f"{e.get('institution') or '?'}"
                            + (f" · {e['period']}" if e.get('period') else "")
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
        st.dataframe(
            pd.DataFrame(log_entries),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Audit-Log ist leer.")

st.caption(
    "Die KI trifft keine Personalentscheidung. Sie strukturiert objektive "
    "Informationen und macht Bewerbungen vergleichbar. Der Geschäftsführer "
    "prüft und entscheidet final."
)
