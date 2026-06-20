"""
Recruiting AI — Multi-Agent HR Demo
===================================

A premium SaaS-style Streamlit application built around a fully deterministic
multi-agent recruiting workflow. The app works without any external API key:
all extraction and matching is done with transparent, rule-based demo logic.

Architecture
------------
The code is intentionally split into three layers:

1. BACKEND / AGENT LOGIC      -> the `agents` section (pure functions + dataclasses)
2. DATA STRUCTURES            -> dataclasses (Job, Requirement, Candidate, ...)
3. UI / LAYOUT / CSS          -> the `ui` section (render_* functions)

Human-in-the-loop principle
---------------------------
The agents only EXTRACT and COMPARE information. They never rank candidates,
never produce a hiring recommendation and never make a decision. The final
judgement is always left to a human recruiter.
"""

from __future__ import annotations

import base64
import io
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

# Optional PDF support — gracefully degrade if pypdf is unavailable.
try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PYPDF = False


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

EDU_LEVELS = {
    "kein abschluss": 0,
    "ausbildung": 1,
    "bachelor": 2,
    "master": 3,
    "diplom": 3,
    "mba": 3,
    "phd": 4,
    "promotion": 4,
    "doktor": 4,
}
EDU_LABELS = {0: "kein Abschluss", 1: "Ausbildung", 2: "Bachelor", 3: "Master / Diplom", 4: "Promotion"}


@dataclass
class Requirement:
    """A single, measurable requirement extracted from a job profile."""

    id: str
    label: str
    kind: str  # 'skill' | 'experience' | 'education' | 'language'
    value: object  # str for skill/language, int for experience years / education level
    importance: str = "muss"  # 'muss' | 'kann'


@dataclass
class Candidate:
    """Structured candidate information extracted from a CV."""

    id: str
    name: str
    source: str
    raw_text: str
    skills: list = field(default_factory=list)
    years: Optional[int] = None
    education: int = 0
    education_label: str = "unbekannt"
    languages: list = field(default_factory=list)
    email: Optional[str] = None
    # Computed by later agents:
    matches: list = field(default_factory=list)  # list[MatchResult]
    gaps: list = field(default_factory=list)  # list[Gap]
    questions: list = field(default_factory=list)  # list[str]


@dataclass
class MatchResult:
    requirement_id: str
    label: str
    kind: str
    importance: str
    status: str  # 'erfuellt' | 'teilweise' | 'offen' | 'nicht_erfuellt'
    note: str = ""


@dataclass
class Gap:
    requirement_label: str
    description: str


@dataclass
class AuditEntry:
    timestamp: str
    agent: str
    action: str
    detail: str


# =============================================================================
#  KNOWLEDGE BASE (deterministic demo extraction vocabulary)
# =============================================================================

SKILL_VOCAB = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "node"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react"],
    "SQL": ["sql", "postgres", "mysql"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Machine Learning": ["machine learning", "ml", "tensorflow", "pytorch"],
    "Excel": ["excel"],
    "SAP": ["sap"],
    "Salesforce": ["salesforce"],
    "Projektmanagement": ["projektmanagement", "project management", "scrum", "agile", "kanban"],
    "Kommunikation": ["kommunikation", "communication"],
    "Vertrieb": ["vertrieb", "sales"],
    "Marketing": ["marketing", "seo", "sea"],
    "Buchhaltung": ["buchhaltung", "accounting", "rechnungswesen"],
}

LANGUAGE_VOCAB = {
    "Deutsch": ["deutsch", "german"],
    "Englisch": ["englisch", "english"],
    "Französisch": ["französisch", "franzosisch", "french"],
    "Spanisch": ["spanisch", "spanish"],
    "Italienisch": ["italienisch", "italian"],
}


# =============================================================================
#  BACKEND / AGENT LOGIC
#  --------------------------------------------------------------------------
#  Every agent is a pure function: it takes input data and returns structured
#  output. None of them mutate global state or make hiring decisions.
# =============================================================================


def _find_terms(text: str, vocab: dict) -> list:
    """Return the canonical names whose synonyms appear in `text`."""
    found = []
    low = text.lower()
    for canonical, synonyms in vocab.items():
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn) + r"\b", low):
                found.append(canonical)
                break
    return found


def _detect_years(text: str) -> Optional[int]:
    """Extract the largest stated number of years of experience."""
    matches = re.findall(r"(\d{1,2})\s*\+?\s*(?:jahre|jahren|years?|j\.)", text.lower())
    nums = [int(m) for m in matches if int(m) <= 50]
    return max(nums) if nums else None


def _detect_education(text: str) -> tuple:
    """Return (level:int, label:str) for the highest education found."""
    low = text.lower()
    best = 0
    for key, level in EDU_LEVELS.items():
        if key in low and level > best:
            best = level
    return best, EDU_LABELS.get(best, "unbekannt")


def _detect_email(text: str) -> Optional[str]:
    m = re.search(r"[\w.\-+]+@[\w.\-]+\.\w+", text)
    return m.group(0) if m else None


# --- Agent 1: Stellenprofil-Agent -------------------------------------------

def stellenprofil_agent(text: str) -> list:
    """Extract measurable requirements from a free-text job profile."""
    requirements: list = []
    low = text.lower()

    def importance_for(snippet: str) -> str:
        nice = ["wünschenswert", "wunschenswert", "von vorteil", "nice to have", "kann", "optional"]
        return "kann" if any(n in snippet for n in nice) else "muss"

    # Skills
    for skill in _find_terms(text, SKILL_VOCAB):
        # look at the line containing the skill to judge importance
        line = next((ln for ln in low.splitlines() if skill.lower() in ln), "")
        requirements.append(
            Requirement(
                id=f"skill_{skill.lower().replace(' ', '_')}",
                label=skill,
                kind="skill",
                value=skill,
                importance=importance_for(line),
            )
        )

    # Experience (years)
    years = _detect_years(text)
    if years:
        requirements.append(
            Requirement(
                id="exp_years",
                label=f"{years}+ Jahre Berufserfahrung",
                kind="experience",
                value=years,
                importance="muss",
            )
        )

    # Education
    edu_level, edu_label = _detect_education(text)
    if edu_level > 0:
        requirements.append(
            Requirement(
                id="education",
                label=f"Bildungsabschluss: {edu_label}",
                kind="education",
                value=edu_level,
                importance="muss",
            )
        )

    # Languages
    for lang in _find_terms(text, LANGUAGE_VOCAB):
        requirements.append(
            Requirement(
                id=f"lang_{lang.lower()}",
                label=f"Sprache: {lang}",
                kind="language",
                value=lang,
                importance="muss",
            )
        )

    return requirements


# --- Agent 2: CV-Agent ------------------------------------------------------

def cv_agent(name: str, source: str, text: str, idx: int) -> Candidate:
    """Extract structured candidate information from raw CV text."""
    skills = _find_terms(text, SKILL_VOCAB)
    languages = _find_terms(text, LANGUAGE_VOCAB)
    years = _detect_years(text)
    edu_level, edu_label = _detect_education(text)
    email = _detect_email(text)

    # Try to find a human-readable name from a "Name:" line, else use given name.
    display_name = name
    m = re.search(r"name\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        candidate_line = m.group(1).strip().splitlines()[0]
        if 2 < len(candidate_line) < 60:
            display_name = candidate_line

    return Candidate(
        id=f"cand_{idx}",
        name=display_name,
        source=source,
        raw_text=text,
        skills=skills,
        years=years,
        education=edu_level,
        education_label=edu_label if edu_level > 0 else "unbekannt",
        languages=languages,
        email=email,
    )


# --- Agent 3: Matching-Agent ------------------------------------------------

def matching_agent(candidate: Candidate, requirements: list) -> list:
    """Compare each requirement against the candidate's data (no ranking)."""
    results: list = []
    for req in requirements:
        status = "offen"
        note = ""

        if req.kind == "skill":
            if req.value in candidate.skills:
                status, note = "erfuellt", "Im Lebenslauf erwähnt."
            else:
                status, note = "offen", "Nicht im Lebenslauf erwähnt."

        elif req.kind == "experience":
            if candidate.years is None:
                status, note = "offen", "Keine Angabe zur Berufserfahrung gefunden."
            elif candidate.years >= int(req.value):
                status, note = "erfuellt", f"{candidate.years} Jahre angegeben."
            else:
                status, note = "teilweise", f"Nur {candidate.years} von {req.value} Jahren angegeben."

        elif req.kind == "education":
            if candidate.education == 0:
                status, note = "offen", "Kein Bildungsabschluss erkennbar."
            elif candidate.education >= int(req.value):
                status, note = "erfuellt", f"{candidate.education_label} vorhanden."
            else:
                status, note = "teilweise", f"{candidate.education_label} (gefordert: höher)."

        elif req.kind == "language":
            if req.value in candidate.languages:
                status, note = "erfuellt", "Sprache genannt."
            else:
                status, note = "offen", "Sprache nicht genannt."

        results.append(
            MatchResult(
                requirement_id=req.id,
                label=req.label,
                kind=req.kind,
                importance=req.importance,
                status=status,
                note=note,
            )
        )
    return results


# --- Agent 4: Informationslücken-Agent --------------------------------------

def informationsluecken_agent(candidate: Candidate, matches: list) -> list:
    """Identify missing or unclear information that blocks a clean assessment."""
    gaps: list = []
    for m in matches:
        if m.status in ("offen", "teilweise"):
            gaps.append(Gap(requirement_label=m.label, description=m.note))

    if not candidate.email:
        gaps.append(Gap(requirement_label="Kontaktdaten", description="Keine E-Mail-Adresse im Lebenslauf gefunden."))
    if candidate.years is None:
        gaps.append(Gap(requirement_label="Berufserfahrung", description="Dauer der Berufserfahrung unklar."))

    return gaps


# --- Agent 5: Rückfragen-Agent ----------------------------------------------

def rueckfragen_agent(candidate: Candidate, gaps: list) -> list:
    """Generate polite, candidate-facing follow-up questions for each gap."""
    questions: list = []
    seen = set()
    for gap in gaps:
        label = gap.requirement_label
        if label in seen:
            continue
        seen.add(label)

        if label == "Kontaktdaten":
            q = "Könnten Sie uns bitte eine aktuelle E-Mail-Adresse für die Kontaktaufnahme nennen?"
        elif label == "Berufserfahrung":
            q = "Wie viele Jahre einschlägige Berufserfahrung bringen Sie insgesamt mit?"
        elif label.startswith("Sprache:"):
            lang = label.replace("Sprache:", "").strip()
            q = f"Auf welchem Niveau beherrschen Sie {lang}? (z. B. fließend, verhandlungssicher, Muttersprache)"
        elif label.startswith("Bildungsabschluss"):
            q = "Welchen höchsten Bildungsabschluss haben Sie und in welchem Fachbereich?"
        elif "Jahre Berufserfahrung" in label:
            q = "Könnten Sie Ihre relevante Berufserfahrung für diese Position kurz beschreiben?"
        else:
            q = f"Könnten Sie Ihre Erfahrung im Bereich „{label}“ näher erläutern?"

        questions.append(q)
    return questions


def generate_email(candidate: Candidate, selected_questions: list) -> tuple:
    """Compose a polite follow-up e-mail draft from the selected questions.

    Deterministic, template-based generation (no API key required). The draft
    is NEVER sent automatically — it is only prepared for human review.
    """
    to = candidate.email or "bewerber@example.com"
    subject = "Rückfragen zu Ihrer Bewerbung"
    lines = [
        f"Sehr geehrte/r {candidate.name},",
        "",
        "vielen Dank für Ihre Bewerbung und Ihr Interesse an der ausgeschriebenen "
        "Position. Um Ihre Unterlagen vollständig und fair bewerten zu können, "
        "hätten wir noch einige kurze Rückfragen an Sie:",
        "",
    ]
    for i, q in enumerate(selected_questions, 1):
        lines.append(f"{i}. {q}")
    lines += [
        "",
        "Über eine kurze Rückmeldung würden wir uns sehr freuen. Bei Fragen stehen "
        "wir Ihnen jederzeit gern zur Verfügung.",
        "",
        "Mit freundlichen Grüßen",
        "Ihr Recruiting-Team",
    ]
    return to, subject, "\n".join(lines)


# --- Agent 6: Audit-Log -----------------------------------------------------

def audit(agent: str, action: str, detail: str = "") -> None:
    """Append a tamper-evident processing record to the audit log."""
    st.session_state.audit.append(
        AuditEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agent=agent,
            action=action,
            detail=detail,
        )
    )


# --- Utilities --------------------------------------------------------------

def extract_text_from_upload(uploaded_file) -> str:
    """Read text from an uploaded PDF or TXT file with graceful fallback."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf") and HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def run_full_analysis() -> None:
    """Run the matching, gap and follow-up agents for every candidate."""
    reqs = st.session_state.requirements
    for cand in st.session_state.candidates:
        cand.matches = matching_agent(cand, reqs)
        cand.gaps = informationsluecken_agent(cand, cand.matches)
        cand.questions = rueckfragen_agent(cand, cand.gaps)
    audit(
        "Matching-Agent",
        "Anforderungsabgleich durchgeführt",
        f"{len(st.session_state.candidates)} Kandidat(en) gegen {len(reqs)} Anforderung(en) geprüft.",
    )
    audit("Informationslücken-Agent", "Lücken analysiert", "Fehlende Angaben pro Kandidat ermittelt.")
    audit("Rückfragen-Agent", "Rückfragen generiert", "Höfliche Rückfragen je Informationslücke erstellt.")
    st.session_state.analysis_done = True


# --- Demo seed data ---------------------------------------------------------

DEMO_JOB = """Senior Backend Entwickler (m/w/d)

Wir suchen eine erfahrene Persönlichkeit für unser Produktteam.

Anforderungen:
- Mindestens 5 Jahre Berufserfahrung in der Softwareentwicklung
- Sehr gute Kenntnisse in Python und SQL
- Erfahrung mit AWS und Docker
- Abgeschlossenes Studium (Bachelor) im Bereich Informatik
- Sehr gute Deutsch- und Englischkenntnisse
- Projektmanagement und Scrum wünschenswert
"""

DEMO_CVS = [
    (
        "Lena Hoffmann",
        """Name: Lena Hoffmann
E-Mail: lena.hoffmann@example.com
Berufserfahrung: 7 Jahre als Softwareentwicklerin
Skills: Python, SQL, AWS, Docker, Scrum
Ausbildung: Master Informatik
Sprachen: Deutsch, Englisch
""",
    ),
    (
        "Tomáš Novák",
        """Name: Tomas Novak
Berufserfahrung: 3 Jahre Backend
Skills: Python, JavaScript
Ausbildung: Bachelor
Sprachen: Englisch
""",
    ),
    (
        "Sarah Klein",
        """Name: Sarah Klein
E-Mail: sarah.klein@example.com
Skills: Java, SQL, Kubernetes
Ausbildung: Bachelor Wirtschaftsinformatik
Sprachen: Deutsch, Englisch, Französisch
""",
    ),
]


def load_demo_data() -> None:
    st.session_state.job_text = DEMO_JOB
    st.session_state.requirements = stellenprofil_agent(DEMO_JOB)
    audit("Stellenprofil-Agent", "Demo-Stellenprofil analysiert", f"{len(st.session_state.requirements)} Anforderungen erkannt.")
    st.session_state.candidates = []
    for i, (name, text) in enumerate(DEMO_CVS):
        st.session_state.candidates.append(cv_agent(name, "Demo-Daten", text, i))
    audit("CV-Agent", "Demo-Bewerbungen analysiert", f"{len(DEMO_CVS)} Lebensläufe verarbeitet.")
    run_full_analysis()


# =============================================================================
#  STATE INITIALISATION
# =============================================================================

def init_state() -> None:
    defaults = {
        "page": "home",
        "job_text": "",
        "requirements": [],
        "candidates": [],
        "audit": [],
        "analysis_done": False,
        "selected_candidate": None,
        "email_drafts": {},
        "intro_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
#  UI / LAYOUT / CSS
# =============================================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --rai-purple:#7c5cff;
  --rai-blue:#3b82f6;
  --rai-cyan:#22d3ee;
  --rai-ink:#0f1226;
  --rai-muted:#5b6178;
}

html, body, [class*="css"], .stApp{
  font-family:'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* App background: soft premium gradient */
.stApp{
  background:
    radial-gradient(1200px 600px at 12% -8%, rgba(124,92,255,.16), transparent 60%),
    radial-gradient(1000px 600px at 100% 0%, rgba(34,211,238,.14), transparent 55%),
    linear-gradient(180deg,#fbfbff 0%, #f5f6fc 100%);
}

/* Hide default Streamlit chrome for a cleaner product feel */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}

.block-container{padding-top:1.2rem; padding-bottom:4rem; max-width:1180px;}

/* Headings */
h1,h2,h3{ color:var(--rai-ink); letter-spacing:-.02em; }

/* Glass cards */
.glass{
  background:rgba(255,255,255,.65);
  backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.7);
  border-radius:24px;
  box-shadow:0 18px 50px rgba(31,38,93,.10);
  padding:30px 32px;
}

/* Section spacing */
.section{ margin:64px 0 24px; }
.eyebrow{
  display:inline-block; font-size:.78rem; font-weight:700; letter-spacing:.14em;
  text-transform:uppercase; color:var(--rai-purple);
  background:rgba(124,92,255,.10); padding:6px 14px; border-radius:999px; margin-bottom:14px;
}
.section h2{ font-size:clamp(1.7rem,3.4vw,2.6rem); font-weight:800; margin:0 0 10px; }
.section .lead{ color:var(--rai-muted); font-size:1.06rem; max-width:680px; line-height:1.6; }

/* Feature / problem grid */
.grid{ display:grid; gap:20px; }
.grid-3{ grid-template-columns:repeat(3,1fr); }
.grid-2{ grid-template-columns:repeat(2,1fr); }
@media (max-width:900px){ .grid-3,.grid-2{ grid-template-columns:1fr; } }

.card{
  background:rgba(255,255,255,.7);
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(124,92,255,.10);
  border-radius:22px; padding:26px 24px;
  box-shadow:0 12px 36px rgba(31,38,93,.07);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease, border-color .35s ease;
}
.card:hover{ transform:translateY(-6px); box-shadow:0 26px 60px rgba(124,92,255,.18); border-color:rgba(124,92,255,.30); }
.card .ic{
  width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  font-size:22px;margin-bottom:16px;
  background:linear-gradient(135deg,var(--rai-purple),var(--rai-blue)); color:#fff;
  box-shadow:0 8px 20px rgba(124,92,255,.35);
}
.card h3{ font-size:1.16rem; font-weight:700; margin:0 0 8px; }
.card p{ color:var(--rai-muted); font-size:.96rem; line-height:1.55; margin:0; }
.step-num{ font-size:.8rem;font-weight:800;color:var(--rai-blue);letter-spacing:.1em; }

/* Scroll reveal (driven by an IntersectionObserver — see render_scroll_reveal) */
@keyframes fadeUp{ from{opacity:0; transform:translateY(26px);} to{opacity:1; transform:translateY(0);} }
.reveal{ opacity:0; transform:translateY(36px);
  transition:opacity .85s ease, transform .85s cubic-bezier(.2,.8,.2,1); }
.reveal.reveal-in{ opacity:1; transform:none; }

/* Buttons (Streamlit) */
.stButton > button{
  border-radius:14px; font-weight:600; padding:.6rem 1.2rem; border:1px solid rgba(124,92,255,.25);
  background:rgba(255,255,255,.8); color:var(--rai-ink); transition:all .25s ease;
}
.stButton > button:hover{ border-color:var(--rai-purple); color:var(--rai-purple); transform:translateY(-2px); }
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg,var(--rai-purple),var(--rai-blue)); color:#fff; border:none;
  box-shadow:0 12px 30px rgba(124,92,255,.40);
}
.stButton > button[kind="primary"]:hover{ filter:brightness(1.06); transform:translateY(-2px); color:#fff; }

/* Status pills */
.pill{ display:inline-block; padding:4px 12px; border-radius:999px; font-size:.8rem; font-weight:700; }
.pill-ok{ background:rgba(16,185,129,.14); color:#0f9d6b; }
.pill-part{ background:rgba(245,158,11,.16); color:#b4790c; }
.pill-open{ background:rgba(99,102,241,.14); color:#4f46e5; }
.pill-no{ background:rgba(239,68,68,.14); color:#dc2626; }

/* HITL banner */
.hitl{
  background:linear-gradient(135deg, rgba(124,92,255,.12), rgba(34,211,238,.10));
  border:1px solid rgba(124,92,255,.22); border-radius:20px; padding:22px 26px; color:var(--rai-ink);
}
.hitl b{ color:var(--rai-purple); }

/* Candidate tiles */
.cand{
  background:rgba(255,255,255,.72); border:1px solid rgba(124,92,255,.12); border-radius:20px;
  padding:20px 22px; box-shadow:0 10px 30px rgba(31,38,93,.06);
  transition:transform .3s ease, box-shadow .3s ease;
}
.cand:hover{ transform:translateY(-4px); box-shadow:0 20px 44px rgba(124,92,255,.16); }
.cand .nm{ font-weight:700; font-size:1.1rem; color:var(--rai-ink); }
.cand .meta{ color:var(--rai-muted); font-size:.9rem; margin-top:4px; }

.tag{ display:inline-block; background:rgba(59,130,246,.10); color:#2563eb; border-radius:8px;
  padding:3px 9px; font-size:.78rem; font-weight:600; margin:3px 4px 0 0; }

/* Smooth scrolling between sections */
html{ scroll-behavior:smooth; }
body{ overflow-x:hidden; }

/* Full-bleed component iframes (Spline background graphics span the viewport).
   Streamlit gives the component iframe the title "st.iframe" in some versions
   and "streamlit_components.v1.html.html" in others — cover both, and make the
   iframe transparent so no white box shows around the scene. */
.element-container:has(iframe[title="st.iframe"]),
.element-container:has(iframe[title="streamlit_components.v1.html.html"]){
  width:100vw !important;
  margin-left:calc(50% - 50vw) !important;
  margin-right:calc(50% - 50vw) !important;
}
iframe[title="st.iframe"],
iframe[title="streamlit_components.v1.html.html"]{
  width:100vw !important; border:none !important; background:transparent !important;
  color-scheme:normal;
}

/* ===================== Premium scroll-storytelling ===================== */
/* Hero background image (bright pastel brand graphic) — full viewport */
.vhero{ position:relative; width:100vw; margin-left:calc(50% - 50vw);
  height:calc(100vh - 86px); min-height:560px; overflow:hidden; }
.vhero-bg{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block;
  /* fade the top & bottom so the hero blends smoothly into the sections
     before and after it (no hard seams) */
  -webkit-mask-image:linear-gradient(180deg, transparent 0, #000 13%, #000 80%, transparent 100%);
  mask-image:linear-gradient(180deg, transparent 0, #000 13%, #000 80%, transparent 100%); }
.vhero-scrim{ position:absolute; inset:0;
  background:
    linear-gradient(90deg, rgba(251,251,255,.94) 0%, rgba(251,251,255,.6) 36%, rgba(251,251,255,0) 64%),
    linear-gradient(180deg, rgba(251,251,255,.55), transparent 26%, transparent 72%, rgba(251,251,255,.95)); }
.vhero-copy{ position:absolute; top:50%; left:max(24px, calc(50vw - 530px)); transform:translateY(-50%);
  max-width:560px; z-index:2; }
.story{ max-width:1060px; margin:0 auto; padding:120px 0 24px; }
.story.first{ padding-top:56px; }
.kicker{ font-size:.8rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase;
  color:var(--rai-purple); margin-bottom:18px; }
.display{ font-size:clamp(2.5rem,5.4vw,4.6rem); font-weight:900; letter-spacing:-.035em;
  line-height:1.02; color:var(--rai-ink); margin:0; }
.display .g{ background:linear-gradient(120deg,#7c5cff,#3b82f6,#22d3ee);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.story .big{ font-size:1.28rem; line-height:1.65; color:var(--rai-muted); max-width:640px; margin:26px 0 0; }
.story .small{ font-size:1.04rem; line-height:1.6; color:var(--rai-muted); max-width:560px; margin:18px 0 0; }

.split{ display:grid; grid-template-columns:1.02fr .98fr; gap:64px; align-items:center; }
.split.rev{ grid-template-columns:.98fr 1.02fr; }
@media(max-width:900px){ .split,.split.rev{ grid-template-columns:1fr; gap:34px; } }

.media-wrap{ border-radius:30px; overflow:hidden; border:1px solid rgba(124,92,255,.12);
  box-shadow:0 36px 90px rgba(31,38,93,.18); line-height:0; }
.media-wrap img{ width:100%; height:auto; display:block; }

/* big stat row */
.stats{ display:grid; grid-template-columns:repeat(3,1fr); gap:30px; }
@media(max-width:900px){ .stats{ grid-template-columns:1fr; gap:18px; } }
.stat .n{ font-size:clamp(2.4rem,4.6vw,3.4rem); font-weight:900; letter-spacing:-.03em;
  background:linear-gradient(120deg,#7c5cff,#22d3ee); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; }
.stat .l{ color:var(--rai-muted); font-size:1.02rem; margin-top:6px; }

/* clean numbered steps */
.steps{ display:grid; grid-template-columns:repeat(3,1fr); gap:24px; }
@media(max-width:900px){ .steps{ grid-template-columns:1fr; } }
.stepcard{ padding:30px 26px; border-radius:24px; background:rgba(255,255,255,.7);
  border:1px solid rgba(124,92,255,.10); box-shadow:0 14px 40px rgba(31,38,93,.07);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease; }
.stepcard:hover{ transform:translateY(-6px); box-shadow:0 28px 64px rgba(124,92,255,.16); }
.stepcard .k{ font-size:.82rem; font-weight:800; letter-spacing:.12em; color:var(--rai-blue); }
.stepcard h3{ font-size:1.2rem; font-weight:800; margin:12px 0 8px; }
.stepcard p{ color:var(--rai-muted); font-size:.98rem; line-height:1.55; margin:0; }

/* pricing */
.pricing{ display:grid; grid-template-columns:repeat(3,1fr); gap:24px; align-items:stretch; }
@media(max-width:900px){ .pricing{ grid-template-columns:1fr; } }
.price{ display:flex; flex-direction:column; padding:34px 30px; border-radius:26px;
  background:rgba(255,255,255,.72); border:1px solid rgba(124,92,255,.12);
  box-shadow:0 16px 44px rgba(31,38,93,.08); }
.price.feat{ background:linear-gradient(160deg,rgba(124,92,255,.10),rgba(34,211,238,.08));
  border:1px solid rgba(124,92,255,.30); box-shadow:0 26px 70px rgba(124,92,255,.20); }
.price .pn{ font-weight:800; font-size:1.05rem; color:var(--rai-ink); }
.price .pp{ font-size:2.6rem; font-weight:900; letter-spacing:-.03em; margin:10px 0 2px; color:var(--rai-ink); }
.price .pp small{ font-size:.95rem; font-weight:600; color:var(--rai-muted); }
.price ul{ list-style:none; padding:0; margin:18px 0 0; }
.price li{ color:var(--rai-muted); font-size:.98rem; padding:7px 0 7px 26px; position:relative; }
.price li:before{ content:"✓"; position:absolute; left:0; color:#0f9d6b; font-weight:800; }
.price .badge-pop{ align-self:flex-start; font-size:.72rem; font-weight:800; letter-spacing:.08em;
  text-transform:uppercase; color:#fff; background:linear-gradient(135deg,#7c5cff,#3b82f6);
  padding:5px 12px; border-radius:999px; margin-bottom:12px; }

/* horizontal scroll gallery (Raycast-style) */
.bleed{ width:100vw; margin-left:calc(50% - 50vw); }
.hscroll{ display:flex; gap:24px; overflow-x:auto; scroll-snap-type:x mandatory;
  -webkit-overflow-scrolling:touch; padding:12px 48px 34px;
  /* soft fade on the left and right edges */
  -webkit-mask-image:linear-gradient(90deg, transparent 0, #000 110px, #000 calc(100% - 110px), transparent 100%);
  mask-image:linear-gradient(90deg, transparent 0, #000 110px, #000 calc(100% - 110px), transparent 100%); }
.hscroll::-webkit-scrollbar{ height:8px; }
.hscroll::-webkit-scrollbar-thumb{ background:rgba(124,92,255,.28); border-radius:99px; }
.hscroll::-webkit-scrollbar-track{ background:transparent; }
/* show ~3 cards at a time */
.hcard{ flex:0 0 calc((100vw - 144px) / 3); max-width:560px; scroll-snap-align:center;
  border-radius:24px; overflow:hidden; background:#fff; border:1px solid rgba(124,92,255,.12);
  box-shadow:0 16px 44px rgba(31,38,93,.09);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease; }
@media(max-width:760px){ .hcard{ flex:0 0 80vw; } }
.hcard:hover{ transform:translateY(-6px); box-shadow:0 28px 64px rgba(124,92,255,.18); }
.hcard img{ width:100%; height:300px; object-fit:cover; display:block;
  background:linear-gradient(160deg,#f6f5ff,#eafaff); }
.hcard .cap{ padding:18px 20px; }
.hcard .cap .k{ font-size:.74rem; font-weight:800; letter-spacing:.12em; color:var(--rai-blue); }
.hcard .cap h4{ margin:6px 0 6px; font-size:1.08rem; font-weight:800; color:var(--rai-ink); }
.hcard .cap p{ margin:0; color:var(--rai-muted); font-size:.92rem; line-height:1.5; }

/* Problem section: text overlaid on the full-bleed Spline background */
.problem-wrap{ position:relative; z-index:3; margin-top:-420px; height:420px; pointer-events:none; }
.problem-copy{ max-width:1060px; margin:0 auto; padding-top:54px; position:relative; }
.problem-copy .display{ text-shadow:0 2px 24px rgba(255,255,255,.6); }
.problem-copy .big{ text-shadow:0 1px 16px rgba(255,255,255,.7); }

/* final cta band */
.cta-band{ text-align:center; padding:70px 40px; border-radius:34px; margin:40px auto 0; max-width:1060px;
  background:linear-gradient(140deg,rgba(124,92,255,.12),rgba(34,211,238,.10));
  border:1px solid rgba(124,92,255,.20); }
</style>
"""


def _logo_data_uri() -> Optional[str]:
    """Return a base64 data URI for a logo file if one exists in the repo."""
    for path in ("assets/logo.png", "logo.png", "assets/logo.svg", "logo.svg",
                 "assets/logo.webp", "logo.webp"):
        if os.path.exists(path):
            mime = ("image/svg+xml" if path.endswith(".svg")
                    else "image/webp" if path.endswith(".webp") else "image/png")
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:{mime};base64,{b64}"
    return None


def _brand_mark(size: int = 32) -> str:
    """Logo image if available, otherwise the ✦ fallback glyph."""
    logo = _logo_data_uri()
    if logo:
        return (
            f'<img src="{logo}" alt="Recruiting AI" '
            f'style="width:{size}px;height:{size}px;border-radius:9px;object-fit:contain;">'
        )
    return (
        f'<span style="width:{size}px;height:{size}px;border-radius:9px;display:inline-flex;'
        "align-items:center;justify-content:center;background:linear-gradient(135deg,#7c5cff,#22d3ee);"
        'color:#fff;font-size:16px;">✦</span>'
    )


def render_nav() -> None:
    """Clean top navigation: brand on the left, a single primary CTA on the
    right (one main action, no button clutter)."""
    brand, spacer, cta = st.columns([3, 2, 1.5])
    with brand:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;
                        color:#0f1226;padding-top:6px;">
              __MARK__
              Recruiting&nbsp;AI
              <span style="font-size:.72rem;font-weight:600;color:#7c5cff;background:rgba(124,92,255,.10);
                           padding:3px 9px;border-radius:999px;margin-left:6px;">Multi-Agent HR</span>
            </div>
            """.replace("__MARK__", _brand_mark(32)),
            unsafe_allow_html=True,
        )
    with cta:
        if st.session_state.page == "home":
            if st.button("Workspace öffnen →", type="primary", use_container_width=True, key="nav_cta"):
                st.session_state.page = "workspace"
                st.rerun()
        else:
            if st.button("← Startseite", use_container_width=True, key="nav_home"):
                st.session_state.page = "home"
                st.session_state.selected_candidate = None
                st.rerun()
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(124,92,255,.12);margin:4px 0 10px;'>",
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    """Immersive fullscreen intro gate.

    On first load this is the ONLY thing rendered — no navbar, no header,
    no page content. Clicking the CTA, scrolling, or pressing a key slides
    the hero smoothly upward and reloads the app with `?started=1`, which
    reveals the actual website (navbar + sections).
    """
    # Page-specific CSS: hide all Streamlit chrome and remove padding so the
    # hero truly fills the viewport with nothing below the fold.
    st.markdown(
        """
        <style>
          header[data-testid="stHeader"]{display:none !important;}
          [data-testid="stToolbar"]{display:none !important;}
          [data-testid="stDecoration"]{display:none !important;}
          .block-container{padding:0 !important; max-width:100% !important;}
          [data-testid="stAppViewContainer"]{overflow:hidden !important;}
          footer{display:none !important;}
          /* Full-bleed, fullscreen hero iframe (only on the intro page) */
          .element-container:has(iframe[title="st.iframe"]),
          .element-container:has(iframe[title="streamlit_components.v1.html.html"]){
            width:100vw !important; margin-left:calc(50% - 50vw) !important;
            margin-right:calc(50% - 50vw) !important; height:100vh !important;
          }
          iframe[title="st.iframe"],
          iframe[title="streamlit_components.v1.html.html"]{
            width:100vw !important; height:100vh !important; border:none !important;
            display:block; background:transparent !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    intro_html = """
    <style>
      html,body{margin:0;padding:0;height:100%;overflow:hidden;background:transparent;
        font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;}
      .hero{
        position:relative; height:100vh; width:100vw; overflow:hidden; cursor:pointer;
        transition:transform .85s cubic-bezier(.76,0,.24,1), opacity .85s ease;
        background:linear-gradient(180deg,#fbfbff 0%, #f1f2fb 100%);
      }
      .hero.lift{ transform:translateY(-100%); opacity:0; }
      /* Spline graphic fills the entire viewport, scaled up to "cover" the
         screen (object-fit: cover behaviour) so the orb + wave dominate and
         no empty white space remains around the scene. */
      spline-viewer{
        position:absolute; top:50%; left:50%;
        width:100vw; height:100vh; display:block;
        /* the orb sits slightly left in the scene's own composition, so nudge
           the canvas right to visually centre it */
        transform:translate(-43%,-50%) scale(1.7);
        transform-origin:center center;
      }
      @media (max-width:900px){
        spline-viewer{ transform:translate(-43%,-50%) scale(2.4); }
      }
      /* transparent full-screen click target -> advance on click anywhere
         (incl. the scene's own "JOIN US NOW") */
      .catcher{position:absolute; inset:0; z-index:3; background:transparent;}
      .scroll{
        position:absolute; bottom:6vh; left:0; right:0; margin:0 auto; width:max-content;
        color:#6b5bd0; font-size:.76rem; letter-spacing:.2em; text-transform:uppercase;
        display:flex; flex-direction:column; align-items:center; gap:10px; cursor:pointer;
        animation:fadeUp 1.1s ease both .4s; z-index:4; pointer-events:none;
      }
      .arrow{ width:18px;height:18px;border-right:2px solid #6b5bd0;
        border-bottom:2px solid #6b5bd0; transform:rotate(45deg);
        animation:bob 1.8s infinite ease-in-out; }
      @keyframes bob{0%,100%{transform:rotate(45deg) translate(0,0);opacity:.4;}
        50%{transform:rotate(45deg) translate(4px,4px);opacity:1;}}
      @keyframes fadeUp{from{opacity:0;transform:translateY(24px);}to{opacity:1;transform:translateY(0);}}
    </style>
    <script type="module"
      src="https://unpkg.com/@splinetool/viewer@1.9.48/build/spline-viewer.js"></script>
    <div class="hero" id="hero">
      <spline-viewer
        url="https://prod.spline.design/Ji0hiX2hb-mU5zX1/scene.splinecode"
        events-target="global"></spline-viewer>
      <div class="catcher" id="catcher"></div>
      <div class="scroll"><span>Scroll to Explore</span><div class="arrow"></div></div>
    </div>
    <script>
      (function(){
        var fired = false;
        function reveal(){
          if(fired) return; fired = true;
          document.getElementById('hero').classList.add('lift');
          setTimeout(function(){
            try{
              var base = window.parent.location.pathname;
              window.parent.location.href = base + '?started=1';
            }catch(e){
              window.location.href = '?started=1';
            }
          }, 780);
        }
        document.getElementById('catcher').addEventListener('click', reveal);
        window.addEventListener('wheel', function(e){ if(e.deltaY > 4) reveal(); }, {passive:true});
        var sy = null;
        window.addEventListener('touchstart', function(e){ sy = e.touches[0].clientY; }, {passive:true});
        window.addEventListener('touchmove', function(e){
          if(sy !== null && (sy - e.touches[0].clientY) > 28) reveal();
        }, {passive:true});
        window.addEventListener('keydown', function(e){
          if(e.key==='ArrowDown' || e.key===' ' || e.key==='Enter' || e.key==='PageDown') reveal();
        });
      })();
    </script>
    """
    components.html(intro_html, height=900, scrolling=False)


# --- Higgsfield brand visuals (generated, hosted on CDN) --------------------
_HF = "https://d8j0ntlcm91z4.cloudfront.net/user_33IJIDZ0cOmwdzXkCP5nbQryjVC/"
IMG_ORB = _HF + "hf_20260618_184056_8486b8b4-b288-4344-996a-53ea801af5bc.png"
IMG_DATA = _HF + "hf_20260618_184103_c3dd220e-bda5-4964-8a2d-0e87b9fd77dd.png"
IMG_WAVE = _HF + "hf_20260618_184108_568fd0ad-6da7-4830-962a-35a70bf8d220.png"
# Step visuals (isometric, 3:4) for the horizontal scroll gallery
IMG_STEP_UPLOAD = _HF + "hf_20260619_105340_e82eb3e8-ca18-4a12-a2a4-0a17fa185ae7.png"
IMG_STEP_ANALYZE = _HF + "hf_20260619_105345_3d63247e-224a-4452-9886-fc6bafea6e9c.png"
IMG_STEP_DECIDE = _HF + "hf_20260619_105350_b7e7816d-3170-4cfb-9f07-be85ce02fce0.png"

# Spline scene used as the visual in the "Das Problem" section
SPLINE_PROBLEM = "https://prod.spline.design/4PF4J4YenOXJHe4A/scene.splinecode"

# Hero background: refined premium brand graphic (matches the page gradient)
IMG_HERO = _HF + "hf_20260620_120052_d9d42f35-b642-41d2-ab9d-5a0eece17eeb.png"


def render_spline_embed(url: str, height: int = 560) -> None:
    """Embed a Spline scene as a large, borderless background graphic
    (transparent — no card, no box) that blends into the section."""
    html = (
        """
        <style>
          html,body{margin:0;padding:0;background:transparent;overflow:hidden;}
          .splinebg{position:relative; width:100%; height:__H__px; background:transparent; overflow:hidden;
            /* soft fade on all edges so the ribbon blends cleanly into the page */
            -webkit-mask-image:
              linear-gradient(180deg, transparent 0%, #000 22%, #000 78%, transparent 100%),
              linear-gradient(90deg, transparent 0%, #000 7%, #000 93%, transparent 100%);
            -webkit-mask-composite:source-in;
            mask-image:
              linear-gradient(180deg, transparent 0%, #000 22%, #000 78%, transparent 100%),
              linear-gradient(90deg, transparent 0%, #000 7%, #000 93%, transparent 100%);
            mask-composite:intersect;}
          /* scale the scene up so the ribbon spans edge-to-edge with no side gap */
          spline-viewer{position:absolute; top:50%; left:50%;
            width:100%; height:100%;
            transform:translate(-50%,-50%) scale(1.1); transform-origin:center center;}
        </style>
        <script type="module"
          src="https://unpkg.com/@splinetool/viewer@1.9.48/build/spline-viewer.js"></script>
        <div class="splinebg">
          <spline-viewer url="__URL__" events-target="global"></spline-viewer>
        </div>
        """
    ).replace("__H__", str(height)).replace("__URL__", url)
    components.html(html, height=height + 4, scrolling=False)


def render_scroll_reveal() -> None:
    """Wire an IntersectionObserver onto the parent document's `.reveal`
    elements so sections fade/slide in smoothly as they enter the viewport.

    Runs inside a 0-height component iframe (same-origin), with a safety
    timeout that force-reveals everything so content is never stuck hidden.
    """
    components.html(
        """
        <script>
        (function(){
          function run(){
            try{
              var doc = window.parent.document;
              var els = doc.querySelectorAll('.reveal:not(.reveal-bound)');
              if(!els.length){ setTimeout(run, 150); return; }
              els.forEach(function(el){ el.classList.add('reveal-bound'); });
              if('IntersectionObserver' in window.parent){
                var io = new window.parent.IntersectionObserver(function(entries){
                  entries.forEach(function(e){
                    if(e.isIntersecting){ e.target.classList.add('reveal-in'); io.unobserve(e.target); }
                  });
                }, {threshold:0.10, rootMargin:'0px 0px -7% 0px'});
                els.forEach(function(el){ io.observe(el); });
                setTimeout(function(){ els.forEach(function(el){ el.classList.add('reveal-in'); }); }, 2800);
              } else {
                els.forEach(function(el){ el.classList.add('reveal-in'); });
              }
            }catch(err){}
          }
          setTimeout(run, 60);
        })();

        /* Force muted autoplay of background videos. Streamlit-injected
           <video> tags don't always start on their own (esp. Safari), so we
           set the muted property and call play() from here. */
        (function(){
          function play(){
            try{
              var vids = window.parent.document.querySelectorAll('video');
              vids.forEach(function(v){
                v.muted = true; v.defaultMuted = true; v.playsInline = true;
                v.setAttribute('muted',''); v.setAttribute('playsinline','');
                var pr = v.play(); if(pr && pr.catch){ pr.catch(function(){}); }
                if(!v.dataset.boundplay){
                  v.dataset.boundplay = '1';
                  ['canplay','loadeddata'].forEach(function(ev){
                    v.addEventListener(ev, function(){ var p=v.play(); if(p&&p.catch) p.catch(function(){}); });
                  });
                }
              });
            }catch(e){}
          }
          [120, 700, 1800].forEach(function(t){ setTimeout(play, t); });
        })();
        </script>
        """,
        height=0,
    )


def _main_cta(label: str, key: str) -> None:
    """The single, centered primary call-to-action -> opens the workspace."""
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button(label, type="primary", use_container_width=True, key=key):
            st.session_state.page = "workspace"
            st.rerun()


def _step(num: str, title: str, body: str) -> str:
    return (
        f'<div class="stepcard"><div class="k">SCHRITT {num}</div>'
        f"<h3>{title}</h3><p>{body}</p></div>"
    )


def render_home() -> None:
    """Premium scroll-storytelling landing page (revealed after the intro):
    Problem -> Lösung -> Demo -> Ergebnisse -> Pricing, with one main CTA."""

    # 1) Hero with a bright pastel brand graphic + one clear message + main CTA
    st.markdown(
        '<div class="vhero reveal">'
        f'<img class="vhero-bg" src="{IMG_HERO}" alt="">'
        '<div class="vhero-scrim"></div>'
        '<div class="vhero-copy">'
        '<div class="kicker">Recruiting AI · Multi-Agent HR</div>'
        '<h1 class="display">Bewerbungen verstehen.<br><span class="g">Menschen entscheiden.</span></h1>'
        '<p class="big">Sechs spezialisierte KI-Agenten lesen Stellenprofile und Lebensläufe, '
        "prüfen Anforderungen und decken Informationslücken auf — die Entscheidung bleibt bei dir.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    _main_cta("🚀  Demo starten", "cta_top")

    # 2) Problem — text first, the full-bleed Spline ribbon sits below it
    st.markdown(
        '<div class="story reveal" style="padding-bottom:0;">'
        '<div class="kicker">Das Problem</div>'
        '<h2 class="display">5–10 Stunden<br><span class="g">pro Stelle.</span></h2>'
        '<p class="big">Ohne eigene HR-Abteilung wird jede Ausschreibung zur Belastung: '
        "unterschiedliche CV-Formate, fehlende Angaben, keine nachvollziehbare Bewertung.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    render_spline_embed(SPLINE_PROBLEM, height=420)
    st.markdown(
        '<div class="story reveal" style="padding-top:0;"><div class="stats">'
        '<div class="stat"><div class="n">5–10 h</div><div class="l">Sichtungsaufwand pro Stelle</div></div>'
        '<div class="stat"><div class="n">100 %</div><div class="l">der Schritte im Audit Log</div></div>'
        '<div class="stat"><div class="n">0</div><div class="l">automatische Entscheidungen</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # 3) Lösung — visual main idea (the orb) + clean steps
    st.markdown(
        '<div class="story reveal"><div class="split rev">'
        f'<div class="media-wrap"><img src="{IMG_ORB}" alt="Recruiting AI"></div>'
        "<div>"
        '<div class="kicker">Die Lösung</div>'
        '<h2 class="display">Ein Agenten-<br><span class="g">team</span>, ein Flow.</h2>'
        '<p class="big">Vom Stellenprofil bis zur Rückfrage: jeder Agent hat genau eine Aufgabe — '
        "transparent, überprüfbar, nachvollziehbar.</p>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="story reveal" style="padding-top:24px;"><div class="steps">'
        + _step("01", "Stelle & Bewerbungen", "Stellenprofil einfügen, Lebensläufe als PDF oder Text hochladen.")
        + _step("02", "Analyse & Abgleich", "Der CV-Agent strukturiert die Daten, der Matching-Agent prüft jede Anforderung.")
        + _step("03", "Lücken & Rückfragen", "Fehlende Angaben werden sichtbar — höfliche Rückfragen entstehen automatisch.")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # 3b) Horizontal scroll gallery (Raycast-style) with Higgsfield visuals
    st.markdown(
        '<div class="story reveal" style="padding-bottom:8px;">'
        '<div class="kicker">Im Detail</div>'
        '<h2 class="display">Drei Schritte. <span class="g">Ein Flow.</span></h2>'
        '<p class="big">Scrolle horizontal durch den Ablauf.</p></div>',
        unsafe_allow_html=True,
    )

    def _hcard(img: str, k: str, title: str, body: str) -> str:
        return (
            f'<div class="hcard"><img src="{img}" alt="{title}" loading="lazy">'
            f'<div class="cap"><div class="k">{k}</div><h4>{title}</h4><p>{body}</p></div></div>'
        )

    st.markdown(
        '<div class="bleed reveal"><div class="hscroll">'
        + _hcard(IMG_STEP_UPLOAD, "SCHRITT 01", "Hochladen", "Stelle & Lebensläufe rein — als PDF oder Text.")
        + _hcard(IMG_STEP_ANALYZE, "SCHRITT 02", "Analysieren", "CV-Agent strukturiert, Matching-Agent prüft jede Anforderung.")
        + _hcard(IMG_STEP_DECIDE, "SCHRITT 03", "Entscheiden", "Der Mensch wählt — nie der Algorithmus.")
        + _hcard(IMG_ORB, "SYSTEM", "Multi-Agent", "Sechs Agenten, ein sauberer, nachvollziehbarer Flow.")
        + _hcard(IMG_WAVE, "TRANSPARENZ", "Audit Log", "Jeder Verarbeitungsschritt lückenlos dokumentiert.")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # 4) Demo — invite into the workspace (single CTA)
    st.markdown(
        '<div class="story reveal" style="text-align:center;">'
        '<div class="kicker" style="text-align:center;">Live-Demo</div>'
        '<h2 class="display">Probier es im Workspace.</h2>'
        '<p class="big" style="margin-left:auto;margin-right:auto;">Lade Demo-Daten oder deine eigenen '
        "Bewerbungen — analysieren, Kandidatenprofile öffnen, Rückfragen als E-Mail-Entwurf erstellen.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _main_cta("Workspace öffnen  →", "cta_demo")

    # 5) Ergebnisse — outcomes + wave visual
    st.markdown(
        '<div class="story reveal"><div class="split">'
        "<div>"
        '<div class="kicker">Ergebnisse</div>'
        '<h2 class="display">Struktur statt<br><span class="g">Bauchgefühl.</span></h2>'
        '<p class="big">Saubere Kandidatenprofile, eine transparente Qualifikationscheckliste und ein '
        "lückenloses Audit Log — Human-in-the-Loop by Design. Keine Rangfolge, keine Empfehlung.</p>"
        "</div>"
        f'<div class="media-wrap"><img src="{IMG_DATA}" alt="Strukturierte Kandidatendaten"></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # 6) Pricing — simple, one highlighted plan
    st.markdown(
        '<div class="story reveal" style="text-align:center;">'
        '<div class="kicker" style="text-align:center;">Preise</div>'
        '<h2 class="display">Einfach. Planbar.</h2></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="story reveal" style="padding-top:16px;"><div class="pricing">'
        '<div class="price"><div class="pn">Starter</div>'
        '<div class="pp">0€<small> / Demo</small></div>'
        "<ul><li>Vollständiger Workspace</li><li>Bis zu 5 Bewerbungen</li><li>Audit Log</li></ul></div>"
        '<div class="price feat"><div class="badge-pop">Beliebt</div><div class="pn">Team</div>'
        '<div class="pp">49€<small> / Monat</small></div>'
        "<ul><li>Unbegrenzte Bewerbungen</li><li>Alle sechs Agenten</li><li>Rückfragen-E-Mails</li>"
        "<li>Priorisierter Support</li></ul></div>"
        '<div class="price"><div class="pn">Enterprise</div>'
        '<div class="pp">Auf Anfrage</div>'
        "<ul><li>SSO &amp; Rollen</li><li>Eigene Modelle / API</li><li>Audit-Export</li></ul></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # Final CTA band (one button)
    st.markdown(
        '<div class="cta-band reveal">'
        '<h2 class="display" style="font-size:clamp(2rem,4vw,3rem);">Bereit für den ersten Screening-Lauf?</h2>'
        '<p class="big" style="margin:14px auto 0;">Starte in Sekunden — ganz ohne Setup.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    _main_cta("🚀  Demo starten", "cta_final")
    st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)

    render_scroll_reveal()


# --- Workspace helpers ------------------------------------------------------

STATUS_META = {
    "erfuellt": ("✓ erfüllt", "pill-ok"),
    "teilweise": ("◐ teilweise", "pill-part"),
    "offen": ("? offen", "pill-open"),
    "nicht_erfuellt": ("✗ nicht erfüllt", "pill-no"),
}


def render_workspace() -> None:
    st.markdown(
        '<div class="section reveal" style="margin-top:18px;">'
        '<span class="eyebrow">Recruiting Workspace</span>'
        "<h2>Dein Screening-Arbeitsbereich</h2>"
        '<p class="lead">Stellenprofil definieren, Bewerbungen analysieren und Ergebnisse '
        "transparent prüfen — die Entscheidung bleibt bei dir.</p></div>",
        unsafe_allow_html=True,
    )

    # Quick action: load demo data
    a1, a2 = st.columns([1, 3])
    with a1:
        if st.button("✨ Demo-Daten laden", type="primary", use_container_width=True):
            load_demo_data()
            st.success("Demo-Stellenprofil und 3 Bewerbungen geladen & analysiert.")

    tabs = st.tabs(
        [
            "① Stellenprofil",
            "② Bewerbungen",
            "③ Kandidaten",
            "④ Informationslücken",
            "⑤ Rückfragen",
            "⑥ Audit Log",
        ]
    )

    # --- Tab 1: Stellenprofil ---
    with tabs[0]:
        st.subheader("Stellenprofil-Agent")
        st.caption("Füge die Stellenbeschreibung ein. Der Agent extrahiert messbare Anforderungen.")
        st.session_state.job_text = st.text_area(
            "Stellenbeschreibung",
            value=st.session_state.job_text,
            height=260,
            placeholder="z. B. Senior Backend Entwickler – 5 Jahre Erfahrung, Python, SQL, AWS …",
        )
        if st.button("🎯 Stellenprofil analysieren", type="primary"):
            if st.session_state.job_text.strip():
                st.session_state.requirements = stellenprofil_agent(st.session_state.job_text)
                audit(
                    "Stellenprofil-Agent",
                    "Stellenprofil analysiert",
                    f"{len(st.session_state.requirements)} Anforderung(en) erkannt.",
                )
                st.session_state.analysis_done = False
                st.success(f"{len(st.session_state.requirements)} Anforderung(en) erkannt.")
            else:
                st.warning("Bitte zuerst eine Stellenbeschreibung einfügen.")

        if st.session_state.requirements:
            st.markdown("##### Erkannte Anforderungen")
            for req in st.session_state.requirements:
                imp = "Muss" if req.importance == "muss" else "Kann"
                imp_color = "#dc2626" if req.importance == "muss" else "#b4790c"
                st.markdown(
                    f'<div class="cand" style="margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">'
                    f'<span><b>{req.label}</b> <span style="color:#5b6178;font-size:.85rem;">· {req.kind}</span></span>'
                    f'<span class="pill" style="background:{imp_color}1a;color:{imp_color};">{imp}</span></div>',
                    unsafe_allow_html=True,
                )

    # --- Tab 2: Bewerbungen ---
    with tabs[1]:
        st.subheader("CV-Agent")
        st.caption("Lade Lebensläufe als PDF oder TXT hoch — oder füge Text manuell ein.")
        if not HAS_PYPDF:
            st.info("Hinweis: `pypdf` ist nicht installiert — PDF-Text kann nicht gelesen werden. TXT-Upload und manuelle Eingabe funktionieren.")

        uploaded = st.file_uploader(
            "Bewerbungen hochladen",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        manual = st.text_area(
            "…oder einen Lebenslauf als Text einfügen",
            height=140,
            placeholder="Name: …\nBerufserfahrung: …\nSkills: …",
        )

        if st.button("📑 Bewerbungen analysieren", type="primary"):
            new_candidates = []
            idx = len(st.session_state.candidates)
            if uploaded:
                for f in uploaded:
                    text = extract_text_from_upload(f)
                    base_name = f.name.rsplit(".", 1)[0]
                    new_candidates.append(cv_agent(base_name, f.name, text, idx))
                    idx += 1
            if manual.strip():
                new_candidates.append(cv_agent(f"Kandidat {idx + 1}", "Manuelle Eingabe", manual, idx))
                idx += 1

            if new_candidates:
                st.session_state.candidates.extend(new_candidates)
                audit("CV-Agent", "Bewerbungen analysiert", f"{len(new_candidates)} Lebenslauf/Lebensläufe verarbeitet.")
                st.session_state.analysis_done = False
                st.success(f"{len(new_candidates)} Bewerbung(en) analysiert. Gesamt: {len(st.session_state.candidates)}.")
            else:
                st.warning("Bitte mindestens eine Datei hochladen oder Text einfügen.")

        if st.session_state.candidates:
            st.markdown(f"##### {len(st.session_state.candidates)} Bewerbung(en) im Workspace")
            for c in st.session_state.candidates:
                st.markdown(
                    f'<div class="cand" style="margin-bottom:10px;">'
                    f'<span class="nm">{c.name}</span> '
                    f'<span class="meta">· {c.source}</span><br>'
                    f'<span class="meta">{len(c.skills)} Skills · '
                    f'{c.years if c.years is not None else "?"} Jahre · {c.education_label}</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑️ Bewerbungen zurücksetzen"):
                st.session_state.candidates = []
                st.session_state.analysis_done = False
                audit("Workspace", "Bewerbungen zurückgesetzt", "Alle Kandidatendaten entfernt.")
                st.rerun()

    # --- Tab 3: Kandidaten (overview + detail) ---
    with tabs[2]:
        st.subheader("Kandidatenübersicht")
        if not st.session_state.requirements:
            st.info("Bitte zuerst ein Stellenprofil analysieren (Tab ①).")
        elif not st.session_state.candidates:
            st.info("Bitte zuerst Bewerbungen hochladen (Tab ②).")
        else:
            if st.button("▶️ Anforderungsabgleich starten", type="primary"):
                run_full_analysis()
                st.success("Matching, Informationslücken und Rückfragen wurden erstellt.")

            st.markdown(
                '<div class="hitl" style="margin:8px 0 18px;padding:14px 20px;">'
                "⚖️ <b>Keine automatische Rangfolge.</b> Die Reihenfolge entspricht dem Upload. "
                "Es gibt keine Bewertung und keine Empfehlung — die Entscheidung triffst du."
                "</div>",
                unsafe_allow_html=True,
            )

            if st.session_state.analysis_done:
                cols = st.columns(2)
                for i, cand in enumerate(st.session_state.candidates):
                    met = sum(1 for m in cand.matches if m.status == "erfuellt")
                    total = len(cand.matches)
                    skills_html = "".join(f'<span class="tag">{s}</span>' for s in cand.skills[:6]) or '<span class="meta">keine erkannt</span>'
                    with cols[i % 2]:
                        st.markdown(
                            f'<div class="cand" style="margin-bottom:14px;">'
                            f'<span class="nm">{cand.name}</span>'
                            f'<div class="meta">{met}/{total} Anforderungen erfüllt · '
                            f'{len(cand.gaps)} Informationslücke(n)</div>'
                            f'<div style="margin-top:10px;">{skills_html}</div></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(f"Profil ansehen — {cand.name}", key=f"sel_{cand.id}", use_container_width=True):
                            st.session_state.selected_candidate = cand.id
                            st.session_state.page = "candidate"
                            st.rerun()
            else:
                st.caption("Noch kein Abgleich durchgeführt — klicke auf „Anforderungsabgleich starten“.")

    # --- Tab 4: Informationslücken ---
    with tabs[3]:
        st.subheader("Informationslücken-Agent")
        if not st.session_state.analysis_done:
            st.info("Bitte zuerst den Anforderungsabgleich starten (Tab ③).")
        else:
            any_gap = False
            for cand in st.session_state.candidates:
                if cand.gaps:
                    any_gap = True
                    st.markdown(f"**{cand.name}** — {len(cand.gaps)} Lücke(n)")
                    for g in cand.gaps:
                        st.markdown(f"- **{g.requirement_label}** — {g.description}")
                    st.markdown("")
            if not any_gap:
                st.success("Keine Informationslücken erkannt.")

    # --- Tab 5: Rückfragen ---
    with tabs[4]:
        st.subheader("Rückfragen-Agent")
        st.caption("Wähle pro Bewerber:in die gewünschten Rückfragen aus und generiere daraus "
                   "einen E-Mail-Entwurf. Die E-Mail wird **nicht** automatisch versendet.")
        if not st.session_state.analysis_done:
            st.info("Bitte zuerst den Anforderungsabgleich starten (Tab ③).")
        else:
            any_q = False
            for cand in st.session_state.candidates:
                if not cand.questions:
                    continue
                any_q = True
                with st.expander(f"💬  {cand.name}  ·  {len(cand.questions)} mögliche Rückfrage(n)", expanded=False):
                    st.markdown("**Rückfragen auswählen:**")
                    selected = []
                    for idx, q in enumerate(cand.questions):
                        if st.checkbox(q, key=f"q_{cand.id}_{idx}"):
                            selected.append(q)

                    if st.button("✉️ Rückfragen-E-Mail generieren", key=f"genmail_{cand.id}", type="primary"):
                        if selected:
                            to, subject, body = generate_email(cand, selected)
                            ver = st.session_state.email_drafts.get(cand.id, {}).get("ver", 0) + 1
                            st.session_state.email_drafts[cand.id] = {
                                "to": to, "subject": subject, "body": body, "ver": ver,
                            }
                            audit(
                                "Rückfragen-Agent",
                                "E-Mail-Entwurf erstellt",
                                f"{len(selected)} Rückfrage(n) für {cand.name} zusammengestellt (nicht versendet).",
                            )
                        else:
                            st.warning("Bitte mindestens eine Rückfrage auswählen.")

                    # --- E-Mail-Fenster (Entwurf, wird NICHT gesendet) ---
                    draft = st.session_state.email_drafts.get(cand.id)
                    if draft:
                        ver = draft["ver"]
                        st.markdown(
                            '<div class="cand" style="margin-top:14px;border:1px solid rgba(124,92,255,.25);">'
                            '<div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#0f1226;">'
                            '✉️ E-Mail-Entwurf <span class="pill pill-part">Entwurf · nicht gesendet</span></div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        to = st.text_input("An", value=draft["to"], key=f"emailto_{cand.id}_{ver}")
                        subject = st.text_input("Betreff", value=draft["subject"], key=f"emailsub_{cand.id}_{ver}")
                        body = st.text_area("Nachricht", value=draft["body"], height=320, key=f"emailbody_{cand.id}_{ver}")

                        mailto = (
                            "mailto:" + urllib.parse.quote(to)
                            + "?subject=" + urllib.parse.quote(subject)
                            + "&body=" + urllib.parse.quote(body)
                        )
                        st.markdown(
                            f'<a href="{mailto}" target="_blank" style="display:inline-block;'
                            "text-decoration:none;background:linear-gradient(135deg,#7c5cff,#3b82f6);"
                            "color:#fff;font-weight:600;padding:.6rem 1.2rem;border-radius:14px;"
                            'box-shadow:0 12px 30px rgba(124,92,255,.40);">📧 In E-Mail-Programm öffnen</a>'
                            '<span class="meta" style="margin-left:12px;">Öffnet deinen Mail-Client mit '
                            "vorausgefülltem Entwurf — Versand bestätigst du selbst.</span>",
                            unsafe_allow_html=True,
                        )
                        if st.button("🗑️ Entwurf verwerfen", key=f"deldraft_{cand.id}"):
                            st.session_state.email_drafts.pop(cand.id, None)
                            st.rerun()

            if not any_q:
                st.success("Keine Rückfragen nötig — alle Angaben vollständig.")

    # --- Tab 6: Audit Log ---
    with tabs[5]:
        st.subheader("Audit Log")
        st.caption("Jeder Verarbeitungsschritt wird lückenlos dokumentiert.")
        if not st.session_state.audit:
            st.info("Noch keine Aktivität protokolliert.")
        else:
            for entry in reversed(st.session_state.audit):
                st.markdown(
                    f'<div class="cand" style="margin-bottom:8px;">'
                    f'<span class="meta">{entry.timestamp}</span> · '
                    f'<b style="color:#7c5cff;">{entry.agent}</b> — {entry.action}'
                    f'<br><span class="meta">{entry.detail}</span></div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑️ Audit Log leeren"):
                st.session_state.audit = []
                st.rerun()


def render_candidate_detail() -> None:
    """Dedicated full-page candidate profile (opened via 'Profil ansehen')."""
    cand = next(
        (c for c in st.session_state.candidates if c.id == st.session_state.selected_candidate),
        None,
    )

    if st.button("← Zurück zur Kandidatenübersicht"):
        st.session_state.page = "workspace"
        st.rerun()

    if not cand:
        st.warning("Kandidat:in nicht gefunden.")
        return

    met = sum(1 for m in cand.matches if m.status == "erfuellt")
    total = len(cand.matches)
    st.markdown(
        f'<div class="section" style="margin-top:14px;">'
        f'<span class="eyebrow">Kandidatenprofil</span>'
        f"<h2>👤 {cand.name}</h2>"
        f'<p class="lead">Quelle: {cand.source} · Kontakt: {cand.email or "nicht gefunden"} · '
        f'{met}/{total} Anforderungen erfüllt · {len(cand.gaps)} Informationslücke(n)</p></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skills erkannt", len(cand.skills))
    m2.metric("Berufserfahrung", f"{cand.years} J." if cand.years is not None else "?")
    m3.metric("Abschluss", cand.education_label)
    m4.metric("Sprachen", len(cand.languages))

    if cand.skills:
        skills_html = "".join(f'<span class="tag">{s}</span>' for s in cand.skills)
        st.markdown(f'<div style="margin:10px 0 4px;">{skills_html}</div>', unsafe_allow_html=True)

    st.markdown("#### Qualifikationscheckliste")
    if cand.matches:
        for m in cand.matches:
            label_txt, pill = STATUS_META.get(m.status, ("?", "pill-open"))
            imp = "Muss" if m.importance == "muss" else "Kann"
            st.markdown(
                f'<div class="cand" style="margin-bottom:8px;display:flex;'
                f'justify-content:space-between;align-items:center;">'
                f'<span><b>{m.label}</b> <span class="meta">· {imp} · {m.note}</span></span>'
                f'<span class="pill {pill}">{label_txt}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Noch kein Abgleich für diese:n Kandidat:in durchgeführt.")

    if cand.gaps:
        st.markdown("#### 🧩 Informationslücken")
        for g in cand.gaps:
            st.markdown(f"- **{g.requirement_label}** — {g.description}")
    if cand.questions:
        st.markdown("#### 💬 Vorgeschlagene Rückfragen")
        for q in cand.questions:
            st.markdown(f"- {q}")
        st.caption("Rückfragen auswählen und als E-Mail-Entwurf generieren kannst du im Tab ⑤ Rückfragen.")

    st.markdown(
        '<div class="hitl" style="margin-top:18px;padding:14px 20px;">'
        "⚖️ <b>Human-in-the-Loop:</b> Diese Ansicht zeigt nur extrahierte Daten — "
        "keine Bewertung, keine Empfehlung. Die Entscheidung triffst du."
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Recruiting AI — Multi-Agent HR",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    init_state()

    # --- Intro gate -------------------------------------------------------
    # On first load show ONLY the immersive fullscreen hero. The intro is
    # dismissed via the CTA / scroll (which sets ?started=1) or once the user
    # has already entered the app in this session.
    started = ("started" in st.query_params) or st.session_state.intro_done
    if not started:
        render_intro()
        return
    st.session_state.intro_done = True

    # --- Actual website (revealed after the intro) ------------------------
    render_nav()
    if st.session_state.page == "candidate":
        render_candidate_detail()
    elif st.session_state.page == "workspace":
        render_workspace()
    else:
        render_home()


if __name__ == "__main__":
    main()
