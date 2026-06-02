from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.feedback_store import (
    add_feedback,
    format_feedback_for_prompt,
    load_feedback,
)
from src.llm_client import extract_cv
from src.pdf_utils import extract_text_from_pdf

load_dotenv()

st.set_page_config(page_title="HR-Recruiting-Assistent", layout="wide")

DISCLAIMER = (
    "Hinweis: Dieses Tool extrahiert ausschließlich objektive Informationen "
    "aus Lebensläufen. Es liefert **kein Ranking, keinen Score und keine "
    "Empfehlung**. Die finale Entscheidung trifft immer ein Mensch."
)


def _candidates_to_dataframe(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        data = c["data"]
        rows.append(
            {
                "Datei": c["filename"],
                "Name": data.get("name", ""),
                "Skills": ", ".join(data.get("skills", [])),
                "Berufserfahrung": " | ".join(
                    f"{e.get('role', '')} @ {e.get('company', '')} ({e.get('period', '')})".strip()
                    for e in data.get("experience", [])
                ),
                "Ausbildung": " | ".join(
                    f"{e.get('degree', '')}, {e.get('institution', '')} ({e.get('period', '')})".strip()
                    for e in data.get("education", [])
                ),
                "Zertifikate": ", ".join(data.get("certificates", [])),
                "Sprachen": ", ".join(
                    f"{lang.get('language', '')} ({lang.get('level', '')})".strip()
                    for lang in data.get("languages", [])
                ),
            }
        )
    return pd.DataFrame(rows)


def _matches(haystack: str, needles: list[str]) -> bool:
    """Case-insensitive UND-Match aller Suchbegriffe in haystack."""
    haystack_low = haystack.lower()
    return all(n.strip().lower() in haystack_low for n in needles if n.strip())


def _filter_candidates(
    candidates: list[dict],
    skill_q: str,
    language_q: str,
    cert_q: str,
    experience_q: str,
) -> list[dict]:
    skill_terms = [t for t in skill_q.split(",") if t.strip()]
    lang_terms = [t for t in language_q.split(",") if t.strip()]
    cert_terms = [t for t in cert_q.split(",") if t.strip()]
    exp_terms = [t for t in experience_q.split(",") if t.strip()]

    result = []
    for c in candidates:
        data = c["data"]
        skills_str = " ".join(data.get("skills", []))
        langs_str = " ".join(
            f"{l.get('language', '')} {l.get('level', '')}"
            for l in data.get("languages", [])
        )
        certs_str = " ".join(data.get("certificates", []))
        exp_str = " ".join(
            f"{e.get('role', '')} {e.get('company', '')} {e.get('description', '')}"
            for e in data.get("experience", [])
        )

        if skill_terms and not _matches(skills_str, skill_terms):
            continue
        if lang_terms and not _matches(langs_str, lang_terms):
            continue
        if cert_terms and not _matches(certs_str, cert_terms):
            continue
        if exp_terms and not _matches(exp_str, exp_terms):
            continue
        result.append(c)
    return result


# ---------- Session State ----------

if "candidates" not in st.session_state:
    st.session_state.candidates = []  # list[{"filename": str, "data": dict}]
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# ---------- UI ----------

st.title("HR-Recruiting-Assistent")
st.info(DISCLAIMER)

with st.sidebar:
    st.header("Lebensläufe hochladen")
    uploaded = st.file_uploader(
        "PDF-Dateien auswählen",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Extraktion starten", type="primary", disabled=not uploaded):
        feedback_block = format_feedback_for_prompt(load_feedback())
        progress = st.progress(0.0)
        new_files = [f for f in uploaded if f.name not in st.session_state.processed_files]

        if not new_files:
            st.warning("Alle ausgewählten Dateien wurden bereits verarbeitet.")
        else:
            for i, f in enumerate(new_files, start=1):
                with st.spinner(f"Verarbeite {f.name} ..."):
                    try:
                        text = extract_text_from_pdf(f.read())
                        if not text:
                            st.warning(f"{f.name}: kein Text extrahierbar.")
                            continue
                        data = extract_cv(text, feedback_block)
                        st.session_state.candidates.append(
                            {"filename": f.name, "data": data}
                        )
                        st.session_state.processed_files.add(f.name)
                    except Exception as e:
                        st.error(f"Fehler bei {f.name}: {e}")
                progress.progress(i / len(new_files))
            st.success(f"{len(new_files)} Lebensläufe extrahiert.")

    if st.session_state.candidates:
        if st.button("Alle Kandidaten löschen"):
            st.session_state.candidates = []
            st.session_state.processed_files = set()
            st.rerun()

# ---------- Ergebnisse ----------

if not st.session_state.candidates:
    st.write("Lade links Lebensläufe hoch, um zu starten.")
    st.stop()

st.subheader("Filter")
col1, col2, col3, col4 = st.columns(4)
with col1:
    skill_q = st.text_input("Skills (Komma-getrennt)", "")
with col2:
    language_q = st.text_input("Sprachen", "")
with col3:
    cert_q = st.text_input("Zertifikate", "")
with col4:
    experience_q = st.text_input("Berufserfahrung", "")

filtered = _filter_candidates(
    st.session_state.candidates, skill_q, language_q, cert_q, experience_q
)

st.subheader(f"Kandidaten ({len(filtered)} von {len(st.session_state.candidates)})")
df = _candidates_to_dataframe(filtered)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Details + Feedback ----------

st.subheader("Details & Feedback")
options = [f"{c['data'].get('name') or '(ohne Name)'} – {c['filename']}" for c in filtered]
if not options:
    st.write("Keine Kandidaten entsprechen den Filtern.")
else:
    idx = st.selectbox(
        "Kandidat auswählen", range(len(options)), format_func=lambda i: options[i]
    )
    selected = filtered[idx]
    st.json(selected["data"])

    with st.form("feedback_form", clear_on_submit=True):
        st.markdown(
            "**Korrektur/Feedback eintragen** (z. B. _„SQL wurde übersehen“_). "
            "Das Feedback wird in `data/feedback.json` gespeichert und beim "
            "nächsten Extraktionslauf in den Prompt einbezogen."
        )
        note = st.text_area("Anmerkung", placeholder="Was wurde übersehen oder falsch extrahiert?")
        submitted = st.form_submit_button("Feedback speichern")
        if submitted:
            add_feedback(selected["data"].get("name", ""), note)
            st.success("Feedback gespeichert. Es wird beim nächsten Extraktionslauf berücksichtigt.")

with st.expander("Bisheriges Feedback"):
    entries = load_feedback()
    if not entries:
        st.write("Noch kein Feedback vorhanden.")
    else:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)

st.caption(DISCLAIMER)
