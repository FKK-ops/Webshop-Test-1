# HR-Recruiting-Assistent (Streamlit-Prototyp)

Einfacher Prototyp, der PDF-Lebensläufe einliest, **ausschließlich objektive
Informationen** per LLM extrahiert und als filterbare Tabelle anzeigt.

> Das Tool liefert bewusst **kein Ranking, keinen Score, keine Top-5 und keine
> Empfehlung**. Die finale Entscheidung trifft immer ein Mensch.

## Features

- Mehrere PDF-Lebensläufe gleichzeitig hochladen
- Textextraktion via `pdfplumber`
- LLM-Extraktion (OpenAI, JSON-Schema-validiert) von:
  - Name, Skills, Berufserfahrung, Ausbildung, Zertifikate, Sprachkenntnisse
- Tabellarische Darstellung
- Filter nach Skills, Sprachen, Zertifikaten, Berufserfahrung
- Feedback/Korrekturen pro Kandidat (z. B. „SQL wurde übersehen“)
- Feedback wird in `data/feedback.json` gespeichert und beim **nächsten**
  Extraktionsprompt automatisch berücksichtigt

## Ordnerstruktur

```
.
├── app.py                 # Streamlit-UI
├── requirements.txt
├── .env.example
├── data/
│   └── feedback.json      # wird zur Laufzeit erzeugt
└── src/
    ├── pdf_utils.py       # PDF -> Text
    ├── llm_client.py      # OpenAI-Aufruf mit JSON-Schema
    ├── prompts.py         # System- und Extraktionsprompt
    └── feedback_store.py  # Persistenz der Feedback-Notizen
```

## Setup

1. Python 3.10+ vorausgesetzt.

2. Abhängigkeiten installieren:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. `.env` anlegen:

   ```bash
   cp .env.example .env
   # OPENAI_API_KEY eintragen, optional OPENAI_MODEL anpassen
   ```

   Standardmodell: `gpt-4o-mini`. Für höhere Genauigkeit z. B. `gpt-4o`.

## Starten

```bash
streamlit run app.py
```

Die App ist dann unter <http://localhost:8501> erreichbar.

## Bedienung

1. Links PDFs hochladen → **Extraktion starten**.
2. Ergebnisse in der Tabelle ansehen, oben filtern.
3. Einzelnen Kandidaten auswählen → JSON-Details prüfen.
4. Bei Bedarf Korrektur eintragen (z. B. *„SQL wurde übersehen“*).
5. Beim nächsten Klick auf **Extraktion starten** fließt das Feedback in
   den Prompt ein.

## Hinweise

- Es findet **kein Ranking** und **keine Eignungsbewertung** statt.
- Das LLM ist angewiesen, ausschließlich Informationen zu extrahieren,
  die wörtlich oder eindeutig im Lebenslauf stehen.
- Die finale Auswahlentscheidung bleibt vollständig beim Menschen.
