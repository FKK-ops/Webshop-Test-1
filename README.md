# HR-Recruiting-Assistent (Streamlit-Prototyp)

Einfacher Prototyp, der PDF-Lebensläufe einliest, **ausschließlich objektive
Informationen** über die **Claude API** extrahiert und als neutrale,
filterbare Tabelle anzeigt.

> Der Assistent hat **keine Entscheidungskompetenz**. Er bewertet Bewerber
> nicht, priorisiert nicht und gibt keine Empfehlungen.
> **Kein Ranking, kein Score, keine Top-5, keine Empfehlung** — die finale
> Entscheidung trifft immer ein Mensch.

## Features

- Mehrere PDF-Lebensläufe gleichzeitig hochladen
- Textextraktion via `pdfplumber`
- Extraktion über die **Claude API** (`anthropic`, strukturierte Ausgabe per
  Pydantic-Schema) von:
  - Name, Skills, Berufserfahrung, Ausbildung, Zertifikate, Sprachkenntnisse
- Neutrale tabellarische Darstellung aller Bewerber
- Filter nach Skills, Sprachen, Zertifikaten, Berufserfahrung
- Feedback/Korrekturen pro Bewerber mit Kategorie:
  - „Skill wurde übersehen“
  - „Zertifikat wurde übersehen“
  - „Sprache falsch erkannt“
  - „Sonstiges“
- Feedback wird in `data/feedback.json` gespeichert und beim **nächsten**
  Extraktionsprompt automatisch als Kontext berücksichtigt
- Verständliche Fehlermeldungen bei API-Problemen (fehlender Key, Rate-Limit,
  Netzwerk, API-Fehler)

## Ordnerstruktur

```
.
├── app.py                 # Streamlit-UI
├── requirements.txt
├── .env.example
├── data/
│   └── feedback.json      # wird zur Laufzeit erzeugt
└── src/
    ├── pdf_utils.py       # PDF -> Text via pdfplumber
    ├── llm_client.py      # Claude-API-Aufruf (messages.parse + Pydantic-Schema)
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
   # ANTHROPIC_API_KEY eintragen, optional ANTHROPIC_MODEL anpassen
   ```

   Standardmodell: `claude-opus-4-8`. Für schnellere/günstigere Läufe kann
   `ANTHROPIC_MODEL` z. B. auf `claude-haiku-4-5` oder `claude-sonnet-4-6`
   gesetzt werden.

## Starten

```bash
streamlit run app.py
```

Die App ist dann unter <http://localhost:8501> erreichbar.

## Bedienung

1. Links PDFs hochladen → **Extraktion starten**.
2. Ergebnisse in der Tabelle ansehen, oben filtern.
3. Einzelnen Bewerber auswählen → JSON-Details prüfen.
4. Bei Bedarf Korrektur mit Kategorie eintragen
   (z. B. *„SQL wurde übersehen“*).
5. Beim nächsten Klick auf **Extraktion starten** fließt das gespeicherte
   Feedback in den Prompt ein.

## Hinweise

- Es findet **kein Ranking** und **keine Eignungsbewertung** statt.
- Das Modell ist angewiesen, ausschließlich Informationen zu extrahieren,
  die wörtlich oder eindeutig im Lebenslauf stehen.
- Die finale Auswahlentscheidung bleibt vollständig beim Menschen.
