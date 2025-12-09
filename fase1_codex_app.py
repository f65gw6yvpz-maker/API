import os
import sys
import json
from typing import Tuple, List, Any

import pandas as pd
from openai import OpenAI

# ==============================
# CONFIGURAZIONE OPENAI / CODEX
# ==============================

# Modello di OpenAI da usare.
# Puoi cambiarlo in futuro (es: "gpt-4.1" o altro).
MODEL_NAME = "gpt-5.1-codex-max"


def get_openai_client() -> OpenAI:
    """
    Crea il client OpenAI usando la variabile d'ambiente OPENAI_API_KEY.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non impostata. "
            "Imposta la variabile d'ambiente prima di eseguire lo script."
        )
    return OpenAI(api_key=api_key)


# ==============================
# LOGICA DI GRUPPO / CSV
# ==============================

GROUP_KEYS = ["Numero Oda", "Oggetto Lavori"]

TEXT_COLS = [
    "Descrizione Riga",
    "Descrizione Abb Riga",
    "Note",
    "Note No Cont.",
]


def norm(val: Any) -> str:
    """Normalizza un valore generico in stringa pulita."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def build_group_context(rows: pd.DataFrame) -> str:
    """
    Costruisce una descrizione testuale del gruppo (stesso Numero Oda + Oggetto Lavori)
    da passare al modello OpenAI.

    Per ogni riga, include:
    - Numero Oda
    - Posizione
    - Oggetto Lavori
    - Descrizione Riga
    - Descrizione Abb Riga
    - Note
    - Note No Cont.
    """
    lines: List[str] = []
    for _, r in rows.iterrows():
        numero_oda = norm(r.get("Numero Oda", ""))
        posizione = norm(r.get("Posizione", ""))
        oggetto_lavori = norm(r.get("Oggetto Lavori", ""))
        desc = norm(r.get("Descrizione Riga", ""))
        desc_abb = norm(r.get("Descrizione Abb Riga", ""))
        note = norm(r.get("Note", ""))
        note_nc = norm(r.get("Note No Cont.", ""))

        line = (
            f"Riga: Numero_Oda={numero_oda}; "
            f"Posizione={posizione}; "
            f"Oggetto_Lavori={oggetto_lavori}; "
            f"Descrizione_Riga={desc}; "
            f"Descrizione_Abb_Riga={desc_abb}; "
            f"Note={note}; "
            f"Note_No_Cont={note_nc}"
        )
        lines.append(line)

    return "\n".join(lines)


def call_codex_for_group(
    client: OpenAI,
    group_rows: pd.DataFrame
) -> Tuple[str, int]:
    """
    Chiama il modello OpenAI (Codex / GPT) per ottenere:
    - sintesi_intervento (testo in italiano)
    - livello_confidenza (0–100)

    Il modello riceve:
    - istruzioni di fase 1
    - il contesto del gruppo (tutte le righe)
    e deve restituire SOLO un JSON con:
      {
        "sintesi_intervento": "...",
        "livello_confidenza": 0-100
      }
    """

    gruppo_testo = build_group_context(group_rows)

    system_instructions = """
Sei un assistente AI interno per il reparto Acquisti/Amministrazione del gruppo API.

FASE 1 – ADDESTRAMENTO DESCRIZIONI MANDATI (SENZA CAPEX/OPEX)

COMPITO:
- NON devi decidere CAPEX o OPEX.
- Devi solo capire cosa è stato fatto, su cosa, e perché.
- Devi generare:
  1) una sintesi testuale in italiano, chiara e professionale,
  2) un livello di confidenza da 0 a 100 su quanto hai capito bene il contenuto.

CONTESTO:
Ti verrà fornito un elenco di righe di un ordine di acquisto, con:
- Numero_Oda
- Posizione
- Oggetto_Lavori
- Descrizione_Riga
- Descrizione_Abb_Riga
- Note
- Note_No_Cont

Le righe con lo stesso Numero_Oda e Oggetto_Lavori appartengono allo stesso "mandato" o intervento.

REGOLE PER LA SINTESI_INTERVENTO:
- Scrivi in ITALIANO.
- Usa frasi complete, scorrevoli, professionali.
- Metti in evidenza:
  * il tipo di intervento (es. aggiornamento progetto impianto elettrico, sanatoria iscrizione CIVA, etc.),
  * su cosa si interviene (impianto, quadri elettrici, pompe, stazione di servizio, ecc.),
  * lo scopo principale (es. adeguamento normativo, omologazione, miglioramento sicurezza, sostituzione componenti, ecc.),
  * eventuali attività aggiuntive rilevanti (sopralluoghi, rilievi, caricamento documenti su portale, recupero documentazione).
- NON elencare la manodopera come voce principale.
- Puoi unificare voci simili in una frase più leggibile (come nell'esempio: aggiornamento progetto, sanatoria, relazione asseverata, classificazione zone, sopralluoghi, ecc.).
- NON nominare CAPEX o OPEX.
- Obiettivo: leggendo la Sintesi_intervento devo capire chiaramente
  * che intervento è stato eseguito,
  * su cosa,
  * a che scopo.

REGOLE PER LIVELLO_CONFIDENZA:
- Numero intero 0–100.
- Alto (90–100): descrizioni chiare, contesto completo.
- Medio (70–89): descrizioni buone ma con qualche dubbio.
- Basso (50–69): descrizioni generiche o poco informative.
- Molto basso (<50): informazioni quasi inutili o mancanti.

FORMATO DI RISPOSTA:
Rispondi SOLO in JSON valido, SENZA testo aggiuntivo, con questa struttura:

{
  "sintesi_intervento": "testo in italiano...",
  "livello_confidenza": 87
}
"""

    user_input = (
        "Di seguito trovi le righe di un singolo mandato (stesso Numero_Oda e Oggetto_Lavori).\n"
        "Genera la sintesi_intervento e il livello_confidenza come da istruzioni.\n\n"
        "RIGHE DEL MANDATO:\n"
        f"{gruppo_testo}"
    )

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system_instructions.strip(),
        input=user_input,
        response_format={"type": "json_object"},
    )

    # Il testo di output (JSON) aggregato
    raw_text = response.output_text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # In caso di errore, fallback molto semplice
        sintesi = raw_text.strip()
        livello = 60
        return sintesi, livello

    sintesi = str(data.get("sintesi_intervento", "")).strip()
    livello_raw = data.get("livello_confidenza", 0)
    try:
        livello = int(livello_raw)
    except (TypeError, ValueError):
        livello = 0

    # clamp 0–100
    livello = max(0, min(100, livello))
    return sintesi, livello


def has_textual_content(group_rows: pd.DataFrame) -> bool:
    """Verifica se il gruppo contiene almeno un campo testuale valorizzato."""
    for col in TEXT_COLS:
        if col not in group_rows.columns:
            continue
        normalized = group_rows[col].apply(norm)
        if normalized.str.len().gt(0).any():
            return True
    return False


def process_file(input_path: str, output_path: str, client: OpenAI) -> None:
    """
    Legge il CSV input_path (sep=';'), chiama Codex per ogni gruppo
    (Numero Oda + Oggetto Lavori) e scrive output_path con due colonne in più:
    - Livello_confidenza
    - Sintesi_intervento
    """
    try:
        df = pd.read_csv(input_path, sep=";", dtype=str)
    except FileNotFoundError as exc:
        raise SystemExit(f"❌ File di input non trovato: {input_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"❌ Il file di input è vuoto: {input_path}") from exc
    except Exception as exc:  # pragma: no cover - fallback difensivo
        raise SystemExit(f"❌ Errore nella lettura del file CSV: {exc}") from exc

    if df.empty:
        raise SystemExit(f"❌ Il file di input non contiene righe di dati: {input_path}")

    # Se mancano le colonne chiave, le aggiungo vuote
    for key in GROUP_KEYS:
        if key not in df.columns:
            df[key] = ""

    # Preparo le colonne output
    df["Livello_confidenza"] = ""
    df["Sintesi_intervento"] = ""

    grouped = df.groupby(GROUP_KEYS, dropna=False)

    for _, idx in grouped.indices.items():
        group_df = df.loc[idx]

        # Saltiamo gruppi totalmente vuoti (nessuna info testuale)
        if not has_textual_content(group_df):
            continue

        sintesi, livello = call_codex_for_group(client, group_df)

        df.loc[idx, "Sintesi_intervento"] = sintesi
        df.loc[idx, "Livello_confidenza"] = livello

    df.to_csv(output_path, sep=";", index=False)
    print(f"✅ File elaborato salvato in: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso:")
        print("  python fase1_codex_app.py input.csv output_elaborato.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"❌ Percorso di input non valido o inesistente: {input_path}")
        sys.exit(1)

    client = get_openai_client()
    process_file(input_path, output_path, client)


if __name__ == "__main__":
    main()
