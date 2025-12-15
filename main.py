from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import csv
import io
import json
import os

import google.generativeai as genai


# ==========================
#   COSTANTI E FILE LOCALI
# ==========================

TRAINING_FILE = "training_data.json"      # Storico Sintesi_intervento + Tipo_spesa (CAPEX/OPEX)
CHAT_TRAIN_FILE = "chat_training.json"    # Note di addestramento per l'assistente

# Manuale fisso di classificazione API (usato dal chatbot E dal classificatore)
MANUALE_API = """
=== REGOLE DI CLASSIFICAZIONE GRUPPO API ===

1. RIAPERTURA PUNTI VENDITA
- Manutenzione ordinaria (serrature non CENCON, pulizia serbatoi, semplici riattivazioni): OPEX.
- Manutenzione straordinaria, sostituzione attrezzature, adeguamenti normativi, cambio modalità vendita, lavori su fabbricati: CAPEX.

2. IMPERMEABILIZZAZIONE (Chioschi / Fabbricati / Pensiline)
- Importo < 2.000 €: OPEX.
- Importo > 2.000 €: CAPEX (include permitting e materiali).

3. FASCIONI (Chioschi / Pensiline / Fabbricati)
- Importo < 2.000 €: OPEX.
- Importo > 2.000 €: CAPEX (include permitting e materiali).

4. PIAZZALI / PAVIMENTAZIONI / GRIGLIE
- Importo < 4.000 €: OPEX.
- Importo > 4.000 €: CAPEX (include permitting e materiali).

5. VERNICIATURE ATTREZZATURE/MANUFATTI
- Sola verniciatura senza opere edili: OPEX.
- Verniciature con opere edili strutturali: CAPEX.

6. IMPIANTO ELETTRICO (I.E.)
- Manutenzione / sostituzione prese/interruttori, attività ordinarie: OPEX.
- Rifacimento linee, adeguamento normativo, manutenzione straordinaria: CAPEX.

7. COMPONENTI IMPIANTISTICI SPECIFICI (CAPEX)
- Tipicamente CAPEX: kit differenziale servito, prezzari manuali, sonde di livello, chiusini PDU, condizionatori.

8. ATTREZZATURE TECNOLOGICHE E SICUREZZA (CAPEX)
- Tipicamente CAPEX: schede CPU, Smart Box/Smart Opt/Ubox, UPS, convertitori di protocollo,
  testate erogatori, serrature CENCON, sistemi recupero vapori, lettori banconote, motori serrande.

9. ILLUMINAZIONE E IMMAGINE
- Sostituzione faro LED esistente su pensilina: di solito OPEX.
- Sostituzione pali illuminazione piazzale: CAPEX.
- Fornitura/ripristino adesivi di immagine: OPEX.

10. VARIE AMMINISTRATIVE E TECNICHE
- Collaudi decennali/quindicennali: CAPEX.
- Volture: OPEX.
- Rinnovo CPI: CAPEX.
"""

# ==========================
#          APP
# ==========================

app = FastAPI()

# CORS per permettere all'HTML (index.html) di chiamare questo backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in produzione meglio limitare al dominio aziendale
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
#   UTILS: TRAINING STORAGE
# ==========================

def load_training() -> List[dict]:
    """Carica lo storico CAPEX/OPEX da JSON (Sintesi_intervento + Tipo_spesa)."""
    if not os.path.exists(TRAINING_FILE):
        return []
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []


def save_training(data: List[dict]) -> None:
    """Salva lo storico CAPEX/OPEX su JSON."""
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chat_training() -> List[dict]:
    """Carica le note di addestramento dell'assistente (chat)."""
    if not os.path.exists(CHAT_TRAIN_FILE):
        return []
    with open(CHAT_TRAIN_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []


def save_chat_training(data: List[dict]) -> None:
    """Salva le note di addestramento dell'assistente (chat)."""
    with open(CHAT_TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==========================
#     UTILS: IMPORTI
# ==========================

def parse_importo(val: str) -> float:
    """Converte una stringa importo in float, gestendo punti/virgole."""
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    s = s.replace("€", "").replace("EUR", "").replace("eur", "").replace(" ", "")
    # Gestione separatori
    if "," in s and "." in s:
        # tipo 1.234,56 -> 1234.56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # tipo 1234,56 -> 1234.56
        s = s.replace(",", ".")
    # altrimenti 1234.56 va già bene
    try:
        return float(s)
    except ValueError:
        return 0.0

# ==========================
#     GEMINI (GOOGLE AI)
# ==========================

def get_gemini_api_key(header_key: Optional[str]) -> str:
    """
    Recupera la API key Gemini:
    - prima dall'header x-api-key (inviato dal frontend),
    - altrimenti dalla variabile d'ambiente GEMINI_API_KEY.
    """
    key = header_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "Nessuna API key Gemini trovata. "
                "Imposta la variabile d'ambiente GEMINI_API_KEY "
                "oppure passa la key nell'header 'x-api-key'."
            ),
        )
    return key


def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Chiama il modello Gemini e restituisce il testo di risposta.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    return text.strip()


def call_gemini_json(
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """
    Chiama Gemini come testo normale e cerca di estrarre un JSON valido
    anche se il modello restituisce testo extra o ```json ...
    """
    raw = call_gemini(prompt, api_key=api_key, model_name=model_name)

    print("[DEBUG Gemini RAW]:", raw[:400])

    if not raw:
        raise ValueError("Gemini ha restituito una risposta vuota.")

    raw_clean = raw.strip()

    # 1) Se ci sono fence tipo ```json ... ``` li rimuovo
    if raw_clean.startswith("```"):
        first_newline = raw_clean.find("\n")
        last_backticks = raw_clean.rfind("```")
        if first_newline != -1 and last_backticks != -1 and last_backticks > first_newline:
            raw_clean = raw_clean[first_newline:last_backticks].strip()

    # 2) Estraggo solo da { ... } nel caso ci sia testo prima/dopo
    start = raw_clean.find("{")
    end = raw_clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_json = raw_clean[start : end + 1]
    else:
        # Se non trovo graffe, provo comunque con la stringa intera
        raw_json = raw_clean

    print("[DEBUG Gemini JSON candidate]:", raw_json[:400])

    # 3) Provo a parsare
    return json.loads(raw_json)

# ==========================
#     MODEL CALL: SINTESI
# ==========================

def call_model_for_summary(api_key: str, descrizioni: List[dict]) -> str:
    """
    Genera la Sintesi_intervento per un gruppo di righe con stesso Numero Oda + Oggetto Lavori.
    """
    rows_text = ""
    for i, row in enumerate(descrizioni, start=1):
        rows_text += f"Riga {i}:\n"
        for k, v in row.items():
            if v:
                rows_text += f"- {k}: {v}\n"
        rows_text += "\n"

    prompt = f"""
Sei un assistente interno del reparto Acquisti di un'azienda petrolifera.

Ti fornisco le righe di un unico intervento (stesso Numero Oda e stesso Oggetto Lavori).

DEVI:
- capire che tipo di attività è stata svolta (manutenzione, adeguamento, sostituzione, fornitura, ecc.),
- capire su cosa si interviene (impianto, pompe, quadro elettrico, pensilina, chiosco, piazzale, ecc.),
- capire, se possibile, PERCHÉ si fa l'intervento (adeguamento normativo, guasto, omologazione, sicurezza, riapertura punto vendita, ecc.),
- ignorare la manodopera come oggetto principale (non citarla nella descrizione),
- mettere in evidenza l'attività principale e l'oggetto principale, tenendo conto della voce con importo più alto se è significativa,
- scrivere una Sintesi_intervento in 2–4 frasi, in italiano chiaro e professionale,
  rielaborando il testo (NON fare copia-incolla delle descrizioni originali).

IMPORTANTISSIMO:
- NON nominare CAPEX o OPEX.
- Non elencare tutte le frasi una dietro l'altra: scrivi un testo fluido, leggibile.
- Concentrati sul "cosa si è fatto" e "perché" dal punto di vista operativo.

Ecco le righe dell'intervento:

{rows_text}

Rispondi SOLO con la Sintesi_intervento finale, senza spiegazioni extra.
"""

    sintesi = call_gemini(prompt, api_key=api_key)
    return sintesi

# ==========================
# MODEL CALL: CAPEX / OPEX
# ==========================

def call_model_for_capex_opex(
    api_key: str,
    sintesi: str,
    examples: List[dict],
    extra_context: str = "",
) -> dict:
    """
    Classifica una Sintesi_intervento come CAPEX o OPEX usando:
    - manuale interno MANUALE_API
    - esempi reali dal training_data.json
    - eventuale contesto numerico (es. importo totale).

    Ritorna:
    {
      "tipo_spesa": "CAPEX"/"OPEX" o "",
      "confidenza": int 0-100,
      "motivazione": str,
      "sintesi_breve": str (max ~40 caratteri)
    }
    """

    # Costruisco il blocco di esempi reali
    examples_text = ""
    for ex in examples[:15]:
        s = ex.get("sintesi", "")
        l = ex.get("label", "")
        if not (s and l):
            continue
        examples_text += f"SINTESI: {s}\n"
        examples_text += f"TIPO_SPESA: {l}\n\n"

    context_block = ""
    if extra_context:
        context_block = f"\nCONTESTO AGGIUNTIVO SULL'INTERVENTO:\n{extra_context}\n"

    user_prompt = f"""
Sei un assistente contabile del reparto Investimenti/Manutenzioni di un gruppo petrolifero.

Hai il seguente MANUALE INTERNO di classificazione CAPEX/OPEX (derivato da IAS 16 e policy aziendali):

{MANUALE_API}

Hai anche alcuni ESEMPI REALI di interventi già classificati:

{examples_text if examples_text else "(nessun esempio fornito, usa solo il manuale e il buon senso contabile-finanziario)."}

{context_block}

La seguente è la descrizione sintetica di UN SINGOLO INTERVENTO
(già raggruppato per Numero Oda + Oggetto Lavori):

\"\"\"{sintesi}\"\"\"


COMPITO:
1. Decidi se l'intervento è CAPEX o OPEX, seguendo il manuale e gli esempi.
2. Stima un livello di confidenza (0–100) sulla classificazione.
3. Scrivi una MOTIVAZIONE in 2–4 frasi in italiano,
   spiegando in modo chiaro perché lo consideri CAPEX oppure OPEX
   (richiamando il tipo di attività, la natura dell'intervento, l'orizzonte temporale del beneficio, le soglie di importo, ecc.).
4. Genera una SINTESI_BREVE (stile titolo) in massimo 40 caratteri:
   - non superare 40 caratteri,
   - evita di scrivere CAPEX/OPEX nel titolo,
   - deve riassumere in modo molto sintetico l'intervento principale.

IMPORTANTISSIMO:
- Rispondi SOLO con un JSON **puro**, senza testo introduttivo o conclusivo.
- NON usare markdown, NON usare ```json, NON mettere niente fuori dalle parentesi graffe.

STRUTTURA ESATTA DEL JSON DA RESTITUIRE:

{
  "tipo_spesa": "CAPEX" oppure "OPEX",
  "confidenza": numero_intero_da_0_a_100,
  "motivazione": "testo in italiano che spiega in 2-4 frasi il perché della classificazione",
  "sintesi_breve": "titolo sintetico massimo 40 caratteri"
}
"""

    # ==========================
    #   CHIAMATA AL MODELLO
    # ==========================
    try:
        data = call_gemini_json(user_prompt, api_key=api_key)
    except Exception as e:
        # Se succede qualcosa, loggo e torno valori vuoti (ma scrivo il motivo in motivazione)
        print(f"[call_model_for_capex_opex] Errore chiamata/parsing JSON: {e}")
        return {
            "tipo_spesa": "",
            "confidenza": 0,
            "motivazione": f"Errore tecnico nella classificazione: {e}",
            "sintesi_breve": "",
        }

    # ==========================
    #   NORMALIZZAZIONE OUTPUT
    # ==========================

    # tipo_spesa – accetto anche nomi "strani" per robustezza
    tipo_keys = ["tipo_spesa", "Tipo_spesa", "tipo", "classificazione", "label"]
    tipo_val = ""
    for k in tipo_keys:
        if k in data and data[k]:
            tipo_val = str(data[k])
            break

    tipo = tipo_val.strip().upper()
    if tipo not in ("CAPEX", "OPEX"):
        tipo = ""

    # confidenza
    conf_raw = data.get("confidenza", data.get("confidence", 0))
    try:
        conf = int(conf_raw)
    except Exception:
        try:
            conf = int(float(conf_raw))
        except Exception:
            conf = 0

    # motivazione: tolgo a capo e spazi doppi, così il CSV non si rompe
    motiv = (data.get("motivazione") or data.get("motivo") or "").strip()
    motiv = motiv.replace("\r\n", "\n").replace("\r", "\n")
    motiv = " ".join(part.strip() for part in motiv.split("\n") if part.strip())
    motiv = " ".join(motiv.split())

    # sintesi_breve: massimo 40 caratteri, accetto anche chiavi alternative
    short = (data.get("sintesi_breve") or data.get("titolo") or "").strip()
    if len(short) > 40:
        short = short[:40].rstrip()

    return {
        "tipo_spesa": tipo,
        "confidenza": conf,
        "motivazione": motiv,
        "sintesi_breve": short,
    }

# ==========================
#     MODELLI Pydantic
# ==========================

class ChatTrainPayload(BaseModel):
    message: str


class ChatQuestionPayload(BaseModel):
    question: str

# ==========================
#  ENDPOINT: HEALTHCHECK
# ==========================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ==========================
#  ENDPOINT: PHASE 1 (solo sintesi)
# ==========================

@app.post("/phase1_sintesi")
async def phase1_sintesi(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """
    Fase 1: genera solo Sintesi_intervento + Livello_confidenza_descrizione
    raggruppando per (Numero Oda, Oggetto Lavori).
    """
    api_key = get_gemini_api_key(x_api_key)

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV vuoto.")

    # Gruppi per (Numero Oda, Oggetto Lavori)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        oda = (row.get("Numero Oda") or "").strip()
        ogg = (row.get("Oggetto Lavori") or "").strip()
        key = f"{oda}|||{ogg}"
        groups.setdefault(key, []).append(row)

    # Sintesi per ogni gruppo
    group_summaries: dict[str, dict] = {}
    for key, grp_rows in groups.items():
        try:
            sintesi = call_model_for_summary(api_key, grp_rows)
        except Exception as e:
            # Se Gemini esplode, non blocchiamo tutto il file
            print(f"[phase1_sintesi] Errore su gruppo {key}: {e}")
            sintesi = ""
        # livello confidenza descrizione: euristica grezza
        text_concat = (
            (grp_rows[0].get("Descrizione Riga") or "") + " " +
            (grp_rows[0].get("Descrizione Abb Riga") or "") + " " +
            (grp_rows[0].get("Note") or "") + " " +
            (grp_rows[0].get("Note No Cont.") or "")
        )
        conf = 80 if len(text_concat.strip()) > 40 else 60
        group_summaries[key] = {"sintesi": sintesi, "conf": conf}

    # Preparo output con colonne aggiuntive
    fieldnames = reader.fieldnames or []
    if "Livello_confidenza_descrizione" not in fieldnames:
        fieldnames.append("Livello_confidenza_descrizione")
    if "Sintesi_intervento" not in fieldnames:
        fieldnames.append("Sintesi_intervento")

    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for row in rows:
        oda = (row.get("Numero Oda") or "").strip()
        ogg = (row.get("Oggetto Lavori") or "").strip()
        key = f"{oda}|||{ogg}"
        info = group_summaries.get(key, {})
        row["Livello_confidenza_descrizione"] = info.get("conf", "")
        row["Sintesi_intervento"] = info.get("sintesi", "")
        writer.writerow(row)

    csv_data = output_io.getvalue()
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="risultato_phase1_sintesi.csv"'
        },
    )

# ==========================
#  ENDPOINT: FULL CLASSIFY
# ==========================

@app.post("/full_classify")
async def full_classify(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """
    Fase completa:
    - genera Sintesi_intervento + Livello_confidenza_descrizione (come phase1_sintesi)
    - usa lo storico (training_data.json) per proporre CAPEX/OPEX
      con:
        * Tipo_spesa_predetta
        * Livello_confidenza_classificazione
        * Motivo_classificazione (testo lungo)
        * Sintesi_breve_40_caratteri (titolo max 40 char)
    """
    api_key = get_gemini_api_key(x_api_key)
    training = load_training()
    if not training:
        raise HTTPException(
            status_code=400,
            detail=(
                "Nessun dato di addestramento presente. "
                "Carica prima un file di addestramento con una colonna CAPEX/OPEX."
            ),
        )

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV vuoto.")

    # Gruppi per (Numero Oda, Oggetto Lavori)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        oda = (row.get("Numero Oda") or "").strip()
        ogg = (row.get("Oggetto Lavori") or "").strip()
        key = f"{oda}|||{ogg}"
        groups.setdefault(key, []).append(row)

    group_summaries: dict[str, dict] = {}
    for key, grp_rows in groups.items():
        try:
            sintesi = call_model_for_summary(api_key, grp_rows)
        except Exception as e:
            print(f"[full_classify] Errore sintesi gruppo {key}: {e}")
            sintesi = ""

        # Confidenza sulla descrizione (stessa logica di phase1)
        text_concat = (
            (grp_rows[0].get("Descrizione Riga") or "") + " " +
            (grp_rows[0].get("Descrizione Abb Riga") or "") + " " +
            (grp_rows[0].get("Note") or "") + " " +
            (grp_rows[0].get("Note No Cont.") or "")
        )
        conf_descr = 80 if len(text_concat.strip()) > 40 else 60

        # Calcolo importo totale dell'intervento (sommando le righe del gruppo)
        total_importo = 0.0
        for r in grp_rows:
            imp_str = (
                r.get("Importo")
                or r.get("Importo netto")
                or r.get("Importo Netto")
                or r.get("Importo totale")
                or r.get("Importo Totale")
                or ""
            )
            total_importo += parse_importo(imp_str)

        # Costruisco un contesto numerico da passare al modello
        oda_val = (grp_rows[0].get("Numero Oda") or "").strip()
        ogg_val = (grp_rows[0].get("Oggetto Lavori") or "").strip()
        extra_parts = []
        if oda_val or ogg_val:
            extra_parts.append(
                f"Numero Oda: {oda_val or 'n/d'}, Oggetto Lavori: {ogg_val or 'n/d'}."
            )
        if total_importo > 0:
            extra_parts.append(
                f"Importo totale stimato dell'intervento: {total_importo:.2f} euro."
            )
        extra_context = "\n".join(extra_parts)

        try:
            pred = call_model_for_capex_opex(
                api_key=api_key,
                sintesi=sintesi,
                examples=training,
                extra_context=extra_context,
            )
        except Exception as e:
            print(f"[full_classify] Errore classificazione gruppo {key}: {e}")
            pred = {
                "tipo_spesa": "",
                "confidenza": 0,
                "motivazione": f"Errore tecnico nella classificazione: {e}",
                "sintesi_breve": "",
            }

        group_summaries[key] = {
            "sintesi": sintesi,
            "conf_descr": conf_descr,
            "tipo_spesa_predetta": pred.get("tipo_spesa", ""),
            "conf_class": pred.get("confidenza", 0),
            "motivo": pred.get("motivazione", ""),
            "short": pred.get("sintesi_breve", ""),
        }

    fieldnames = reader.fieldnames or []
    extra_cols = [
        # Già esistenti dalla fase 1:
        "Livello_confidenza_descrizione",
        "Sintesi_intervento",
        # Classificazione:
        "Tipo_spesa_predetta",
        "Livello_confidenza_classificazione",
        # Nuove colonne richieste:
        "Motivo_classificazione",
        "Sintesi_breve_40_caratteri",
    ]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()

    for row in rows:
        oda = (row.get("Numero Oda") or "").strip()
        ogg = (row.get("Oggetto Lavori") or "").strip()
        key = f"{oda}|||{ogg}"
        info = group_summaries.get(key, {})

        row["Livello_confidenza_descrizione"] = info.get("conf_descr", "")
        row["Sintesi_intervento"] = info.get("sintesi", "")
        row["Tipo_spesa_predetta"] = info.get("tipo_spesa_predetta", "")
        row["Livello_confidenza_classificazione"] = info.get("conf_class", "")
        row["Motivo_classificazione"] = info.get("motivo", "")
        row["Sintesi_breve_40_caratteri"] = info.get("short", "")

        writer.writerow(row)

    csv_data = output_io.getvalue()
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="risultato_full_classify.csv"'
        },
    )

# ==========================
#  ENDPOINT: TRAIN CAPEX/OPEX (ADATTABILE)
# ==========================

@app.post("/train")
async def train_endpoint(file: UploadFile = File(...)):
    """
    Addestra il sistema CAPEX/OPEX a partire da un CSV etichettato.

    Funziona sia se hai la colonna 'Sintesi_intervento' sia se NON ce l'hai.

    Requisiti minimi del file:
    - Una colonna con valori CAPEX / OPEX (es. 'Tipo_spesa', 'Tipo spesa', 'CAPEX_OPEX').

    Colonne usate come contesto, se presenti:
    - 'Numero Oda'
    - 'Oggetto Lavori'
    - 'Descrizione Riga'
    - 'Descrizione Abb Riga'
    - 'Note'
    - 'Note No Cont.'
    - 'Importo' / 'Importo netto' / 'Importo totale' ...
    """
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    # --- 1) Provo a capire il delimitatore automaticamente (; oppure ,) ---
    def parse_with_delimiter(delim: str):
        r = csv.DictReader(io.StringIO(text), delimiter=delim)
        rows_local = list(r)
        fields_local = r.fieldnames or []
        return fields_local, rows_local

    fieldnames, rows = parse_with_delimiter(";")
    # Se sembra che ci sia una sola colonna, può essere il delimitatore sbagliato
    if len(fieldnames) <= 1:
        fieldnames2, rows2 = parse_with_delimiter(",")
        if len(fieldnames2) > len(fieldnames):
            fieldnames, rows = fieldnames2, rows2

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=f"CSV vuoto o non leggibile. Colonne rilevate: {fieldnames!r}",
        )

    print("DEBUG /train – colonne CSV:", fieldnames)

    # --- 2) Trovo la colonna etichetta CAPEX/OPEX ---
    def norm(name: str) -> str:
        return name.strip().lower().replace(" ", "").replace("_", "")

    candidate_norms = {
        "tipospesa",
        "capexopex",
        "capex_opex",
        "capexopex",
        "classificazione",
        "label",
    }

    label_col = None
    for col in fieldnames:
        if norm(col) in candidate_norms:
            label_col = col
            break

    # Se non ho trovato nulla per nome, provo a capire per contenuto
    if not label_col:
        possible = []
        for col in fieldnames:
            values = [(row.get(col) or "").strip().upper() for row in rows]
            nonempty = [v for v in values if v]
            if not nonempty:
                continue
            capex_like = [v for v in nonempty if v in ("CAPEX", "OPEX")]
            if len(capex_like) >= max(3, int(len(nonempty) * 0.5)):
                possible.append(col)
        if possible:
            label_col = possible[0]

    if not label_col:
        raise HTTPException(
            status_code=400,
            detail=(
                "Non trovo alcuna colonna che sembri contenere CAPEX/OPEX.\n"
                f"Colonne trovate: {', '.join(fieldnames)}.\n"
                "Assicurati che esista una colonna (es. 'Tipo_spesa', 'Tipo spesa', 'CAPEX_OPEX') "
                "con valori CAPEX/OPEX."
            ),
        )

    print("DEBUG /train – colonna etichetta usata:", label_col)

    # --- 3) Costruisco gli esempi di addestramento ---
    new_examples = []

    for row in rows:
        raw_label = (row.get(label_col) or "").strip().upper()
        if raw_label not in ("CAPEX", "OPEX"):
            # Riga non utilizzabile come addestramento
            continue
        label = raw_label

        # Campi di contesto
        numero_oda = (row.get("Numero Oda") or row.get("Numero ODA") or "").strip()
        oggetto_lavori = (row.get("Oggetto Lavori") or "").strip()
        descr_lunga = (row.get("Descrizione Riga") or "").strip()
        descr_breve = (row.get("Descrizione Abb Riga") or "").strip()
        note = (row.get("Note") or "").strip()
        note_no_cont = (row.get("Note No Cont.") or "").strip()

        # Importo: prendo la prima colonna sensata che trovo
        importo = (
            row.get("Importo")
            or row.get("Importo netto")
            or row.get("Importo Netto")
            or row.get("Importo totale")
            or row.get("Importo Totale")
            or ""
        )
        importo = str(importo).strip()

        # Sintesi di training: uso 'Sintesi_intervento' se esiste, altrimenti la costruisco
        sintesi = (row.get("Sintesi_intervento") or "").strip()
        if not sintesi:
            parts = []
            if oggetto_lavori:
                parts.append(oggetto_lavori)
            if descr_breve:
                parts.append(descr_breve)
            if descr_lunga:
                parts.append(descr_lunga)
            if note:
                parts.append(f"Note: {note}")
            if note_no_cont and not note:
                parts.append(f"Note: {note_no_cont}")

            sintesi = " – ".join(parts).strip()
            if not sintesi:
                sintesi = f"Intervento su ODA {numero_oda or 'sconosciuta'}"

        example = {
            "sintesi": sintesi,
            "label": label,
            "context": {
                "numero_oda": numero_oda,
                "oggetto_lavori": oggetto_lavori,
                "descrizione_lunga": descr_lunga,
                "descrizione_breve": descr_breve,
                "note": note,
                "note_no_cont": note_no_cont,
                "importo": importo,
            },
        }
        new_examples.append(example)

    print("DEBUG /train – esempi validi trovati:", len(new_examples))

    if not new_examples:
        raise HTTPException(
            status_code=400,
            detail=(
                "Ho trovato la colonna etichetta, ma nessuna riga con CAPEX/OPEX validi.\n"
                f"Colonna etichetta: {label_col}\n"
                "Controlla che i valori siano proprio 'CAPEX' o 'OPEX' (anche con maiuscole/minuscole diverse va bene)."
            ),
        )

    # --- 4) Unisco con lo storico esistente ---
    data = load_training()
    if not isinstance(data, list):
        data = []
    data.extend(new_examples)
    save_training(data)

    return {
        "status": "ok",
        "aggiunti": len(new_examples),
        "totale": len(data),
    }

# ==========================
#  ENDPOINT: CHAT TRAIN
# ==========================

@app.post("/chat_train")
async def chat_train(payload: ChatTrainPayload):
    """
    Salva una nota di addestramento 'libera' per l'assistente.
    Es: nuove circolari, eccezioni, aggiornamenti alle regole.
    """
    notes = load_chat_training()
    notes.append({"message": payload.message})
    save_chat_training(notes)
    return {"status": "ok", "totale_note": len(notes)}

# ==========================
#  ENDPOINT: CHAT ANSWER
# ==========================

@app.post("/chat_answer")
async def chat_answer(
    payload: ChatQuestionPayload,
    x_api_key: Optional[str] = Header(None),
):
    """
    Chat aziendale CAPEX/OPEX con tre fonti:
    - manuale base del Gruppo API,
    - note di addestramento (chat_training.json),
    - esempi reali (training_data.json).
    """
    try:
        api_key = get_gemini_api_key(x_api_key)
    except HTTPException:
        return {
            "answer": (
                "Non posso contattare il modello perché non trovo nessuna API key Gemini.\n\n"
                "Vai nel menù a sinistra, inserisci la tua API key nel box "
                "\"API key\", premi \"Salva\" e poi riprova a fare la domanda."
            )
        }

    # Note interne
    notes = load_chat_training()
    notes_text = ""
    if notes:
        notes_text = "NOTE INTERNE (circolari, eccezioni, aggiornamenti):\n"
        for i, n in enumerate(notes, start=1):
            msg = n.get("message", "")
            if msg:
                notes_text += f"- Nota {i}: {msg}\n"
        notes_text += "\n"
    else:
        notes_text = "(nessuna nota interna presente al momento)\n\n"

    # Esempi reali dallo storico CAPEX/OPEX
    training = load_training()
    examples_text = ""
    if training:
        examples_text = "ESEMPI REALI DI SINTESI_INTERVENTO GIÀ CLASSIFICATI:\n"
        for ex in training[:10]:
            sint = ex.get("sintesi", "")
            label = ex.get("label", "")
            if sint and label:
                short_sint = sint[:200] + ("..." if len(sint) > 200 else "")
                examples_text += f"- [{label}] {short_sint}\n"
        examples_text += "\n"
    else:
        examples_text = "(nessun esempio reale disponibile al momento)\n\n"

    user_question = payload.question

    prompt = f"""
Sei l'assistente interno del reparto Acquisti/Amministrazione del Gruppo API.

Puoi usare SOLO queste fonti:

1) MANUALE UFFICIALE DI CLASSIFICAZIONE:
{MANUALE_API}

2) NOTE INTERNE DI ADDESTRAMENTO (NUOVE CIRCOLARI, ECCEZIONI):
{notes_text}

3) ESEMPI REALI GIÀ CLASSIFICATI:
{examples_text}

DOMANDA DELL'UTENTE:
\"\"\"{user_question}\"\"\"


COMPITO:
- Rispondi in modo chiaro e professionale, in italiano.
- Usa SOLO le informazioni contenute in manuale, note interne ed esempi reali.
- Se la domanda riguarda un intervento specifico (es. lavori, manutenzioni, sostituzioni):
  - spiega il ragionamento,
  - proponi una classificazione CAPEX o OPEX,
  - chiudi SEMPRE con una frase del tipo:
    "Classificazione proposta: CAPEX (motivo: ...)" oppure
    "Classificazione proposta: OPEX (motivo: ...)".*
- Se le informazioni non sono sufficienti per una risposta motivata:
  - dillo chiaramente,
  - spiega quali dati mancano (importo, tipo di intervento, opere edili sì/no, ecc.),
  - NON inventare regole che non derivano dalle fonti.

IMPORTANTISSIMO:
- Se una nota interna (circolare) contraddice il manuale, dai PRIORITÀ alla nota interna più recente.
- Non usare conoscenza esterna al contesto fornito. Se non trovi regole rilevanti, di' che non puoi classificare con sicurezza.
"""

    try:
        answer = call_gemini(prompt, api_key=api_key)
        if not answer:
            answer = "Non sono riuscito a generare una risposta sulla base delle informazioni disponibili."
    except Exception as e:
        answer = (
            "Si è verificato un errore tecnico nel chiamare il modello di IA (Gemini).\n\n"
            f"Dettaglio tecnico: {e}"
        )

    return {"answer": answer}

# ==========================
#  ENDPOINT: STORICI
# ==========================

@app.get("/training_list")
async def training_list():
    """Ritorna lo storico CAPEX/OPEX (Sintesi_intervento + label)."""
    data = load_training()
    return {"data": data}


@app.post("/training_clear")
async def training_clear():
    """Svuota completamente lo storico CAPEX/OPEX."""
    save_training([])
    return {"status": "ok", "message": "Storico CAPEX/OPEX svuotato."}


@app.get("/chat_training_list")
async def chat_training_list():
    """Ritorna tutte le note di addestramento dell'assistente."""
    data = load_chat_training()
    return {"data": data}


@app.post("/chat_training_clear")
async def chat_training_clear():
    """Svuota tutte le note di addestramento dell'assistente."""
    save_chat_training([])
    return {"status": "ok", "message": "Storico note assistente svuotato."}
