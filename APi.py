#!/usr/bin/env python3
"""
Backend FastAPI per Assistente CAPEX/OPEX
- GEMINI/AI key via env var GEMINI_API_KEY (fallback: x-api-key if ENABLE_X_API_KEY=true)
- CORS via env var ALLOWED_ORIGINS (comma-separated)
- CSV parsing with delimiter detection
- File locking for training_data.json and chat_training.json (FileLock)
- Endpoints: /phase1_sintesi, /train, /full_classify, chat endpoints and history
"""
import os
import io
import csv
import json
import re
import random
from typing import Optional, List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from filelock import FileLock, Timeout
import requests

# Config
DATA_DIR = os.getenv("DATA_DIR", ".")
TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
CHAT_TRAINING_FILE = os.path.join(DATA_DIR, "chat_training.json")
TRAINING_LOCK = TRAINING_FILE + ".lock"
CHAT_LOCK = CHAT_TRAINING_FILE + ".lock"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # preferred in production
ENABLE_X_API_KEY = os.getenv("ENABLE_X_API_KEY", "false").lower() in ("1", "true", "yes")
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://api.openai.com/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000").split(",")

# Ensure data files exist
for path, default in [(TRAINING_FILE, "[]"), (CHAT_TRAINING_FILE, "[]")]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(default)

app = FastAPI(title="Assistente CAPEX/OPEX")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utilities
def detect_delimiter(sample: str) -> str:
    if ";" in sample:
        return ";"
    return ","

def read_csv_text(content: bytes) -> List[Dict[str, str]]:
    text = content.decode("utf-8", errors="replace")
    first_lines = "\n".join(text.splitlines()[:5])
    delimiter = detect_delimiter(first_lines)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return list(reader)

def group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str,str], List[Dict[str,str]]]:
    groups = {}
    for r in rows:
        key1 = (r.get("Numero Oda") or r.get("Numero ODA") or "").strip()
        key2 = (r.get("Oggetto Lavori") or "").strip()
        if not key1 or not key2:
            continue
        key = (key1, key2)
        groups.setdefault(key, []).append(r)
    return groups

def build_group_text(rows: List[Dict[str,str]]) -> str:
    parts = []
    for r in rows:
        for k, v in r.items():
            if v is None:
                continue
            v = str(v).strip()
            if not v:
                continue
            parts.append(f"{k}: {v}")
    return "\n".join(parts)

def heuristic_confidence_by_text(s: str) -> int:
    if len(s.strip()) > 40:
        return 80
    return 60

def parse_importo(s: str) -> float:
    if not s:
        return 0.0
    s_clean = re.sub(r"[^\d,.\-]", "", str(s))
    if "." in s_clean and "," in s_clean:
        if s_clean.rfind(",") > s_clean.rfind("."):
            s_normal = s_clean.replace(".", "").replace(",", ".")
        else:
            s_normal = s_clean.replace(",", "")
    else:
        if "," in s_clean and "." not in s_clean:
            s_normal = s_clean.replace(",", ".")
        else:
            s_normal = s_clean
    try:
        return float(s_normal)
    except Exception:
        return 0.0

def choose_examples(training: List[Dict], n: int = 6) -> List[Dict]:
    if not training:
        return []
    if len(training) <= n:
        return training
    return random.sample(training, n)

def call_model_system(prompt: str, api_key: Optional[str]) -> str:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="API key missing (server).")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.0,
    }
    resp = requests.post(MODEL_API_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"Model API error: {resp.status_code} {resp.text}")
    data = resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    return data.get("text", "")

def safe_load_json_file(path: str, lock_path: str) -> List:
    lock = FileLock(lock_path, timeout=5)
    try:
        with lock:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Timeout:
        raise HTTPException(status_code=500, detail="File lock timeout")
    except json.JSONDecodeError:
        return []

def safe_append_and_save(path: str, lock_path: str, items: List[Dict]) -> int:
    lock = FileLock(lock_path, timeout=5)
    try:
        with lock:
            current = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        current = json.load(f)
                    except json.JSONDecodeError:
                        current = []
            current.extend(items)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
            return len(current)
    except Timeout:
        raise HTTPException(status_code=500, detail="File lock timeout")

# ---------- Endpoints ----------

@app.post("/phase1_sintesi")
async def phase1_sintesi(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """
    Fase 1: raggruppa per (Numero Oda, Oggetto Lavori), costruisce blocchi e chiede al modello
    una Sintesi_intervento (2-4 frasi, italiano). Restituisce CSV con due colonne in più.
    """
    # API key resolution
    api_key = None
    if ENABLE_X_API_KEY and x_api_key:
        api_key = x_api_key
    elif GEMINI_API_KEY:
        api_key = GEMINI_API_KEY
    else:
        # require API key for model call
        raise HTTPException(status_code=400, detail="API key not configured on server")

    content = await file.read()
    rows = read_csv_text(content)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV vuoto o non leggibile")

    groups = group_rows(rows)

    # build a map (key -> sintesi, conf)
    group_results = {}
    for (oda, obj), g_rows in groups.items():
        block = build_group_text(g_rows)
        # construct prompt per tua specifica
        prompt = (
            "Ignora la manodopera come oggetto principale. "
            "Descrivi cosa è stato fatto e perché in 2–4 frasi, in italiano. "
            "Non nominare CAPEX o OPEX. "
            "Testo da leggere:\n" + block
        )
        try:
            model_text = call_model_system(prompt, api_key)
        except HTTPException as e:
            raise e
        sintesi = model_text.strip()
        conf = heuristic_confidence_by_text(sintesi)
        group_results[(oda, obj)] = {"sintesi": sintesi, "conf": conf}

    # Produce CSV output (same headers + new cols) keeping original order
    input_text = content.decode("utf-8", errors="replace")
    delimiter = detect_delimiter(input_text.splitlines()[0] if input_text else "")
    first_row = rows[0] if rows else {}
    fieldnames = list(first_row.keys())
    # ensure new columns appended
    extra_cols = ["Livello_confidenza_descrizione", "Sintesi_intervento"]
    out_fieldnames = fieldnames + [c for c in extra_cols if c not in fieldnames]

    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=out_fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        key = ((r.get("Numero Oda") or r.get("Numero ODA") or "").strip(), (r.get("Oggetto Lavori") or "").strip())
        res = group_results.get(key, {"sintesi": "", "conf": 60})
        r_out = dict(r)
        r_out["Livello_confidenza_descrizione"] = res["conf"]
        r_out["Sintesi_intervento"] = res["sintesi"]
        writer.writerow(r_out)

    output_io.seek(0)
    filename = os.path.splitext(file.filename)[0] + "_fase1.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return StreamingResponse(io.BytesIO(output_io.getvalue().encode("utf-8")), media_type="text/csv", headers=headers)

class TrainResponse(BaseModel):
    status: str
    aggiunti: int
    totale: int

LABEL_COLUMNS = ["Tipo_spesa", "Tipo spesa", "CAPEX_OPEX", "CAPEX OPEX", "Classificazione", "Label"]

@app.post("/train", response_model=TrainResponse)
async def train(file: UploadFile = File(...)):
    """
    Carica CSV etichettato e aggiunge esempi a training_data.json
    """
    content = await file.read()
    rows = read_csv_text(content)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV vuoto o non leggibile")
    added_items = []
    label_col = None
    # determine label column
    header = rows[0].keys()
    for c in LABEL_COLUMNS:
        if c in header:
            label_col = c
            break
    if not label_col:
        # try case-insensitive match
        for h in header:
            if h.lower() in [x.lower() for x in LABEL_COLUMNS]:
                label_col = h
                break
    if not label_col:
        raise HTTPException(status_code=400, detail="Colonna etichetta (CAPEX/OPEX) non trovata")

    for r in rows:
        raw_label = (r.get(label_col) or "").strip()
        if not raw_label:
            continue
        label = raw_label.upper()
        if label not in ("CAPEX", "OPEX"):
            continue
        # build sintesi training
        sintesi = (r.get("Sintesi_intervento") or "").strip()
        if not sintesi:
            parts = []
            for k in ["Oggetto Lavori", "Descrizione Abb Riga", "Descrizione Riga", "Note", "Note No Cont."]:
                v = r.get(k) or r.get(k.title()) or ""
                v = str(v).strip()
                if v:
                    parts.append(v)
            sintesi = " · ".join(parts)[:2000]  # limit length
        context = {
            "numero_oda": (r.get("Numero Oda") or r.get("Numero ODA") or "").strip(),
            "oggetto_lavori": (r.get("Oggetto Lavori") or "").strip(),
            "descrizione_lunga": (r.get("Descrizione Riga") or "").strip(),
            "descrizione_breve": (r.get("Descrizione Abb Riga") or "").strip(),
            "note": (r.get("Note") or "").strip(),
            "note_no_cont": (r.get("Note No Cont.") or "").strip(),
            "importo": (r.get("Importo") or r.get("Importo netto") or r.get("Importo Netto") or r.get("Importo totale") or r.get("Importo Totale") or "").strip(),
        }
        item = {
            "sintesi": sintesi,
            "label": label,
            "context": context
        }
        added_items.append(item)

    if not added_items:
        return TrainResponse(status="ok", aggiunti=0, totale=len(safe_load_json_file(TRAINING_FILE, TRAINING_LOCK)))

    totale = safe_append_and_save(TRAINING_FILE, TRAINING_LOCK, added_items)
    return TrainResponse(status="ok", aggiunti=len(added_items), totale=totale)

@app.post("/full_classify")
async def full_classify(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    """
    Per ogni gruppo genera sintesi (come fase1), somma importi e chiama il modello con esempi da training_data.json
    """
    # resolve API key
    api_key = None
    if ENABLE_X_API_KEY and x_api_key:
        api_key = x_api_key
    elif GEMINI_API_KEY:
        api_key = GEMINI_API_KEY
    else:
        raise HTTPException(status_code=400, detail="API key not configured on server")

    training = safe_load_json_file(TRAINING_FILE, TRAINING_LOCK)
    if not training:
        raise HTTPException(status_code=400, detail="Storico training vuoto, aggiungi esempi con /train")

    content = await file.read()
    rows = read_csv_text(content)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV vuoto o non leggibile")

    groups = group_rows(rows)
    group_results = {}
    # preselect examples to include in prompt
    examples = choose_examples(training, n=6)

    for (oda, obj), g_rows in groups.items():
        block = build_group_text(g_rows)
        # sum importi in group
        total_import = 0.0
        for r in g_rows:
            for col in ("Importo", "Importo netto", "Importo Netto", "Importo totale", "Importo Totale"):
                v = r.get(col) or ""
                if v:
                    total_import += parse_importo(v)
        extra_ctx = f"Numero Oda: {oda}\nOggetto Lavori: {obj}\n"
        if total_import > 0:
            extra_ctx += f"Importo totale: {total_import:.2f} EUR\n"

        prompt_examples = "\n".join([f"EXAMPLE: label={e.get('label')}\n{e.get('sintesi')}\n" for e in examples])
        prompt = (
            "Hai a disposizione il MANUALE_API con le regole aziendali CAPEX/OPEX (non fornito qui). "
            "Usa gli esempi seguenti come riferimento:\n"
            f"{prompt_examples}\n"
            "Ora valuta il seguente intervento (non nominare manualmente CAPEX/OPEX nella spiegazione iniziale):\n"
            f"{block}\n"
            f"{extra_ctx}\n"
            "Rispondi SOLO con JSON contenente esattamente le chiavi: tipo_spesa (CAPEX|OPEX), confidenza (0-100), motivazione (2-4 frasi, italiano), sintesi_breve (max 40 caratteri)."
        )

        try:
            model_text = call_model_system(prompt, api_key)
        except HTTPException as e:
            raise e

        # Attempt to extract JSON from model_text
        model_json = None
        try:
            idx = model_text.find("{")
            if idx >= 0:
                model_json = json.loads(model_text[idx:])
            else:
                model_json = json.loads(model_text)
        except Exception:
            # attempt to sanitize common single quotes -> double quotes
            try:
                cleaned = model_text.replace("'", '"')
                idx = cleaned.find("{")
                if idx >= 0:
                    model_json = json.loads(cleaned[idx:])
            except Exception:
                model_json = {"tipo_spesa": "", "confidenza": 0, "motivazione": "", "sintesi_breve": ""}

        tipo = (model_json.get("tipo_spesa") or "").upper()
        if tipo not in ("CAPEX", "OPEX"):
            tipo = ""
        try:
            conf = int(float(model_json.get("confidenza") or model_json.get("conf") or 0))
            conf = max(0, min(100, conf))
        except Exception:
            conf = 0
        motiv = str(model_json.get("motivazione") or "").replace("\n", " ").strip()
        sint_brev = str(model_json.get("sintesi_breve") or "")[:40].strip()

        # also generate sintesi_intervento via model like phase1 (or reuse)
        prompt_sintesi = (
            "Ignora la manodopera come oggetto principale. "
            "Descrivi cosa è stato fatto e perché in 2–4 frasi, in italiano. "
            "Non nominare CAPEX o OPEX. "
            "Testo da leggere:\n" + block
        )
        try:
            sintesi_text = call_model_system(prompt_sintesi, api_key)
        except Exception:
            sintesi_text = ""

        group_results[(oda, obj)] = {
            "sintesi": sintesi_text.strip(),
            "conf_descr": heuristic_confidence_by_text(sintesi_text or ""),
            "tipo_spesa_predetta": tipo,
            "conf_class": conf,
            "motivo": motiv,
            "sintesi_breve": sint_brev
        }

    # produce CSV with extra 6 columns
    input_text = content.decode("utf-8", errors="replace")
    delimiter = detect_delimiter(input_text.splitlines()[0] if input_text else "")
    first_row = rows[0] if rows else {}
    fieldnames = list(first_row.keys())
    extra_cols = [
        "Livello_confidenza_descrizione",
        "Sintesi_intervento",
        "Tipo_spesa_predetta",
        "Livello_confidenza_classificazione",
        "Motivo_classificazione",
        "Sintesi_breve_40_caratteri"
    ]
    out_fieldnames = fieldnames + [c for c in extra_cols if c not in fieldnames]

    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=out_fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        key = ((r.get("Numero Oda") or r.get("Numero ODA") or "").strip(), (r.get("Oggetto Lavori") or "").strip())
        gr = group_results.get(key, {"sintesi": "", "conf_descr": 60, "tipo_spesa_predetta": "", "conf_class": 0, "motivo": "", "sintesi_breve": ""})
        r_out = dict(r)
        r_out["Livello_confidenza_descrizione"] = gr["conf_descr"]
        r_out["Sintesi_intervento"] = gr["sintesi"]
        r_out["Tipo_spesa_predetta"] = gr["tipo_spesa_predetta"]
        r_out["Livello_confidenza_classificazione"] = gr["conf_class"]
        r_out["Motivo_classificazione"] = gr["motivo"]
        r_out["Sintesi_breve_40_caratteri"] = gr["sintesi_breve"]
        writer.writerow(r_out)

    output_io.seek(0)
    filename = os.path.splitext(file.filename)[0] + "_classificato.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(output_io.getvalue().encode("utf-8")), media_type="text/csv", headers=headers)

# ---------- Chat & history endpoints ----------

@app.post("/chat_train")
async def chat_train(payload: Dict):
    """
    Salva una nota di addestramento per l'assistente (chat_training.json).
    body: { "message": "..." }
    """
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message vuoto")
    item = {"message": message}
    totale = safe_append_and_save(CHAT_TRAINING_FILE, CHAT_LOCK, [item])
    return {"status": "ok", "totale_note": totale}

@app.post("/chat_answer")
async def chat_answer(payload: Dict, x_api_key: Optional[str] = Header(None)):
    """
    Risponde a una domanda usando GEMINI + training notes + training_data.json
    body: { "question": "..." }
    """
    api_key = None
    if ENABLE_X_API_KEY and x_api_key:
        api_key = x_api_key
    elif GEMINI_API_KEY:
        api_key = GEMINI_API_KEY
    else:
        raise HTTPException(status_code=400, detail="API key not configured on server")

    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question vuota")

    training = safe_load_json_file(TRAINING_FILE, TRAINING_LOCK)
    chats = safe_load_json_file(CHAT_TRAINING_FILE, CHAT_LOCK)

    prompt = "Usa le note di addestramento e lo storico per rispondere. Note:\n"
    for c in chats[-8:]:
        prompt += f"- {c.get('message')}\n"
    prompt += "\nEsempi di training (alcuni):\n"
    for e in choose_examples(training, n=4):
        prompt += f"- [{e.get('label')}] {e.get('sintesi')}\n"
    prompt += f"\nDomanda: {question}\nRispondi in italiano."

    try:
        answer = call_model_system(prompt, api_key)
    except HTTPException as e:
        raise e
    return {"answer": answer}

@app.get("/training_list")
async def training_list():
    data = safe_load_json_file(TRAINING_FILE, TRAINING_LOCK)
    return {"status": "ok", "data": data}

@app.get("/chat_training_list")
async def chat_training_list():
    data = safe_load_json_file(CHAT_TRAINING_FILE, CHAT_LOCK)
    return {"status": "ok", "data": data}

@app.post("/training_clear")
async def training_clear():
    lock = FileLock(TRAINING_LOCK, timeout=5)
    try:
        with lock:
            with open(TRAINING_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        return {"status": "ok"}
    except Timeout:
        raise HTTPException(status_code=500, detail="File lock timeout")

@app.post("/chat_training_clear")
async def chat_training_clear():
    lock = FileLock(CHAT_LOCK, timeout=5)
    try:
        with lock:
            with open(CHAT_TRAINING_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        return {"status": "ok"}
    except Timeout:
        raise HTTPException(status_code=500, detail="File lock timeout")

# Simple health
@app.get("/health")
async def health():
    return {"status": "ok"}
