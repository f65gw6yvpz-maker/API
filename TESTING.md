# Guida per Testare l'Applicazione

Questa guida spiega come eseguire e testare l'applicazione CAPEX/OPEX.

## Prerequisiti

- Python 3.12 o superiore
- pip (gestore pacchetti Python)

## Installazione

1. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

## Avvio dell'Applicazione

### 1. Avvia il Backend (API Server)

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Il server sarà disponibile su `http://127.0.0.1:8000`

### 2. Verifica che il Backend sia attivo

Puoi testare che il server funzioni correttamente visitando:

```bash
curl http://127.0.0.1:8000/health
```

Dovresti ricevere: `{"status":"ok"}`

### 3. Apri il Frontend

In un'altra finestra del terminale, avvia un server HTTP per servire l'interfaccia web:

```bash
python -m http.server 8080 --bind 127.0.0.1
```

Poi apri il browser e vai a:

```
http://127.0.0.1:8080/index.html
```

## Configurazione API Key

Prima di utilizzare l'applicazione, devi configurare la tua API key di Gemini:

1. Vai su [Google AI Studio](https://makersuite.google.com/app/apikey) per ottenere una API key
2. Nell'interfaccia web, inserisci la tua API key nel campo "API key IA (Gemini / OpenAI)"
3. Clicca su "Salva"

La chiave verrà salvata nel browser e utilizzata per tutte le richieste.

## Test delle Funzionalità

### 1. Health Check

```bash
curl http://127.0.0.1:8000/health
```

### 2. Lista Training CAPEX/OPEX

```bash
curl http://127.0.0.1:8000/training_list
```

### 3. Lista Note Assistente

```bash
curl http://127.0.0.1:8000/chat_training_list
```

## Utilizzo dell'Interfaccia Web

L'interfaccia web include le seguenti sezioni:

1. **Panoramica**: Descrizione del flusso di lavoro
2. **Fase 1 – Sintesi**: Carica un CSV per generare sintesi automatiche
3. **Addestramento CAPEX/OPEX**: Carica file etichettati per addestrare il sistema
4. **Classificatore finale**: Classifica automaticamente gli interventi come CAPEX/OPEX
5. **Assistente IA**: Chat per domande e addestramento personalizzato
6. **Storico addestramento**: Visualizza e gestisci i dati di addestramento

## Note

- I dati di addestramento sono salvati in file JSON locali (`training_data.json` e `chat_training.json`)
- Il server deve essere in esecuzione per utilizzare l'interfaccia web
- Assicurati di avere una connessione internet per utilizzare l'API di Gemini
