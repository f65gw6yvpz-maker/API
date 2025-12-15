# Assistente CAPEX/OPEX — Backend & Frontend (setup rapido)

Questa repo contiene una demo di backend FastAPI + frontend statico per supportare il flusso:
- /phase1_sintesi → sintesi interventi via modello IA
- /train → carica esempi etichettati (CAPEX/OPEX) nello storico locale (training_data.json)
- /full_classify → usa storico + IA per classificare interventi e fornire motivazione/confidenza
- Chat: /chat_train e /chat_answer

Installazione rapida (locale):
1. git checkout fix/refactor-backend
2. python -m venv .venv
3. .\\.venv\\Scripts\\Activate.ps1
4. pip install -r requirements.txt
5. uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Nota sulla API key:
- In produzione imposta GEMINI_API_KEY come variabile d'ambiente sul server.
- Per test locali puoi impostare $env:GEMINI_API_KEY in PowerShell.
