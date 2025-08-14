## Transaction Message Classifier API

FastAPI service that loads a fine-tuned Hugging Face text classifier once and exposes an HTTP API to classify SMS/email messages as transactional or non-transactional.

### Features
- Loads model and tokenizer once at startup
- Batch inference
- Two request formats supported
- Returns probability of transaction and a human-readable label

### Requirements
- Python 3.10+
- pip

Install dependencies:
```bash
pip install fastapi uvicorn transformers torch
```

### Minimal model folder for inference
Place your model files in a folder (default: `model/`). Only the following are required:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

Optional (tiny):
- `label_mapping.json`

You can point the app to a different directory via `MODEL_DIR`.

### Run the API
Using Python directly:
```bash
# Optionally set your model directory (if not using ./model)
export MODEL_DIR=/absolute/path/to/your/model

python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

### Request/Response
Endpoint: `POST /predict`

Accepts either of the following request bodies.

1) Preferred array form:
```json
{
  "msgs": [
    {"id": "1", "message": "USD 23.6 spent on Kotak Bank Card ..."},
    {"id": "2", "message": "Auto-Pay activated on Kotak Card ..."}
  ]
}
```

2) Backward-compatible mapping form:
```json
{
  "1": "USD 23.6 spent on Kotak Bank Card ...",
  "2": "Auto-Pay activated on Kotak Card ..."
}
```

Response (for both forms):
```json
{
  "1": { "prediction": 0.999997615814209, "result": "Transactional" },
  "2": { "prediction": 0.01110817026346922, "result": "Non-Transactional" }
}
```

Example curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "msgs": [
      {"id": "1", "message": "USD 23.6 spent on Kotak Bank Card x7822 on 04-MAR-2025 ..."},
      {"id": "2", "message": "Auto-Pay activated on Kotak Card x7822 for OpenAILLC ..."}
    ]
  }'
```

### Configuration
- `MODEL_DIR`: absolute/relative path to the model directory. If unset, the app looks for `./model`.
- Max sequence length is set to 192 in `app.py`.

### Production notes
- Use the minimal model folder listed above to keep your Docker image or deployment small.
- Run with a production server (e.g., `uvicorn` + process manager or `gunicorn -k uvicorn.workers.UvicornWorker`).
- Ensure the model directory is mounted/readable by the service.

### Repository structure (key files)
```
app.py                  # FastAPI app
README.md               # This file
train.ipynb             # Training experiments (not required for serving)
test.ipynb              # Local testing (not required for serving)
model/                  # Minimal model folder for inference (see above)
```


