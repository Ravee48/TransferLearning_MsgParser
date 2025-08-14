import os
from typing import Any, Dict, List, Tuple

import torch
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


app = FastAPI(title="Transaction Message Classifier", version="1.0.0")

# Load environment variables from a local .env file if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


tokenizer: AutoTokenizer | None = None
model: AutoModelForSequenceClassification | None = None
device = torch.device("cpu")


def resolve_model_dir() -> str:
    env_path = os.environ.get("MODEL_DIR")
    force_fetch = os.environ.get("MODEL_FORCE_FETCH", "0").lower() in {"1", "true", "yes"}
    if not force_fetch and env_path and os.path.isdir(env_path):
        return env_path

    candidates: List[str] = [
        "model",
    ]

    if not force_fetch:
        for path in candidates:
            if os.path.isdir(path):
                has_config = os.path.isfile(os.path.join(path, "config.json"))
                has_weights = os.path.isfile(os.path.join(path, "model.safetensors")) or os.path.isfile(
                    os.path.join(path, "pytorch_model.bin")
                )
                if has_config and has_weights:
                    return path

    # If not found locally, attempt to fetch using ModelFetcher
    try:
        from model_fetcher import ModelFetcher  # local utility

        target = os.environ.get("MODEL_DIR", "model")
        os.makedirs(target, exist_ok=True)
        # Default to the provided Google Drive share URL if no explicit envs are set
        default_gdrive_url = (
            os.environ.get("MODEL_ZIP_URL")
            or os.environ.get("GDRIVE_FILE_URL")
            or "https://drive.google.com/file/d/1BaGYUisX0s-LmCOH9FegsBF2YITFlcqQ/view?usp=sharing"
        )
        fetcher = ModelFetcher(model_zip_url=default_gdrive_url, gdrive_file_id=os.environ.get("GDRIVE_FILE_ID"))
        fetcher.ensure_model_present(target)
        if os.path.isdir(target) and (
            os.path.isfile(os.path.join(target, "config.json"))
            or os.path.isfile(os.path.join(target, "model.safetensors"))
            or os.path.isfile(os.path.join(target, "pytorch_model.bin"))
        ):
            return target
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to fetch model into '{target}'. Reason: {e}. If using Google Drive, ensure the file is shared as 'Anyone with the link' and is a ZIP of the model folder."
        )

    raise FileNotFoundError(
        "Could not locate a valid model directory. Set env MODEL_DIR to the path containing config.json/model files."
    )


def load_model_and_tokenizer() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    model_dir = resolve_model_dir()
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    mdl.eval()
    return tok, mdl


# ----------------------------- Schemas -----------------------------
class PredictionOut(BaseModel):
    prediction: float
    result: str


@app.on_event("startup")
def _startup_load_once() -> None:
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer, model = load_model_and_tokenizer()


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": (
            "POST either {id: message} or {\"msgs\":[{\"id\":..., \"message\":...}, ...]} to /predict. "
            "Response maps ids to {prediction: <prob_of_transaction>, result: 'Transactional'|'Non-Transactional'}."
        ),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def batch_predict(messages: List[str], max_length: int = 192) -> Tuple[List[int], List[float]]:
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")

    inputs = tokenizer(
        messages,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        prob_tx = probs[:, 1]
    return [int(p.item()) for p in preds], [float(s.item()) for s in prob_tx]


@app.post("/predict", response_model=Dict[str, PredictionOut])
def predict(
    payload: Dict[str, Any] = Body(
        ...,
        examples={
            "default": {
                "summary": "Batch messages",
                "value": {
                    "msgs": [
                        {"id": "1", "message": "USD 23.6 spent on Kotak Bank Card ..."},
                        {"id": "2", "message": "Auto-Pay activated on Kotak Card ..."},
                    ]
                },
            }
        },
    )
) -> Dict[str, PredictionOut]:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                "Body must be an object: either {id: message} or {\"msgs\": [{\"id\":..., \"message\":...}]}"
            ),
        )

    ids: List[str] = []
    texts: List[str] = []

    if "msgs" in payload:
        msgs = payload.get("msgs")
        if msgs is None:
            return {}
        if not isinstance(msgs, list):
            raise HTTPException(status_code=400, detail="'msgs' must be a list of objects with 'id' and 'message'")
        for item in msgs:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Each entry in 'msgs' must be an object")
            if "id" not in item or "message" not in item:
                raise HTTPException(status_code=400, detail="Each message must include 'id' and 'message'")
            ids.append(str(item.get("id")))
            texts.append(str(item.get("message", "")))
    else:
        # Back-compat: accept flat mapping {id: message}
        if len(payload) == 0:
            return {}
        for k, v in payload.items():
            ids.append(str(k))
            texts.append(str(v) if v is not None else "")

    if len(texts) == 0:
        return {}

    preds, scores = batch_predict(texts)
    id_to_obj: Dict[str, PredictionOut] = {}
    for k, pred, score in zip(ids, preds, scores):
        label = "Transactional" if pred == 1 else "Non-Transactional"
        id_to_obj[k] = PredictionOut(prediction=score, result=label)
    return id_to_obj


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


