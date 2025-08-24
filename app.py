
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
import os


app = FastAPI()

# Define the path to the saved model and tokenizer
MODEL_DIR = "./model"
MAX_LEN = 128  # Define the maximum sequence length for padding

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None
tokenizer = None


class PredictionOut(BaseModel):
    prediction: float
    result: str


def load_model_and_tokenizer():
    global model, tokenizer
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        print("Model loaded successfully!")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        print("Tokenizer loaded successfully!")



def batch_predict(messages: List[str], max_length: int = 128) -> Tuple[List[int], List[float]]:
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded")
    inputs = tokenizer(
        messages,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        prob_tx = probs[:, 1]
    return [int(p.item()) for p in preds], [float(s.item()) for s in prob_tx]



def predict_messages_with_details(input_dict):
    load_model_and_tokenizer()  # Load model and tokenizer if not already loaded
    payload = input_dict
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Input payload must be a dictionary")
    ids: List[str] = []
    messages: List[str] = []
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
            messages.append(str(item.get("message", "")))
    else:
        # Back-compat: accept flat mapping {id: message}
        if len(payload) == 0:
            return {}
        for k, v in payload.items():
            ids.append(str(k))
            messages.append(str(v) if v is not None else "")
    if len(messages) == 0:
        return {}
    preds, scores = batch_predict(messages)
    result: Dict[str, PredictionOut] = {}
    for k, pred, score in zip(ids, preds, scores):
        label = "Transactional" if pred == 1 else "Non-Transactional"
        result[k] = PredictionOut(prediction=score, result=label)
    return result



# Flask route removed. Only FastAPI is used.


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    predictions = predict_messages_with_details(data)
    predictions_dict = {k: v.dict() for k, v in predictions.items()}
    return JSONResponse(content=predictions_dict)


# ...existing code...



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )