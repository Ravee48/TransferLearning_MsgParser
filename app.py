# app.py  — ONNX Runtime + Tokenizers-only FastAPI server
import os
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
import asyncio

# ---------------- Config (env-overridable) ----------------
ONNX_PATH = os.getenv("ONNX_PATH", "./onnx_export/model.int8.onnx")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "./tokenizer")
TOKENIZER_JSON = os.getenv("TOKENIZER_JSON", f"{TOKENIZER_DIR}/tokenizer.json")
MAX_LEN = int(os.getenv("MAX_LEN", "128"))          # 96–128 is a sweet spot for speed
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))    # 1–2 on small boxes
BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "1024")) # safety upper bound
# For large batches, allow more threads for ONNX and numpy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

print(f"Number of threads set for ONNX and numpy: {NUM_THREADS}")

# ---------------- App ----------------
app = FastAPI(title="ONNX Text Classifier", version="1.0")

# ---------------- Load tokenizer (HF tokenizers) ----------------
try:
    tok = Tokenizer.from_file(TOKENIZER_JSON)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer at {TOKENIZER_JSON}: {e}")

# enable CLS/SEP via post-processor already stored in tokenizer.json
# padding/truncation
pad_id = tok.token_to_id("[PAD]")
if pad_id is None:
    pad_id = 0  # many BERT-family tokenizers use 0 for [PAD]
tok.enable_truncation(max_length=MAX_LEN)
tok.enable_padding(length=MAX_LEN, pad_id=pad_id, pad_token="[PAD]")

# ---------------- Load ONNX model ----------------
so = ort.SessionOptions()
so.intra_op_num_threads = NUM_THREADS
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])

# Discover optional inputs (e.g., token_type_ids)
session_inputs = {i.name for i in sess.get_inputs()}  # {'input_ids','attention_mask',...}


# ---------------- I/O schemas ----------------
class PredictIn(BaseModel):
    texts: List[str]                 # send up to 500 (or more) per call
    prob_threshold: Optional[float] = None  # binary only; default 0.5

class PredictWithIdsIn(BaseModel):
    ids: List[str]
    texts: List[str]
    prob_threshold: Optional[float] = None

# New schema for msgs input
class MsgItem(BaseModel):
    id: str
    message: str

class MsgsIn(BaseModel):
    msgs: List[MsgItem]
    prob_threshold: Optional[float] = None

# ---------------- Helpers ----------------
def _prepare_batch(texts: List[str]):
    if not isinstance(texts, list) or not texts:
        raise HTTPException(status_code=400, detail="`texts` must be a non-empty list of strings.")
    if len(texts) > BATCH_LIMIT:
        raise HTTPException(status_code=413, detail=f"Batch too large (> {BATCH_LIMIT}).")
    # Use numpy array allocation for speed
    encs = tok.encode_batch([t if isinstance(t, str) else str(t) for t in texts])
    input_ids = np.empty((len(encs), MAX_LEN), dtype=np.int64)
    attention_mask = np.empty((len(encs), MAX_LEN), dtype=np.int64)
    for i, e in enumerate(encs):
        input_ids[i, :] = e.ids
        attention_mask[i, :] = e.attention_mask
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in session_inputs:
        ort_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
    return ort_inputs

def _postprocess_logits(logits: np.ndarray, prob_threshold: Optional[float]):
    """
    Works for binary (shape [N,1] or [N,2]) and multi-class (shape [N,C]).
    Returns labels (ints) and probs (float or list of floats).
    """
    if logits.ndim != 2:
        raise HTTPException(status_code=500, detail="Unexpected logits shape.")
    N, C = logits.shape
    # Binary with single logit
    if C == 1:
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        thr = 0.5 if prob_threshold is None else float(prob_threshold)
        labels = (probs > thr).astype(int)
        return labels.tolist(), probs.tolist()
    # Multi-class (or binary with two logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    probs = e / e.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1)
    # For binary C==2, return prob of class 1; for multi-class, return full distribution
    if probs.shape[1] == 2:
        return labels.tolist(), probs[:, 1].tolist()
    else:
        return labels.tolist(), probs.tolist()


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "onnx": os.path.basename(ONNX_PATH), "max_len": MAX_LEN}

# New endpoint for your input format
@app.post("/predict")
async def predict(inp: MsgsIn):
    texts = [item.message for item in inp.msgs]
    ids = [item.id for item in inp.msgs]
    ort_inputs = _prepare_batch(texts)
    loop = asyncio.get_event_loop()
    logits = await loop.run_in_executor(None, lambda: sess.run(None, ort_inputs)[0])
    labels, probs = _postprocess_logits(logits, inp.prob_threshold)
    result = {}
    for i, l, p in zip(ids, labels, probs):
        label = "Transactional" if l == 1 else "Non-Transactional"
        result[i] = {"prediction": float(p), "result": label}
    return result

# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
