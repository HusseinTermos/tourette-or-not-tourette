
import io, json, numpy as np, base64
from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel

import torch, torch.nn as nn
import tensorflow_hub as hub
import soundfile as sf  # read wav/flac/ogg/aiff from bytes
import librosa

SR = 16000

class TicClassifier(nn.Module):
    def __init__(self, yamnet_handle: str, sr: int = SR, dropout: float = 0.2):
        super().__init__()
        self.yamnet = hub.load(yamnet_handle)  # TF-Hub (CPU)
        self.sample_rate = sr
        hidden = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    @torch.no_grad()
    def _yamnet_embed(self, wav_np: np.ndarray) -> np.ndarray:
        _, embeddings, _ = self.yamnet(wav_np.astype(np.float32))
        return embeddings.numpy()

    @torch.no_grad()
    def forward(self, wave: np.ndarray, max_len_s: float | None = 2.0):
        if max_len_s is not None:
            L = int(self.sample_rate * max_len_s)
            if wave.shape[0] > L:
                s = (wave.shape[0] - L) // 2
                wave = wave[s:s+L]
        emb = self._yamnet_embed(wave)                      # [frames,1024]
        feat = emb.mean(axis=0).astype(np.float32)[None]    # [1,1024]
        logit = self.classifier(torch.from_numpy(feat)).squeeze(1)  # [1]
        return float(logit.item())

app = FastAPI()
MODEL = None
CFG = None

class PredictResp(BaseModel):
    prob: float
    pred: int

@app.on_event("startup")
def load_model():
    global MODEL, CFG
    with open("artifacts/model_config.json") as f:
        CFG = json.load(f)
    MODEL = TicClassifier(
        yamnet_handle=CFG["yamnet_handle"],
        sr=CFG.get("sr", SR),
        dropout=CFG.get("dropout", 0.2),
    )
    state = torch.load("artifacts/classifier_head.pt", map_location="cpu")
    MODEL.classifier.load_state_dict(state)
    MODEL.eval()

@app.post("/predict", response_model=PredictResp)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != CFG["sr"]:
        # quick resample via librosa (installed via requirements)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CFG["sr"])
    logit = MODEL(audio, max_len_s=CFG.get("max_len_s", 2.0))
    prob = float(torch.sigmoid(torch.tensor([logit]))[0].item())
    pred = int(prob >= CFG.get("threshold", 0.5))
    return PredictResp(prob=prob, pred=pred)

class PredictBase64Req(BaseModel):
    audio_b64: str  # base64 string

@app.post("/predict_b64", response_model=PredictResp)
async def predict_b64(req: PredictBase64Req = Body(...)):
    # Decode base64 string into raw bytes
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio: {e}")

    # Load waveform
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    target_sr = CFG.get("sr", SR)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Run through model
    logit = MODEL(audio, max_len_s=CFG.get("max_len_s", 2.0))
    prob = float(torch.sigmoid(torch.tensor([logit]))[0].item())
    pred = int(prob >= CFG.get("threshold", 0.5))
    return PredictResp(prob=prob, pred=pred)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
