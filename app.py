from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer

# =========================
# 🚀 INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 📂 BASE DIR (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 📂 SERVE STATIC FILES (CSS)
# =========================
app.mount(
    "/frontend",
    StaticFiles(directory=os.path.join(BASE_DIR, "frontend")),
    name="frontend"
)

# =========================
# 📂 LOAD MODEL
# =========================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "emotion_distilbert_amp_final",
    "best_model"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📦 Loading model...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}

print("✅ Model loaded!")

# =========================
# 📥 REQUEST MODEL
# =========================
class TextInput(BaseModel):
    text: str

# =========================
# 🔮 PREDICT FUNCTION
# =========================
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))

    return ID2LABEL[pred_id], probs

# =========================
# 🌐 SERVE HTML
# =========================
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

# =========================
# 🔌 API ENDPOINT
# =========================
@app.post("/predict")
def predict_api(input: TextInput):
    pred, probs = predict(input.text)

    return {
        "prediction": pred,
        "confidence": float(np.max(probs)),
        "probabilities": {
            ID2LABEL[i]: float(probs[i]) for i in range(len(probs))
        },
        "source": "model"
    }

# =========================
# ▶️ RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)