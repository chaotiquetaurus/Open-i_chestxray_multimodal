from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import torch
import torch.nn as nn
import io
import os

# ── Sécurité ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "poc-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")

# ── Rate limiting ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Classes — ordre alphabétique exact de ton dataset ─────────────────────
CLASSES = [
    "Atelectasis",
    "Calcified Granuloma",
    "Cardiomegaly",
    "Edema",
    "Effusion",
    "Emphysema",
    "Granuloma",
    "Infiltration",
    "Lung Opacity",
    "Nodule",
    "Normal",
    "Pneumonia",
]

# ── Chargement du modèle au démarrage ─────────────────────────────────────
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    backbone = efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(CLASSES)),
    )

    # Le checkpoint a été sauvegardé depuis un wrapper (self.backbone = efficientnet)
    # → on supprime le préfixe "backbone." et on ignore "loss_fn.weight"
    checkpoint = torch.load("model_full.pth", map_location="cpu")
    fixed_state_dict = {
        k[len("backbone."):]: v
        for k, v in checkpoint.items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(fixed_state_dict)

    backbone.eval()
    model_store["model"] = backbone
    print(f"✅ Modèle EfficientNet-B0 chargé — {len(CLASSES)} classes")
    yield
    model_store.clear()

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Image API",
    description="API de reconnaissance d'imagerie médicale — EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Transformations — identiques au val_transform de l'entraînement ───────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in model_store,
        "classes": CLASSES,
    }

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    _: str = Depends(verify_key),
):
    # Vérifie le format
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Format accepté : JPG, PNG uniquement")

    # Lecture en mémoire (jamais sur disque)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Fichier vide")

    # Limite la taille à 10 Mo
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Fichier trop volumineux (max 10 Mo)")

    # Inférence
    img = Image.open(io.BytesIO(content)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model_store["model"](tensor)
        probs = torch.softmax(output, dim=-1)[0]

    top_idx = probs.argmax().item()

    # Top 3 prédictions
    top3 = sorted(
        [{"label": CLASSES[i], "confidence": round(probs[i].item(), 4)}
         for i in range(len(CLASSES))],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:3]

    return {
        "prediction": CLASSES[top_idx],
        "confidence": round(probs[top_idx].item(), 4),
        "top3": top3,
        "all_scores": {
            CLASSES[i]: round(probs[i].item(), 4)
            for i in range(len(CLASSES))
        },
    }