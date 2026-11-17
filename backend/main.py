"""Private, research-only inference API. Never invent a result if weights are absent."""
import io
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from ml.models import build, NAMES
from ml.preprocessing import transform
from ml.gradcam import explain

Image.MAX_IMAGE_PIXELS = 16_000_000
MAX_BYTES = 10 * 1024 * 1024
DISCLAIMER = 'Research prototype. Not validated for diagnosis or clinical decisions.'
WEIGHTS = Path(os.getenv('WEIGHTS_DIR', 'ml/weights'))
METRICS = Path(os.getenv('METRICS_FILE', 'outputs/evaluation/metrics.json'))
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///./deepcancer.db'), pool_pre_ping=True)
class Base(DeclarativeBase):
    pass
class Record(Base):
    __tablename__ = 'predictions'
    id = Column(String, primary_key=True)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    architecture = Column(String, nullable=False)
    magnification = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
Base.metadata.create_all(engine)

class Prediction(BaseModel):
    id: str
    prediction: Literal['benign', 'malignant']
    confidence: float = Field(ge=0, le=1)
    model: str
    explanation: dict[str, str]
    disclaimer: str = DISCLAIMER

app = FastAPI(title='DeepCancer AI', description=DISCLAIMER)
origins = [s.strip() for s in os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',') if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
lock = threading.Lock()
cache = {}

@app.get('/health')
def health():
    return {'status': 'ok', 'research_only': True, 'models': {k: (WEIGHTS / (v + '.pt')).exists() for k, v in NAMES.items()}}

@app.get('/metrics')
def metrics():
    if not METRICS.exists():
        return {'models': [], 'status': 'not_evaluated'}
    return json.loads(METRICS.read_text())

@app.post('/predict', response_model=Prediction)
def predict(file: UploadFile = File(...), model: str = Form('ResNet50'), magnification: str = Form('Unknown')):
    if model not in NAMES:
        raise HTTPException(422, 'Unsupported architecture')
    if magnification not in ['40×', '100×', '200×', '400×', 'Unknown']:
        raise HTTPException(422, 'Unsupported magnification')
    if file.content_type not in ['image/png', 'image/jpeg']:
        raise HTTPException(415, 'Only PNG and JPEG images are accepted')
    data = file.file.read(MAX_BYTES + 1)
    file.file.close()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image exceeds 10 MB')
    try:
        with Image.open(io.BytesIO(data)) as raw:
            if raw.format not in ['PNG', 'JPEG'] or min(raw.size) < 32 or raw.width * raw.height > 16_000_000:
                raise ValueError()
            raw.verify()
        with Image.open(io.BytesIO(data)) as raw:
            image = raw.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(422, 'Invalid, oversized or unsupported image')
    path = WEIGHTS / (NAMES[model] + '.pt')
    if not path.exists():
        raise HTTPException(503, 'Trained weights unavailable. Prepare BreakHis and train this architecture first.')
    # Serialise gradient hooks: requests must not share active Grad-CAM hooks.
    with lock:
        if model not in cache:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint.get('classes') != ['benign', 'malignant'] or checkpoint.get('architecture') != NAMES[model]:
                raise HTTPException(503, 'Checkpoint metadata mismatch')
            net, target = build(NAMES[model])
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            cache[model] = (net, target)
        net, target = cache[model]
        cls, probs, explanation = explain(net, target, transform()(image).unsqueeze(0), image.resize((700, 460)))
    identity = str(uuid.uuid4())
    label = ['benign', 'malignant'][cls]
    with Session(engine) as session:
        session.add(Record(id=identity, prediction=label, confidence=probs[cls], architecture=model, magnification=magnification, created_at=datetime.now(timezone.utc)))
        session.commit()
    return Prediction(id=identity, prediction=label, confidence=probs[cls], model=model, explanation=explanation)
