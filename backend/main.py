import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import io

MODEL_REPO = os.getenv("MODEL_REPO", "Imagers/Imager-1")
DATABASE_URL = os.getenv("DATABASE_URL", "")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "10"))
# Set this to the exact public website origin, e.g. https://your-site.pages.dev
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
MODEL = None
PROCESSOR = None

def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured. Attach a Railway PostgreSQL database and set DATABASE_URL.")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=8)

def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imager_daily_usage (
                day_utc DATE NOT NULL,
                visitor_hash TEXT NOT NULL,
                ip_hash TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (day_utc, visitor_hash, ip_hash)
            )
        """)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, PROCESSOR
    init_db()
    # Load model when server starts. If the model's architecture is not supported
    # by AutoModelForImageClassification, startup logs will show the model-loading error.
    PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_REPO)
    MODEL = AutoModelForImageClassification.from_pretrained(MODEL_REPO)
    MODEL.eval()
    yield

app = FastAPI(title="Imager-1 API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def hash_value(value: str) -> str:
    secret = os.getenv("RATE_LIMIT_SECRET", "")
    if not secret:
        raise RuntimeError("RATE_LIMIT_SECRET is not configured.")
    return hashlib.sha256((secret + ":" + value).encode("utf-8")).hexdigest()

def client_ip(request: Request) -> str:
    # Railway/proxies set X-Forwarded-For. Only the first address is used.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:100]
    return request.client.host if request.client else "unknown"

def consume_daily_quota(visitor_id: str, ip: str):
    if not visitor_id or len(visitor_id) > 200:
        raise HTTPException(status_code=400, detail="A valid visitor ID is required.")
    visitor_hash = hash_value(visitor_id)
    ip_hash = hash_value(ip)
    today = datetime.now(timezone.utc).date()
    with db_connect() as conn:
        row = conn.execute("""
            SELECT request_count FROM imager_daily_usage
            WHERE day_utc=%s AND visitor_hash=%s AND ip_hash=%s
            FOR UPDATE
        """, (today, visitor_hash, ip_hash)).fetchone()
        if row and row["request_count"] >= DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached: {DAILY_LIMIT} analyses per visitor. Please try again tomorrow (UTC).",
                headers={"Retry-After": str(int((datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds()))}
            )
        conn.execute("""
            INSERT INTO imager_daily_usage (day_utc, visitor_hash, ip_hash, request_count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (day_utc, visitor_hash, ip_hash)
            DO UPDATE SET request_count=imager_daily_usage.request_count + 1, updated_at=NOW()
        """, (today, visitor_hash, ip_hash))

@app.get("/")
def home():
    return {"name": "Imager-1 API", "endpoint": "/predict", "daily_limit": DAILY_LIMIT}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), visitor_id: str = Form(...)):
    if MODEL is None or PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")
    content = await file.read(MAX_IMAGE_MB * 1024 * 1024 + 1)
    if len(content) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image must be {MAX_IMAGE_MB} MB or smaller.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    # Count only valid images that proceed to inference.
    try:
        consume_daily_quota(visitor_id, client_ip(request))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Usage limiter unavailable: {exc}")

    try:
        inputs = PROCESSOR(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = MODEL(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        labels = MODEL.config.id2label or {}
        scored = []
        for i, score in enumerate(probs):
            label = str(labels.get(i, labels.get(str(i), f"class_{i}")))
            scored.append((label, float(score)))
        # Prefer label semantics if the model labels are clear; otherwise use the top score.
        ai_entry = next((x for x in scored if any(k in x[0].lower() for k in ("ai", "fake", "generated", "synthetic"))), None)
        real_entry = next((x for x in scored if any(k in x[0].lower() for k in ("real", "photo", "authentic", "human"))), None)
        if ai_entry and real_entry:
            ai_score, real_score = ai_entry[1] * 100, real_entry[1] * 100
            is_ai = ai_score >= real_score
            confidence = max(ai_score, real_score)
        else:
            label, score = max(scored, key=lambda x: x[1])
            is_ai = any(k in label.lower() for k in ("ai", "fake", "generated", "synthetic"))
            confidence = score * 100
            ai_score = confidence if is_ai else 100 - confidence
            real_score = 100 - ai_score
            label = "AI-Generated" if is_ai else "Real Image"
        return {
            "model_repo": MODEL_REPO,
            "label": "AI-Generated" if is_ai else "Real Image",
            "isAI": bool(is_ai),
            "confidence": round(confidence, 2),
            "aiScore": round(ai_score, 2),
            "realScore": round(real_score, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}: {exc}")
