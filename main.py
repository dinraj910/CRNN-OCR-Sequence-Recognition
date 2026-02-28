"""
CRNN OCR API — FastAPI Backend
Author : Dinraj K Dinesh
Project: #11 — CNN + BiLSTM + CTC Loss OCR System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import time
import os
import io
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from model.predictor import CRNNPredictor

# ─────────────────────────────────────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRNN OCR Engine",
    description="""
## 🔤 Neural OCR — CRNN + CTC Loss

A production-grade Optical Character Recognition system built with:
- **CNN Encoder**: 6-block VGG-style feature extractor with asymmetric pooling
- **BiLSTM Decoder**: 2-layer bidirectional LSTM for sequence modeling
- **CTC Loss**: Connectionist Temporal Classification for alignment-free training
- **Beam Search**: Width-10 beam search decoding for highest accuracy

Trained on **IIIT5K-Words** dataset (~5,000 word images).

### Key Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /api/ocr` | Perform OCR on uploaded image |
| `POST /api/ocr/base64` | OCR via base64-encoded image |
| `GET /api/health` | Service health check |
| `GET /api/stats` | Inference statistics |
""",
    version="1.0.0",
    contact={
        "name": "Dinraj K Dinesh",
        "url": "https://dinrajkdinesh.vercel.app",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Static Files & Templates
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

predictor: Optional[CRNNPredictor] = None
inference_stats = {
    "total_requests": 0,
    "total_chars_recognized": 0,
    "avg_inference_ms": 0.0,
    "model_loaded": False,
    "model_name": "CRNN (CNN + BiLSTM + CTC)",
    "dataset": "IIIT5K-Words",
    "started_at": datetime.now().isoformat(),
}


@app.on_event("startup")
async def load_model():
    global predictor
    try:
        vocab_path = BASE_DIR / "model" / "vocab_config.json"
        model_path = BASE_DIR / "model" / "DINRAJ_CRNN_OCR.keras"

        if not vocab_path.exists():
            print("⚠️  vocab_config.json not found — using demo mode")
            inference_stats["model_loaded"] = False
            return

        predictor = CRNNPredictor(
            model_path=str(model_path),
            vocab_path=str(vocab_path),
        )
        inference_stats["model_loaded"] = True
        print("✅ CRNN model loaded successfully")

    except Exception as e:
        print(f"❌ Model load failed: {e}")
        inference_stats["model_loaded"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Routes — UI
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ─────────────────────────────────────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {
        "status": "healthy" if inference_stats["model_loaded"] else "demo_mode",
        "model_loaded": inference_stats["model_loaded"],
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/stats")
async def stats():
    return inference_stats


@app.post("/api/ocr")
async def ocr_from_upload(file: UploadFile = File(...)):
    """
    Perform OCR on an uploaded image file.

    - **file**: Image file (PNG, JPG, JPEG, BMP, TIFF, WebP)

    Returns detected text regions with bounding boxes,
    the annotated image as base64, and performance metrics.
    """
    # Validate file type
    allowed = {"image/png", "image/jpeg", "image/jpg",
               "image/bmp", "image/tiff", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PNG, JPG, BMP, TIFF, WebP"
        )

    # Read image bytes
    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    return await _run_inference(img_bytes, file.filename)


@app.post("/api/ocr/base64")
async def ocr_from_base64(payload: dict):
    """
    Perform OCR on a base64-encoded image.

    Body: `{ "image": "<base64_string>", "filename": "optional_name.png" }`
    """
    if "image" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    try:
        # Strip data URI prefix if present
        b64_data = payload["image"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    filename = payload.get("filename", "upload.png")
    return await _run_inference(img_bytes, filename)


async def _run_inference(img_bytes: bytes, filename: str) -> dict:
    """Core inference pipeline — shared by both OCR endpoints."""
    global inference_stats

    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    try:
        import numpy as np
        import cv2

        # Decode image
        nparr   = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h_orig, w_orig = img_bgr.shape[:2]

        # Run predictor (or demo mode)
        if predictor is not None and inference_stats["model_loaded"]:
            result = predictor.predict(img_bgr)
        else:
            # Demo mode — return mock result so UI is testable without model
            result = _demo_result(img_bgr)

        t_end   = time.perf_counter()
        inf_ms  = round((t_end - t_start) * 1000, 2)

        # Update stats
        inference_stats["total_requests"] += 1
        n = inference_stats["total_requests"]
        inference_stats["avg_inference_ms"] = round(
            ((n - 1) * inference_stats["avg_inference_ms"] + inf_ms) / n, 2
        )
        inference_stats["total_chars_recognized"] += sum(
            len(r["text"]) for r in result["regions"]
        )

        # Encode annotated image as base64
        _, buf = cv2.imencode(".png", result["annotated_bgr"])
        annotated_b64 = base64.b64encode(buf).decode("utf-8")

        # Also encode original as base64 for display
        _, buf2 = cv2.imencode(".png", img_bgr)
        original_b64 = base64.b64encode(buf2).decode("utf-8")

        return {
            "request_id"   : request_id,
            "filename"     : filename,
            "timestamp"    : datetime.now().isoformat(),
            "image_size"   : {"width": w_orig, "height": h_orig},
            "regions"      : result["regions"],
            "full_text"    : result["full_text"],
            "region_count" : len(result["regions"]),
            "inference_ms" : inf_ms,
            "original_b64" : f"data:image/png;base64,{original_b64}",
            "annotated_b64": f"data:image/png;base64,{annotated_b64}",
            "model_info"   : {
                "name"   : "CRNN (CNN + BiLSTM + CTC)",
                "dataset": "IIIT5K-Words",
                "version": "1.0.0",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


def _demo_result(img_bgr):
    """Demo mode result when model file is not present."""
    import numpy as np
    import cv2

    annotated = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    cv2.rectangle(annotated, (10, 10), (W-10, H-10), (0, 200, 100), 3)
    label = " demo "
    cv2.rectangle(annotated, (10, 0), (100, 28), (0, 200, 100), -1)
    cv2.putText(annotated, label, (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return {
        "regions"      : [{"text": "demo", "bbox": [10, 10, W-10, H-10], "confidence": 0.95}],
        "full_text"    : "demo",
        "annotated_bgr": annotated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
