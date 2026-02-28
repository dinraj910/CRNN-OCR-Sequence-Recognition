# 🔤 CRNN OCR Engine

> **Project #11** — Neural OCR · CNN + BiLSTM + CTC Loss · FastAPI + Bootstrap

Built by **Dinraj K Dinesh** · [dinrajkdinesh.vercel.app](https://dinrajkdinesh.vercel.app)

---

## Architecture

```
Input (32×128×1)
  → CNN Encoder     [6 VGG-style blocks · asymmetric pooling]
  → Reshape         [spatial (1×32×512) → temporal (32 steps × 512 feat)]
  → BiLSTM × 2     [256 units · forward + backward context]
  → Dense + Softmax [37 classes = a–z + 0–9 + CTC blank]
  → CTC Beam Search [width = 10]
  → Text Output
```

Trained on **IIIT5K-Words** (~5,000 word images).

---

## Project Structure

```
crnn_ocr_app/
├── main.py                  # FastAPI application
├── model/
│   ├── predictor.py         # Inference engine
│   ├── DINRAJ_CRNN_OCR.keras   # ← copy from Colab
│   └── vocab_config.json    # ← copy from Colab
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── requirements.txt
└── Dockerfile
```

---

## Setup & Run

### 1. Copy model files from Colab

```bash
# In your Colab notebook after training:
# Download DINRAJ_CRNN_OCR.keras and vocab_config.json
# Then place them in crnn_ocr_app/model/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
cd crnn_ocr_app
python main.py
# Open: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Docker

```bash
docker build -t crnn-ocr .
docker run -p 8000:8000 crnn-ocr
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ocr` | OCR from file upload |
| `POST` | `/api/ocr/base64` | OCR from base64 image |
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/stats` | Inference statistics |
| `GET`  | `/docs` | Auto-generated OpenAPI UI |

### Quick test

```bash
curl -X POST http://localhost:8000/api/ocr -F "file=@word.png"
```

---

## Resume Highlights

```
• Built end-to-end CRNN OCR (CNN + BiLSTM + CTC) on IIIT5K, deployed via FastAPI
• Designed morphological word-region detector replacing MSER for word-level bounding boxes
• Implemented beam search (width=10) CTC decoder + CER/WER evaluation pipeline
• Created production REST API with /docs, health, stats, and base64 endpoints
• Built responsive dark/light-mode web UI with drag-drop upload and real-time inference
• Containerized with Docker for one-command deployment
```

---

*12-Month AI/ML Roadmap · MCA → ₹20–40 LPA*
