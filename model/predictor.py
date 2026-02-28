"""
CRNN OCR Predictor — Inference Engine
Compatible with TensorFlow 2.20 + Keras 3.13
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


class CRNNPredictor:
    IMG_HEIGHT = 32
    IMG_WIDTH  = 128

    COLORS = [
        (100, 220, 50),
        (255, 180, 30),
        (50, 180, 255),
        (180, 80, 255),
        (50, 230, 200),
        (255, 100, 100),
    ]

    def __init__(self, model_path: str, vocab_path: str):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self._model     = None
        self._vocab     = None
        self._load()

    def _load(self):
        import keras

        with open(self.vocab_path, "r") as f:
            self._vocab = json.load(f)

        self._idx_to_char   = {
            int(k): v for k, v in self._vocab["idx_to_char"].items()
        }
        self._num_timesteps = self._vocab.get("num_timesteps", 32)

        model_path = Path(self.model_path)
        h5_path    = model_path.with_suffix(".h5")

        # Prefer .keras; fall back to .h5 if not present
        if model_path.exists():
            load_path = str(model_path)
        elif h5_path.exists():
            load_path = str(h5_path)
            print(f"⚠️  Using .h5 fallback: {h5_path.name}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._model = keras.models.load_model(load_path, compile=False)
        print(f"✅ Model loaded: {Path(load_path).name}")
        self._keras    = keras
        # Blank token is always the last class (CTC convention)
        self._blank_idx = self._vocab.get("num_classes", 37) - 1

    def _preprocess_crop(self, img_bgr: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        h, w  = gray.shape
        ratio = self.IMG_HEIGHT / h
        new_w = min(int(w * ratio), self.IMG_WIDTH)
        resized = cv2.resize(gray, (new_w, self.IMG_HEIGHT),
                             interpolation=cv2.INTER_CUBIC)
        padded  = np.pad(resized,
                         ((0, 0), (0, self.IMG_WIDTH - new_w)),
                         mode="constant", constant_values=255)
        norm = padded.astype(np.float32) / 255.0
        return np.expand_dims(norm, axis=-1)

    def _decode(self, y_pred: np.ndarray, beam_width: int = 10) -> List[str]:
        # keras.ops.ctc_decode: returns (top_paths, batch, T)
        # -1 = padding, blank_idx = CTC blank — both must be excluded
        batch_size = y_pred.shape[0]
        seq_lens   = np.full(batch_size, self._num_timesteps, dtype=np.int32)

        decoded, _ = self._keras.ops.ctc_decode(
            y_pred,
            sequence_lengths=seq_lens,
            strategy="beam_search",
            beam_width=beam_width,
        )
        # decoded: (top_paths, batch, T) — take top-1 path
        results = []
        for b in range(batch_size):
            seq = decoded[0][b]
            try:
                seq = seq.numpy()
            except AttributeError:
                pass
            text = "".join(
                self._idx_to_char.get(int(i), "")
                for i in seq
                # -1 = ctc_decode padding; blank_idx = CTC blank token
                if int(i) >= 0 and int(i) != self._blank_idx
            ).upper()   # model trained on lowercase — uppercase for display
            results.append(text)
        return results

    def _decode_confidence(self, y_pred: np.ndarray) -> float:
        max_probs = np.max(y_pred[0], axis=-1)
        return float(np.mean(max_probs))

    def _detect_word_regions(self, img_bgr: np.ndarray) -> List[tuple]:
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W   = gray.shape

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=8,
        )
        kw      = max(int(W * 0.04), 20)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 500:    continue
            if h < 8:          continue
            if w < 10:         continue
            if w > W * 0.98:   continue
            if h > H * 0.90:   continue
            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1] // 50, b[0]))
        if not boxes:
            boxes = [(0, 0, W, H)]
        return boxes

    def _annotate(self, img_bgr: np.ndarray, regions: List[Dict]) -> np.ndarray:
        out = img_bgr.copy()
        for i, r in enumerate(regions):
            x1, y1, x2, y2 = r["bbox"]
            color = self.COLORS[i % len(self.COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = min(1.0, max(0.45, (x2 - x1) / 150))
            label  = f"  {r['text']}  "
            (tw, th), _ = cv2.getTextSize(label, font, fscale, 2)
            ly = max(y1 - th - 10, 0)
            cv2.rectangle(out, (x1, ly), (x1 + tw, ly + th + 10), color, -1)
            cv2.putText(out, label, (x1, ly + th + 4),
                        font, fscale, (10, 10, 10), 2, cv2.LINE_AA)
        return out

    def predict(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        H, W  = img_bgr.shape[:2]
        boxes = self._detect_word_regions(img_bgr)

        regions = []
        for x, y, bw, bh in boxes[:8]:
            x1 = max(0, x);      y1 = max(0, y)
            x2 = min(W, x + bw); y2 = min(H, y + bh)
            if x2 <= x1 or y2 <= y1: continue

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0: continue

            proc   = self._preprocess_crop(crop)
            batch  = np.expand_dims(proc, 0)
            y_pred = self._model.predict(batch, verbose=0)
            texts  = self._decode(y_pred)
            conf   = self._decode_confidence(y_pred)

            if texts and texts[0]:
                regions.append({
                    "text"      : texts[0],
                    "bbox"      : [x1, y1, x2, y2],
                    "confidence": round(conf, 4),
                })

        return {
            "regions"      : regions,
            "full_text"    : " ".join(r["text"] for r in regions),
            "annotated_bgr": self._annotate(img_bgr, regions),
        }