"""
CRNN OCR Predictor — Inference Engine
Handles: preprocessing → model forward pass → CTC decode → annotation
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class CRNNPredictor:
    """
    Production inference wrapper for the CRNN OCR model.

    Architecture:
        Input (32×128×1) → CNN (6 blocks) → Reshape → BiLSTM×2 → Softmax → CTC decode

    Usage:
        predictor = CRNNPredictor(model_path, vocab_path)
        result    = predictor.predict(img_bgr)
    """

    IMG_HEIGHT = 32
    IMG_WIDTH  = 128

    # Color palette for bounding boxes (BGR for OpenCV)
    COLORS = [
        (100, 220, 50),   # bright green
        (255, 180, 30),   # amber
        (50, 180, 255),   # sky blue
        (180, 80, 255),   # violet
        (50, 230, 200),   # teal
        (255, 100, 100),  # coral
    ]

    def __init__(self, model_path: str, vocab_path: str):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self._model     = None
        self._vocab     = None
        self._load()

    # ─── Loading ─────────────────────────────────────────────────────────────

    def _load(self):
        """Load model weights and vocabulary config."""
        import tensorflow as tf

        # Load vocabulary
        with open(self.vocab_path, "r") as f:
            self._vocab = json.load(f)

        self._idx_to_char = {
            int(k): v for k, v in self._vocab["idx_to_char"].items()
        }
        self._num_timesteps = self._vocab.get("num_timesteps", 32)

        # Load Keras model
        if Path(self.model_path).exists():
            self._model = tf.keras.models.load_model(
                self.model_path, compile=False
            )
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    # ─── Preprocessing ───────────────────────────────────────────────────────

    def _preprocess_crop(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a word crop for CRNN.
        Steps: BGR→Gray → CLAHE → aspect-ratio resize → pad → normalize
        """
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
        return np.expand_dims(norm, axis=-1)   # (32, 128, 1)

    # ─── CTC Decode ──────────────────────────────────────────────────────────

    def _decode(self, y_pred: np.ndarray, beam_width: int = 10) -> List[str]:
        """
        Beam-search CTC decode.
        y_pred: (batch, timesteps, num_classes) float32
        Returns list of decoded strings, one per batch item.
        """
        import tensorflow as tf

        batch_size  = y_pred.shape[0]
        input_lens  = np.full(batch_size, self._num_timesteps)

        decoded, _  = tf.keras.backend.ctc_decode(
            y_pred,
            input_length=input_lens,
            greedy=False,
            beam_width=beam_width,
        )
        results = []
        for seq in decoded[0].numpy():
            text = "".join(
                [self._idx_to_char.get(int(i), "") for i in seq if i > 0]
            )
            results.append(text)
        return results

    def _decode_confidence(self, y_pred: np.ndarray) -> float:
        """
        Approximate confidence: mean of max softmax prob across timesteps.
        Not a true posterior but useful for UI display.
        """
        max_probs = np.max(y_pred[0], axis=-1)  # (timesteps,)
        return float(np.mean(max_probs))

    # ─── Region Detection ────────────────────────────────────────────────────

    def _detect_word_regions(self, img_bgr: np.ndarray) -> List[tuple]:
        """
        Word-level region detection via morphological grouping.

        Algorithm:
        1. Adaptive threshold  → foreground mask
        2. Horizontal dilation → merge characters into word blobs
        3. Contour extraction  → bounding boxes
        4. Filter & sort       → valid word-sized regions
        """
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
            if w * h < 500: continue
            if h < 8:       continue
            if w < 10:      continue
            if w > W * 0.98: continue
            if h > H * 0.90: continue
            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1] // 50, b[0]))

        if not boxes:
            boxes = [(0, 0, W, H)]

        return boxes

    # ─── Annotate ────────────────────────────────────────────────────────────

    def _annotate(
        self,
        img_bgr: np.ndarray,
        regions: List[Dict],
    ) -> np.ndarray:
        """Draw bounding boxes + text labels on the image."""
        out = img_bgr.copy()

        for i, r in enumerate(regions):
            x1, y1, x2, y2 = r["bbox"]
            color = self.COLORS[i % len(self.COLORS)]
            text  = r["text"]

            # Rectangle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

            # Label background
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = min(1.0, max(0.45, (x2 - x1) / 150))
            label  = f"  {text}  "
            (tw, th), _ = cv2.getTextSize(label, font, fscale, 2)
            ly = max(y1 - th - 10, 0)
            cv2.rectangle(out, (x1, ly), (x1 + tw, ly + th + 10), color, -1)
            cv2.putText(out, label, (x1, ly + th + 4),
                        font, fscale, (10, 10, 10), 2, cv2.LINE_AA)

        return out

    # ─── Public API ──────────────────────────────────────────────────────────

    def predict(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Full OCR pipeline on a BGR image.

        Returns:
            {
                "regions"      : [{"text", "bbox", "confidence"}, ...],
                "full_text"    : "all words joined",
                "annotated_bgr": np.ndarray (BGR image with boxes drawn),
            }
        """
        H, W = img_bgr.shape[:2]

        # Detect word regions
        boxes = self._detect_word_regions(img_bgr)

        regions = []
        for i, (x, y, bw, bh) in enumerate(boxes[:8]):
            x1 = max(0, x);      y1 = max(0, y)
            x2 = min(W, x + bw); y2 = min(H, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

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

        full_text    = " ".join(r["text"] for r in regions)
        annotated    = self._annotate(img_bgr, regions)

        return {
            "regions"      : regions,
            "full_text"    : full_text,
            "annotated_bgr": annotated,
        }
