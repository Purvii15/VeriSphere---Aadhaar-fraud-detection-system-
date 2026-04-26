# VeriSphere — Aadhaar Fraud Detection System

An AI-powered web app that verifies Aadhaar cards through a 5-layer detection pipeline — catching fakes using computer vision, OCR, QR validation, face matching, and digital forensics.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Gemini](https://img.shields.io/badge/Gemini-AI-orange)

---

## What It Does

Upload a photo of an Aadhaar card and VeriSphere runs it through five checks in sequence:

1. **YOLOv8 Detection** — finds and crops each field on the card (name, DOB, Aadhaar number, face, QR code)
2. **EasyOCR Extraction** — reads text from each cropped region with preprocessing to handle low-quality scans
3. **Verhoeff Checksum** — mathematically validates the 12-digit Aadhaar number
4. **QR Cross-Check** — decodes the embedded QR (supports both old XML and newer secure binary formats) and compares every field against what OCR read
5. **Deep Forensics** — runs ELA (Error Level Analysis), noise uniformity, and sharpness checks to catch pixel-level tampering

All five signals feed into a weighted fraud score that gives a final verdict: **Genuine**, **Suspicious**, or **Fake**.

---

## Features

- **Live QR Decoder** (`/qr-decode`) — paste a raw QR string from a handheld scanner to instantly see all decoded fields and the embedded photo
- **Gemini AI Chat** — floating chat widget powered by Gemini 2.0 Flash; falls back to a free HF model if quota runs out
- **Animated processing UI** — radar rings, stage progress indicators, and particle canvas during analysis
- **Persistent fraud analytics** — every analysis is saved to a local SQLite database so stats survive server restarts
- **Auto-analysis** — analysis starts the moment you drop a file, no button press needed

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| QR Decoding | zxing-cpp / pyzbar |
| Face Comparison | SSIM (scikit-image) + Haar cascade |
| Checksum Validation | Verhoeff Algorithm |
| Forensics | OpenCV (ELA, noise, sharpness) |
| AI Assistant | Google Gemini 2.0 Flash + Llama 3.2 fallback |
| Web Server | Flask + Gunicorn |
| Frontend | Tailwind CSS (dark theme) |
| Database | SQLite (persistent fraud analytics) |

---

## Project Structure

```
verisphere/
├── aadhaar_pipeline/
│   ├── pipeline.py        # Main orchestration — runs all 5 stages
│   ├── detector.py        # YOLOv8 inference and crop extraction
│   ├── ocr.py             # EasyOCR with multi-variant preprocessing
│   ├── qr_validation.py   # QR decode, field comparison, photo extraction
│   ├── photo_compare.py   # SSIM face matching with Haar alignment
│   ├── tampering.py       # ELA, noise, sharpness forensics
│   ├── validator.py       # Verhoeff checksum
│   ├── consistency.py     # Cross-field consistency checks
│   └── decision.py        # Fraud scoring and verdict
├── flask_app.py           # Flask server, all routes, HTML templates
├── analytics_db.py        # SQLite-backed fraud analytics storage
├── requirements.txt       # Python dependencies
├── aadhaar_best.pt        # YOLOv8 model weights (not in git)
└── resnet_aadhaar.pth     # ResNet tampering model (not in git)
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Purvii15/VeriSphere---Aadhaar-fraud-detection-system-.git
cd VeriSphere---Aadhaar-fraud-detection-system-

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key (optional — app works without it)
# Create a .env file:
echo GEMINI_API_KEY=your_key_here > .env
# Get a free key at: https://aistudio.google.com/app/apikey

# 5. Run
python flask_app.py
```

Open `http://localhost:5000` in your browser.

> **Note:** Model weights (`aadhaar_best.pt`, `resnet_aadhaar.pth`) are not included in this repo due to file size. Contact the team to get them.

---

## How the Fraud Score Works

Each check contributes a penalty to the fraud score (0.0 → 1.0):

| Signal | Penalty | Notes |
|---|---|---|
| Verhoeff checksum fail | +0.30 | Strong indicator |
| Aadhaar number mismatch (QR vs OCR) | +0.25 | Critical field |
| Name mismatch (QR vs OCR) | +0.20 | |
| DOB mismatch (QR vs OCR) | +0.20 | |
| Face photo doesn't match | +0.50 | Instant Fake |
| Face photo suspicious | +0.25 | |
| Tampering detected (forensics) | up to +0.30 | |

**Verdict thresholds:**
- Score `< 0.15` → ✅ Genuine
- Score `0.15 – 0.40` → ⚠️ Suspicious
- Score `> 0.40` → ❌ Fake

---

## QR Code Support

**Old format (pre-2018):** XML-based, contains full UID, name, DOB, gender, address.

**Secure format (post-2018):** Binary with `0xFF` delimiters. Stores only the last 4 digits of the UID (UIDAI privacy policy). May contain an embedded JPEG2000 photo — extracted and compared against the card face using SSIM.

---

## AI Assistant

The floating **✨ chat button** (bottom-right on the dashboard) lets you ask questions about:
- What a specific fraud indicator means
- How the verification pipeline works
- What the fraud score and confidence values represent

It uses **Gemini 2.0 Flash** as the primary model and automatically falls back to **Llama 3.2 via Hugging Face** if the Gemini quota runs out — so the demo never breaks.

---

## Notes

- Works best with clear, well-lit front-side scans
- Physical printed cards usually don't embed photos in the QR — only eAadhaar digital downloads do
- The `analytics.db` file is local only (gitignored) — it stores your analysis history
- Run with `python flask_app.py` on any platform
