# VeriSphere — Aadhaar Fraud Detection

A multi-stage AI-powered system for detecting fraudulent Aadhaar cards using computer vision, OCR, QR validation, face photo comparison, and digital forensics.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

---

## Features

- **YOLOv8 Detection** — Locates card regions (name, DOB, Aadhaar number, face, QR code) with bounding boxes
- **EasyOCR Text Extraction** — English-only reader with multi-variant preprocessing and OCR correction rules
- **Verhoeff Checksum** — Mathematically validates the 12-digit Aadhaar number
- **QR Cross-Check** — Decodes embedded QR data (both old XML and secure binary formats) and compares against OCR-extracted fields with auto-detected field offsets
- **Face Photo Comparison** — Extracts JPEG2000 photo from QR code, expands and re-crops using Haar cascade, then compares with card face crop using SSIM
- **Digital Forensics** — ELA (Error Level Analysis), noise uniformity, and sharpness consistency checks
- **Fraud Scoring** — Weighted scoring engine produces a final verdict: Genuine / Suspicious / Fake
- **Live QR Decode** — Paste raw QR numeric string to instantly decode all fields and view embedded photo (useful for handheld QR scanner demos)
- **Cinematic Processing UI** — Animated radar rings, stage progress indicators, and particle canvas during analysis

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR (English-only) |
| QR Decoding | pyzbar / OpenCV |
| Face Comparison | SSIM (scikit-image) + Haar cascade alignment |
| Checksum | Verhoeff Algorithm |
| Forensics | OpenCV (ELA, noise, sharpness) |
| Web Server | Flask |
| Frontend | Tailwind CSS (dark theme, animated) |

---

## Project Structure

```
├── aadhaar_pipeline/
│   ├── pipeline.py        # Main orchestration
│   ├── detector.py        # YOLOv8 inference
│   ├── ocr.py             # EasyOCR extraction + preprocessing
│   ├── qr_validation.py   # QR decode + field comparison + photo extraction
│   ├── photo_compare.py   # SSIM-based face photo matching with Haar alignment
│   ├── tampering.py       # Forensics (ELA, noise, sharpness)
│   ├── validator.py       # Verhoeff checksum
│   ├── consistency.py     # Cross-field consistency checks
│   └── decision.py        # Fraud scoring + verdict
├── Templates/
│   └── stitch 1/
│       └── code.html      # Main frontend template
├── flask_app.py           # Flask server + JS glue + QR decode page
├── app.py                 # Streamlit version (legacy)
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate virtual environment
py -m venv .venv
.venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your YOLOv8 weights
#    Copy aadhaar_best.pt into the project root

# 4. Run the Flask app
py flask_app.py
```

Open `http://localhost:5000` in your browser.

---

## How It Works

1. Upload a front-side Aadhaar card image (JPG/PNG/WEBP) — analysis starts automatically
2. YOLOv8 detects and crops each field region (name, DOB, Aadhaar number, face, QR)
3. EasyOCR reads text from each crop with multi-variant preprocessing
4. Verhoeff algorithm validates the Aadhaar number checksum
5. QR code is decoded (handles both old XML and secure binary formats, auto-detects field offset)
6. QR data is compared against OCR values (name, DOB, last 4 digits of UID)
7. If QR contains embedded photo (JPEG2000), it is extracted, padded for Haar detection, and compared with the card face crop using SSIM
8. ELA + noise + sharpness forensics flag pixel-level anomalies
9. A weighted fraud score determines the final verdict

---

## Fraud Detection Weights

| Signal | Max Penalty | Effect |
|---|---|---|
| Verhoeff checksum fail | +0.30 | Strong fraud signal |
| Aadhaar number mismatch (QR vs OCR) | +0.25 | Critical field |
| Name mismatch (QR vs OCR) | +0.20 | → Suspicious alone |
| DOB mismatch (QR vs OCR) | +0.20 | → Suspicious alone |
| Face photo NO_MATCH | +0.50 | → Fake alone (photo swap) |
| Face photo SUSPICIOUS | +0.25 | → Suspicious alone |
| Forensics (fake) | up to +0.30 | Tampering detected |
| Forensics (suspicious) | up to +0.12 | Minor anomalies |

**Verdict Thresholds:**
- `< 0.15` → Genuine
- `0.15–0.40` → Suspicious
- `> 0.40` → Fake

---

## QR Code Support

### Old Format (pre-2018)
- XML-based structure
- Contains full UID, name, DOB, gender, address
- Photo rarely embedded

### Secure Format (post-2018)
- Binary format with `0xFF` delimiters
- Field layout: version · ref_number · name · DOB · gender · C/O · VTC · post_office · house · street · pincode · district · state · locality · sub-district
- UID masked — only last 4 digits stored in reference number (first 4 chars)
- Photo embedded as JPEG2000 (extracted and compared with card face)
- Auto-detects field base offset to handle pyzbar vs manual decode differences

### Live QR Decode (`/qr-decode`)
Paste the raw numeric QR string (from a handheld scanner) to instantly see all decoded fields and the embedded photo. Accessible from the sidebar and top nav.

---

## Face Photo Comparison

When a QR code contains an embedded photo (eAadhaar digital cards):
1. Photo is extracted from QR binary data (JPEG2000 format)
2. Converted to JPEG using Pillow or OpenCV
3. QR photo is padded with white border (50% expansion) so Haar cascade can detect the face despite tight cropping
4. Both photos are face-cropped to the same framing before SSIM comparison
5. Thresholds:
   - SSIM > 0.75 → MATCH (no penalty)
   - SSIM 0.50–0.75 → SUSPICIOUS (+0.25 fraud score)
   - SSIM < 0.50 → NO_MATCH (+0.50 fraud score → Fake)

---

## UI Features

- Auto-analysis on file select — no button press needed
- Filename shown in preview overlay
- Old results cleared when a new card is uploaded
- Processing animation: radar rings + 5 sequential stage indicators + particle canvas (~7.5s paced to match ~10s processing time)
- Lightbox for full-size image preview
- QR Decode page in sidebar for live demo with handheld scanner
- Shield favicon on all pages
- "Coming soon" tooltips on unimplemented nav items

---

## Notes

- Model weights (`*.pt`, `*.h5`) are excluded from this repo via `.gitignore` due to file size
- The system works best with clear, well-lit front-side scans
- Secure-format QR codes (post-2018) do not store the full UID — handled gracefully with last-4 comparison
- Physical printed cards typically don't embed photos in QR — only eAadhaar digital downloads do
- ResNet tampering classifier is disabled by default (forensics-only mode preferred)
- Run with `py flask_app.py` on Windows (uses `py` launcher)
