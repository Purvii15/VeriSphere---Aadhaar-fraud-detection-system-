import re
import cv2
import numpy as np
import easyocr

# keep the reader alive between calls so we don't reload the model every time
_reader = None
_reader_en = None


def get_reader(languages=["en"]):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def get_reader_en():
    """English-only reader for name fields — less noise than bilingual."""
    global _reader_en
    if _reader_en is None:
        _reader_en = easyocr.Reader(["en"], gpu=False)
    return _reader_en


def _preprocess_for_ocr(crop, aggressive=False):
    """Upscale and threshold a crop to improve OCR accuracy."""
    h, w = crop.shape[:2]

    if h < 80 or w < 250:
        scale = max(80 / h, 250 / w, 2.0)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if aggressive:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        binary = cv2.filter2D(binary, -1, kernel)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _preprocess_name_crop(crop):
    """
    High-quality preprocessing variants specifically for name fields.
    Returns a list of preprocessed images to try.
    """
    h, w = crop.shape[:2]

    # aggressively upscale — name text is often small
    scale = max(120 / h, 400 / w, 3.0)
    large = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY) if len(large.shape) == 3 else large

    variants = []

    # variant 1: CLAHE contrast enhancement + Otsu threshold
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # variant 2: bilateral filter + adaptive threshold
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    # variant 3: morphological closing to reconnect broken characters + Otsu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, otsu2 = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu2, cv2.COLOR_GRAY2BGR))

    return variants


# Common OCR misread corrections for Aadhaar card fonts
_NAME_CORRECTIONS = [
    (r'\bDl(?=[a-z])', 'Dh'),       # Dlieeraj → Dheeraj
    (r'(?<=[A-Za-z])li(?=[a-z])', 'h'),  # elieeraj → eheeraj
    (r'(?<=\s)li(?=[A-Z])', 'H'),   # li at word start
    (r'rn(?=[a-z])', 'm'),          # rn → m
    (r'(?<=[a-z])rn', 'm'),
    (r'\bvv\b', 'w'), (r'\bVV\b', 'W'),
    (r'(?<=[A-Za-z])1(?=[a-z])', 'l'),  # digit 1 → letter l mid-word
    (r'(?<=[a-z])1(?=[A-Za-z])', 'l'),
    (r'\b0(?=[A-Z])', 'O'),         # digit 0 → letter O at word start
    (r'(?<=[A-Za-z])0(?=[A-Za-z])', 'o'),
    (r'\|', 'I'), (r'!(?=[a-z])', 'i'),
    (r'Sinclia\b', 'Singha'), (r'Sincl(?=[a-z])', 'Singh'),
    (r'Sinch\b', 'Singh'),
]


def _fix_name_ocr(text):
    """Apply common OCR correction rules for name fields."""
    for pattern, replacement in _NAME_CORRECTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def extract_text(crop, languages=["en"]):
    """Pull text out of a single cropped region."""
    if crop is None or crop.size == 0:
        return ""
    reader = get_reader()
    processed = _preprocess_for_ocr(crop, aggressive=False)
    results = reader.readtext(processed, detail=0, paragraph=False)
    return " ".join(results).strip()


def extract_name(crop):
    """
    High-accuracy name extraction.
    Tries multiple preprocessed variants, picks the one with the most
    alphabetic content, then applies OCR correction rules.
    """
    if crop is None or crop.size == 0:
        return ""

    reader = get_reader_en()  # English-only — Hindi reader adds noise to Latin names
    variants = _preprocess_name_crop(crop)

    best_text = ""
    best_score = -1

    for variant in variants:
        results = reader.readtext(variant, detail=0, paragraph=False)
        text = " ".join(results).strip()
        # score by ratio of alphabetic chars — higher = cleaner read
        chars = text.replace(" ", "")
        score = sum(c.isalpha() for c in chars) / len(chars) if chars else 0
        if score > best_score or (score == best_score and len(text) > len(best_text)):
            best_score = score
            best_text = text

    return _fix_name_ocr(best_text)


def extract_text_digits(crop):
    """Optimised extraction for numeric fields."""
    if crop is None or crop.size == 0:
        return ""
    processed = _preprocess_for_ocr(crop, aggressive=True)
    reader = get_reader()
    results = reader.readtext(processed, detail=0, paragraph=False, allowlist="0123456789 ")
    return " ".join(results).strip()


def clean_dob(raw):
    """Extract just the date from noisy OCR output."""
    m = re.search(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b", raw)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b", raw)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b", raw)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return raw


def clean_name(raw):
    """Remove label noise from name field."""
    cleaned = re.sub(r"(?i)^(name\s*[:\-]?\s*|नाम\s*[:\-]?\s*|to\s+)", "", raw).strip()
    cleaned = re.sub(r"[^A-Za-z\s\.]", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if len(cleaned) >= 3 else raw


def extract_all_text(crops_by_class):
    """
    Run OCR on each text region YOLO found.
    For aadhaar_number we return ALL candidates so the validator can pick
    the one that passes Verhoeff.
    """
    text_fields = {}

    # address
    address_crops = crops_by_class.get("address", [])
    text_fields["address"] = extract_text(address_crops[0]) if address_crops else ""

    # name — use high-accuracy dedicated extractor
    name_crops = crops_by_class.get("name", [])
    if name_crops:
        raw_name = extract_name(name_crops[0])
        text_fields["name"] = clean_name(raw_name)
    else:
        text_fields["name"] = ""

    # dob
    dob_crops = crops_by_class.get("dob", [])
    text_fields["dob"] = clean_dob(extract_text(dob_crops[0])) if dob_crops else ""

    # aadhaar number — all candidates
    aadhaar_crops = crops_by_class.get("aadhaar_number", [])
    candidates = []
    for crop in aadhaar_crops:
        for text_fn in [extract_text_digits, extract_text]:
            text = normalize_aadhaar(text_fn(crop))
            if text:
                candidates.append(text)

    seen = set()
    unique_candidates = []
    for c in candidates:
        digits = re.sub(r"\D", "", c)
        if digits not in seen:
            seen.add(digits)
            unique_candidates.append(c)

    text_fields["aadhaar_candidates"] = unique_candidates
    text_fields["aadhaar_number"] = unique_candidates[0] if unique_candidates else ""

    return text_fields


def normalize_aadhaar(raw):
    """Clean up the Aadhaar number and format it as XXXX XXXX XXXX."""
    m = re.search(r"\b(\d{4})\s+(\d{4})\s+(\d{4})\b", raw)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    m = re.search(r"\b([2-9]\d{11})\b", raw)
    if m:
        d = m.group(1)
        return f"{d[:4]} {d[4:8]} {d[8:]}"
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 12:
        d = digits[:12]
        return f"{d[:4]} {d[4:8]} {d[8:]}"
    return digits
