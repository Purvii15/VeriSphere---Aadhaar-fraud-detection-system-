import re
from difflib import SequenceMatcher


def _similarity(a, b):
    """How similar are two strings? Returns 0.0 to 1.0."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def check_name_match(ocr_name, qr_name, threshold=0.75):
    """See if the name from OCR roughly matches what's in the QR code."""
    score = _similarity(ocr_name, qr_name)
    return {
        "ocr_name": ocr_name,
        "qr_name":  qr_name,
        "score":    round(score, 3),
        "match":    score >= threshold,
    }


def check_dob_match(ocr_dob, qr_dob):
    """Compare dates of birth — strip separators before comparing."""
    def digits_only(d):
        return re.sub(r"[^0-9]", "", d)

    ocr_d = digits_only(ocr_dob)
    qr_d  = digits_only(qr_dob)
    return {
        "ocr_dob": ocr_dob,
        "qr_dob":  qr_dob,
        "match":   bool(ocr_d and qr_d and ocr_d == qr_d),
    }


def check_aadhaar_format(aadhaar_text):
    """Make sure the Aadhaar number looks like XXXX XXXX XXXX."""
    digits = re.sub(r"\D", "", aadhaar_text)
    formatted = f"{digits[:4]} {digits[4:8]} {digits[8:]}" if len(digits) == 12 else aadhaar_text
    return {
        "raw":        aadhaar_text,
        "formatted":  formatted,
        "pattern_ok": bool(re.fullmatch(r"\d{4} \d{4} \d{4}", formatted)),
    }


def check_qr_uid_match(ocr_aadhaar, qr_uid_last4):
    """
    The QR code only stores the last 4 digits of the UID for privacy.
    Check that those 4 digits match what OCR read.
    """
    ocr_last4 = re.sub(r"\D", "", ocr_aadhaar)[-4:]
    return {
        "ocr_last4": ocr_last4,
        "qr_last4":  qr_uid_last4,
        "match":     bool(ocr_last4 and qr_uid_last4 and ocr_last4 == qr_uid_last4),
    }


def run_all_checks(ocr_fields, qr_fields):
    """
    Run all four consistency checks and return a combined report with an overall score.
    ocr_fields comes from OCR, qr_fields comes from the decoded QR code.
    QR-dependent checks are skipped (marked as None) when QR data is unavailable.
    """
    qr_available = bool(qr_fields)

    format_check = check_aadhaar_format(ocr_fields.get("aadhaar_number", ""))

    # Only run QR-dependent checks when we actually have QR data
    if qr_available:
        name_check = check_name_match(ocr_fields.get("name", ""), qr_fields.get("name", ""))
        dob_check  = check_dob_match(ocr_fields.get("dob", ""), qr_fields.get("dob", ""))
        uid_check  = check_qr_uid_match(ocr_fields.get("aadhaar_number", ""), qr_fields.get("uid_last4", ""))
        passed = [name_check["match"], dob_check["match"], format_check["pattern_ok"], uid_check["match"]]
    else:
        name_check = {"ocr_name": ocr_fields.get("name", ""), "qr_name": "", "score": None, "match": None}
        dob_check  = {"ocr_dob": ocr_fields.get("dob", ""), "qr_dob": "", "match": None}
        uid_check  = {"ocr_last4": "", "qr_last4": "", "match": None}
        passed = [format_check["pattern_ok"]]  # only score what we can actually check

    overall = sum(p for p in passed if p is not None) / len(passed) if passed else 0.0

    return {
        "name_match":    name_check,
        "dob_match":     dob_check,
        "format_check":  format_check,
        "uid_match":     uid_check,
        "qr_available":  qr_available,
        "overall_score": round(overall, 3),
    }
