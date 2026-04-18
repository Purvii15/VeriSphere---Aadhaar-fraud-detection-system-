"""
Aadhaar Fraud Detection Pipeline
"""

import argparse
import json
import cv2
from difflib import SequenceMatcher

from aadhaar_pipeline.detector      import load_model, detect_regions, get_crops_by_class, draw_detections
from aadhaar_pipeline.ocr           import extract_all_text, normalize_aadhaar
from aadhaar_pipeline.validator     import validate_aadhaar_number
from aadhaar_pipeline.qr_validation import QRValidator
from aadhaar_pipeline.consistency   import run_all_checks
from aadhaar_pipeline.tampering     import load_tampering_model, predict_tampering
from aadhaar_pipeline.decision      import make_decision


def _refine_name_with_qr(ocr_name, qr_name):
    """
    OCR often picks up Hindi label noise before the actual name,
    e.g. 'Tangi fzRr Dheeraj Singha' when QR says 'Dheeraj Singha'.
    Slide a word-window over the OCR output and return the window
    that best matches the QR name (if it's a meaningful improvement).
    """
    ocr_words = ocr_name.split()
    n, m = len(ocr_words), len(qr_name.split())

    if n == 0 or m == 0:
        return None

    best_score, best_window = 0.0, None

    for size in range(max(1, m - 1), m + 2):
        for start in range(n - size + 1):
            window = " ".join(ocr_words[start:start + size])
            score  = SequenceMatcher(None, window.lower(), qr_name.lower()).ratio()
            if score > best_score:
                best_score, best_window = score, window

    orig_score = SequenceMatcher(None, ocr_name.lower(), qr_name.lower()).ratio()
    if best_window and best_score > orig_score and best_score >= 0.60:
        return best_window
    return None


def run_pipeline(image_path, yolo_weights, resnet_weights=None, device="cpu",
                 save_vis=None, verbose=True, yolo_model=None, resnet_tuple=None):
    """Run the full pipeline on a single Aadhaar card image."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't load image: {image_path}")
    log(verbose, f"Loaded image: {image_path}  ({image.shape[1]}x{image.shape[0]})")

    # step 1 — YOLO region detection
    yolo = yolo_model if yolo_model is not None else load_model(yolo_weights)
    detections = detect_regions(yolo, image)
    crops = get_crops_by_class(detections)
    log(verbose, f"YOLO found {len(detections)} regions: {[d['class_name'] for d in detections]}")

    if save_vis:
        cv2.imwrite(save_vis, draw_detections(image, detections))

    # step 2 — OCR
    ocr_fields = extract_all_text(crops)

    # step 3 — Verhoeff: pick the Aadhaar candidate that passes checksum
    candidates = ocr_fields.get("aadhaar_candidates", [])
    verified_aadhaar, verhoeff_passed = "", None

    for candidate in candidates:
        result = validate_aadhaar_number(candidate)
        if result["valid"]:
            verified_aadhaar, verhoeff_passed = candidate, result
            log(verbose, f"  Verhoeff PASSED: {candidate}")
            break
        else:
            log(verbose, f"  Verhoeff failed: {candidate} — {result['reason']}")

    if not verified_aadhaar:
        verified_aadhaar = ocr_fields.get("aadhaar_number", "")
        verhoeff_passed  = validate_aadhaar_number(verified_aadhaar)

    ocr_fields["aadhaar_number"] = verified_aadhaar
    log(verbose, f"OCR fields (pre-QR): {ocr_fields}")

    # step 4 — QR decode
    qr_validator = QRValidator()
    qr_crop      = crops.get("qr_code", [None])[0]
    qr_result    = qr_validator.validate_qr(image, ocr_fields, qr_crop=qr_crop)
    qr_fields    = qr_result.get("qr_parsed_data") or {}

    log(verbose, f"QR found: {qr_result['qr_found']} | valid: {qr_result['qr_valid']} | "
                 f"format: {qr_result.get('qr_format')} | match: {qr_result.get('match_score', 0):.0f}%")
    if qr_result.get("error"):
        log(verbose, f"  QR error: {qr_result['error']}")

    # step 4b — refine OCR name using QR name (removes Hindi label noise)
    # Must happen before consistency checks but comparison_details also needs updating
    qr_name = qr_fields.get("name", "")
    if qr_name and ocr_fields.get("name"):
        refined = _refine_name_with_qr(ocr_fields["name"], qr_name)
        if refined:
            log(verbose, f"  Name refined: '{ocr_fields['name']}' → '{refined}'")
            ocr_fields["name"] = refined
            # update comparison_details so the name check uses the refined value
            comp = qr_result.get("comparison_details", {})
            if "name" in comp:
                from difflib import SequenceMatcher as _SM
                qn = comp["name"].get("qr_value", qr_name)
                on = refined
                match = _SM(None, on.lower(), qn.lower()).ratio() >= 0.75 or \
                        on.lower() in qn.lower() or qn.lower() in on.lower()
                comp["name"] = {"qr_value": qn, "ocr_value": on, "match": match}
                qr_result["comparison_details"] = comp
                # recalculate match_score from updated comparison_details
                total = len(comp)
                matched = sum(1 for v in comp.values() if v.get("match") is True)
                qr_result["match_score"] = (matched / total * 100) if total else 0

    # step 5 — consistency checks
    verhoeff = verhoeff_passed
    log(verbose, f"Verhoeff: {verhoeff['reason']} ({verified_aadhaar})")

    consistency = run_all_checks(ocr_fields, qr_fields)
    consistency["qr_available"]   = qr_result["qr_found"] and qr_result["qr_valid"]
    consistency["qr_format"]      = qr_result.get("qr_format")
    consistency["qr_match_score"] = qr_result.get("match_score", 0)
    consistency["qr_comparison"]  = qr_result.get("comparison_details", {})
    consistency["qr_parsed_data"] = qr_result.get("qr_parsed_data") or {}
    # convert photo bytes (JPEG2000) → JPEG base64 so browser can display it
    _parsed = consistency["qr_parsed_data"]
    _photo = _parsed.get("photo")
    if isinstance(_photo, (bytes, bytearray)) and len(_photo) > 100:
        import base64 as _b64
        import io as _io
        converted = False
        # Try imagecodecs.jpeg2k_decode — correct API for JP2/J2K
        if not converted:
            try:
                import imagecodecs
                import numpy as _np
                arr = imagecodecs.jpeg2k_decode(bytes(_photo))  # returns numpy array (H,W,C)
                if arr is not None:
                    from PIL import Image as _Img
                    buf = _io.BytesIO()
                    _Img.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=85)
                    consistency["qr_parsed_data"]["photo_b64"] = _b64.b64encode(buf.getvalue()).decode()
                    converted = True
            except Exception as _e:
                print(f"[DEBUG] imagecodecs.jpeg2k_decode failed: {_e}")
        # Fallback: Pillow (works if imagecodecs plugin is registered)
        if not converted:
            try:
                from PIL import Image as _Img
                img = _Img.open(_io.BytesIO(bytes(_photo)))
                buf = _io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                consistency["qr_parsed_data"]["photo_b64"] = _b64.b64encode(buf.getvalue()).decode()
                converted = True
            except Exception as _e:
                print(f"[DEBUG] Pillow JP2 failed: {_e}")
        # Fallback: OpenCV
        if not converted:
            try:
                import numpy as _np
                arr = _np.frombuffer(bytes(_photo), dtype=_np.uint8)
                img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    _, buf = cv2.imencode(".jpg", img_cv)
                    consistency["qr_parsed_data"]["photo_b64"] = _b64.b64encode(buf).decode()
                    converted = True
            except Exception as _e:
                print(f"[DEBUG] OpenCV JP2 failed: {_e}")
        if not converted:
            # dump first 16 bytes to help diagnose format
            print(f"[DEBUG] photo decode failed. First bytes: {bytes(_photo)[:16].hex()}")
    _raw_qr_photo_for_compare = bytes(_photo) if _photo else None  # save before pop
    consistency["qr_parsed_data"].pop("photo", None)  # remove raw bytes
    consistency["qr_error"]       = qr_result.get("error")

    # step 5b — photo comparison (QR photo vs card face crop)
    photo_comparison = {"decision": "UNAVAILABLE", "fraud_contribution": 0}
    _face_crops = crops.get("face", [])
    _face_crop  = _face_crops[0] if _face_crops else None
    if _raw_qr_photo_for_compare and _face_crop is not None:
        from aadhaar_pipeline.photo_compare import compare_photos
        photo_comparison = compare_photos(_raw_qr_photo_for_compare, _face_crop)
        log(verbose, f"Photo comparison: {photo_comparison['decision']} (SSIM={photo_comparison.get('ssim')})")
        # factor photo into QR match score — photo is an additional QR field
        photo_dec = photo_comparison.get("decision", "UNAVAILABLE")
        if photo_dec != "UNAVAILABLE":
            # treat photo as one extra check: MATCH=pass, SUSPICIOUS=half, NO_MATCH=fail
            photo_pass = 1.0 if photo_dec == "MATCH" else 0.5 if photo_dec == "SUSPICIOUS" else 0.0
            old_score  = consistency.get("qr_match_score", 0)
            # blend: existing score covers N fields, photo adds 1 more
            n_fields   = max(qr_result.get("total_checks", 2), 1)
            new_score  = (old_score * n_fields + photo_pass * 100) / (n_fields + 1)
            consistency["qr_match_score"] = round(new_score, 1)
    consistency["photo_comparison"] = photo_comparison
    log(verbose, f"Consistency: {consistency['overall_score']} | QR match: {consistency['qr_match_score']:.0f}%")

    # step 6 — tampering detection
    resnet_tuple = resnet_tuple if resnet_tuple is not None else load_tampering_model(resnet_weights, device=device)
    tampering = predict_tampering(resnet_tuple, image, device=device)
    log(verbose, f"Tampering: {tampering['label']} ({tampering['confidence']:.2f})")

    # step 7 — final decision
    decision = make_decision(detections, verhoeff, consistency, tampering)
    log(verbose, f"\n{'='*45}\n{decision}\n{'='*45}")

    return {
        "image_path":         image_path,
        "detections":         [{k: v for k, v in d.items() if k != "crop"} for d in detections],
        "ocr_fields":         ocr_fields,
        "aadhaar_candidates": candidates,
        "qr_result":          {k: v for k, v in qr_result.items() if k not in ("qr_parsed_data",)},
        "qr_fields":          qr_fields,
        "verhoeff":           verhoeff,
        "consistency":        consistency,
        "tampering":          tampering,
        "verdict":            decision.verdict,
        "fraud_score":        decision.fraud_score,
        "confidence":         decision.confidence,
        "reasons":            decision.reasons,
    }


def log(verbose, msg):
    if verbose:
        print(msg)


def main():
    parser = argparse.ArgumentParser(description="Aadhaar Fraud Detection")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--yolo",     required=True)
    parser.add_argument("--resnet",   default=None)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--save-vis", default=None)
    parser.add_argument("--json",     action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.image, yolo_weights=args.yolo,
        resnet_weights=args.resnet, device=args.device,
        save_vis=args.save_vis, verbose=not args.json,
    )
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
