from dataclasses import dataclass, field


@dataclass
class FraudDecision:
    verdict:     str          # "Genuine", "Suspicious", or "Fake"
    fraud_score: float        # 0.0 = clean, 1.0 = definitely fake
    confidence:  float        # how confident we are in the verdict
    reasons:     list = field(default_factory=list)
    details:     dict = field(default_factory=dict)

    def __str__(self):
        lines = [
            f"Verdict     : {self.verdict}",
            f"Fraud Score : {self.fraud_score:.3f}",
            f"Confidence  : {self.confidence:.3f}",
            "Reasons:",
        ] + [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def make_decision(yolo_detections, verhoeff_result, consistency_result, tampering_result):
    """
    Take all the signals from the pipeline and produce a final verdict.
    We accumulate a fraud_score — the higher it gets, the more suspicious the card.
    """
    reasons = []
    score = 0.0

    # check if YOLO found the important regions
    detected = {d["class_name"] for d in yolo_detections}
    avg_conf = sum(d["confidence"] for d in yolo_detections) / len(yolo_detections) if yolo_detections else 0.0

    missing = {"aadhaar_number", "name", "dob"} - detected
    if missing:
        reasons.append(f"Couldn't detect these regions: {', '.join(missing)}")
        score += 0.08 * len(missing)  # softer penalty — layout varies across card versions

    if avg_conf < 0.4:
        reasons.append(f"YOLO confidence was pretty low ({avg_conf:.2f})")
        score += 0.10

    # Verhoeff checksum — critical validation, card is Fake if this fails
    if not verhoeff_result.get("valid", False):
        reasons.append(f"Aadhaar number failed validation: {verhoeff_result.get('reason', '')}")
        score += 0.45  # pushes score past 0.40 Fake threshold on its own

    # consistency between OCR and QR data
    if not consistency_result.get("format_check", {}).get("pattern_ok", True):
        reasons.append("Aadhaar number doesn't match the XXXX XXXX XXXX format")
        score += 0.15

    # only penalise QR cross-checks when QR data was actually decoded
    qr_available = consistency_result.get("qr_available", False)
    qr_found = consistency_result.get("qr_available", False) or consistency_result.get("qr_error") != "No QR code detected in image"

    # QR not found at all → suspicious by default
    if not qr_available:
        qr_error = consistency_result.get("qr_error", "")
        if "No QR code detected" in str(qr_error):
            reasons.append("No QR code could be detected in the image — card is suspicious")
            score += 0.25  # pushes into Suspicious territory

    if qr_available:
        qr_match = consistency_result.get("qr_match_score", 0)
        qr_comparison = consistency_result.get("qr_comparison", {})

        # If QR decoded but nothing matches at all → Fake
        qr_match = consistency_result.get("qr_match_score", 0)
        if qr_match == 0 and qr_comparison:
            reasons.append("QR code data does not match any field on the card — likely a fake or swapped QR")
            score += 0.45  # push straight to Fake
        elif qr_match < 50 and qr_comparison:
            reasons.append(f"QR data only {qr_match:.0f}% consistent with card fields")
            score += 0.25

        # use the richer QRValidator comparison if available
        if qr_comparison:
            if qr_comparison.get("name", {}).get("match") is False:
                ocr_n = qr_comparison.get("name", {}).get("ocr_value", "?")
                qr_n  = qr_comparison.get("name", {}).get("qr_value", "?")
                reasons.append(f"Name mismatch — OCR read '{ocr_n}', QR says '{qr_n}'")
                score += 0.20
            if qr_comparison.get("dob", {}).get("match") is False:
                ocr_d = qr_comparison.get("dob", {}).get("ocr_value", "?")
                qr_d  = qr_comparison.get("dob", {}).get("qr_value", "?")
                reasons.append(f"Date of birth mismatch — OCR read '{ocr_d}', QR says '{qr_d}'")
                score += 0.20
            if qr_comparison.get("aadhaar_number", {}).get("match") is False:
                ocr_a = qr_comparison.get("aadhaar_number", {}).get("ocr_value", "?")
                qr_a  = qr_comparison.get("aadhaar_number", {}).get("qr_value", "?")
                reasons.append(f"Aadhaar number mismatch — OCR read '{ocr_a}', QR UID is '{qr_a}'")
                score += 0.25
            if qr_comparison.get("last4", {}).get("match") is False:
                ocr_l = qr_comparison.get("last4", {}).get("ocr_value", "?")
                qr_l  = qr_comparison.get("last4", {}).get("qr_value", "?")
                reasons.append(f"Aadhaar last 4 digits mismatch — card shows '{ocr_l}', QR stores '{qr_l}'")
                score += 0.25
        else:
            # fallback to old consistency checks
            if consistency_result.get("name_match", {}).get("match") is False:
                ocr_n = consistency_result.get("name_match", {}).get("ocr_name", "?")
                qr_n  = consistency_result.get("name_match", {}).get("qr_name", "?")
                reasons.append(f"Name mismatch — OCR read '{ocr_n}', QR says '{qr_n}'")
                score += 0.20
            if consistency_result.get("dob_match", {}).get("match") is False:
                ocr_d = consistency_result.get("dob_match", {}).get("ocr_dob", "?")
                qr_d  = consistency_result.get("dob_match", {}).get("qr_dob", "?")
                reasons.append(f"Date of birth mismatch — OCR read '{ocr_d}', QR says '{qr_d}'")
                score += 0.20
            if consistency_result.get("uid_match", {}).get("match") is False:
                ocr_a = consistency_result.get("uid_match", {}).get("ocr_last4", "?")
                qr_a  = consistency_result.get("uid_match", {}).get("qr_last4", "?")
                reasons.append(f"Aadhaar last 4 digits mismatch — OCR read '{ocr_a}', QR says '{qr_a}'")
                score += 0.25

    # Photo comparison (QR photo vs card face)
    photo_cmp = consistency_result.get("photo_comparison", {})
    photo_decision = photo_cmp.get("decision", "UNAVAILABLE")
    if photo_decision == "NO_MATCH":
        reasons.append(f"Face photo mismatch — QR photo does not match card face (SSIM={photo_cmp.get('ssim', 0):.2f})")
        score += 0.50  # photo swap is definitive fraud — push straight to Fake
    elif photo_decision == "SUSPICIOUS":
        reasons.append(f"Face photo similarity is low — possible substitution (SSIM={photo_cmp.get('ssim', 0):.2f})")
        score += 0.25
    resnet_label   = tampering_result.get("label", "unknown")
    resnet_conf    = tampering_result.get("confidence", 0.0)
    forensics_data = tampering_result.get("forensics", {})

    def _forensics_detail(f):
        if not f:
            return ["no forensics data available"]
        parts = []
        ela   = f.get("ela_mean", 0)
        noise = f.get("noise_cv", 0)
        sharp = f.get("sharpness_cv", 0)
        flags = f.get("flags_triggered", 0)

        if f.get("ela_flagged"):
            parts.append(f"re-compression artifacts (ELA={ela:.1f}/255, threshold>10)")
        else:
            parts.append(f"ELA={ela:.1f}/255 (normal)")

        if f.get("noise_flagged"):
            parts.append(f"uneven noise across regions (CV={noise:.2f}, threshold>0.7)")
        else:
            parts.append(f"noise CV={noise:.2f} (normal)")

        if f.get("sharpness_flagged"):
            parts.append(f"sharpness inconsistency (CV={sharp:.2f}, threshold>1.8)")
        else:
            parts.append(f"sharpness CV={sharp:.2f} (normal)")

        parts.append(f"{flags}/3 flags triggered")
        return parts

    if resnet_label == "fake":
        parts = _forensics_detail(forensics_data)
        reasons.append("Image forensics detected tampering — " + " | ".join(parts))
        score += 0.30 * resnet_conf
    elif resnet_label == "suspicious":
        parts = _forensics_detail(forensics_data)
        reasons.append("Image forensics found minor anomalies — " + " | ".join(parts))
        score += 0.12 * resnet_conf
    # no bonus for "real" — absence of evidence isn't evidence of absence

    # clamp score between 0 and 1
    fraud_score = max(0.0, min(1.0, score))

    # verdict thresholds
    if fraud_score < 0.15:
        verdict    = "Genuine"
        confidence = 1.0 - fraud_score
    elif fraud_score < 0.40:
        verdict    = "Suspicious"
        confidence = 1.0 - abs(fraud_score - 0.275)
    else:
        verdict    = "Fake"
        confidence = fraud_score

    if not reasons:
        reasons.append("All checks passed.")

    return FraudDecision(
        verdict=verdict,
        fraud_score=round(fraud_score, 3),
        confidence=round(confidence, 3),
        reasons=reasons,
        details={
            "yolo_avg_conf":    round(avg_conf, 3),
            "detected_classes": list(detected),
            "verhoeff":         verhoeff_result,
            "consistency":      consistency_result,
            "tampering":        tampering_result,
        },
    )
