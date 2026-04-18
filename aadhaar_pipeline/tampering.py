"""
Tampering detection — supports both ResNet (trained) and forensics (rule-based).

If ResNet weights are provided: uses trained classifier
Otherwise: uses image forensics (ELA + noise + sharpness analysis)
"""

import io
import cv2
import numpy as np
from PIL import Image


# ── ELA ───────────────────────────────────────────────────────────────────────

def _ela_score(image_bgr, quality=90):
    """Error Level Analysis — detects JPEG re-compression artifacts."""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    
    orig = np.array(pil).astype(np.float32)
    recomp = np.array(recompressed).astype(np.float32)
    ela = np.abs(orig - recomp)
    
    return float(ela.mean()), float(ela.max()), ela


def _noise_inconsistency(image_bgr, grid=4):
    """Measure noise variance across grid cells — spliced regions have different noise."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    ch, cw = h // grid, w // grid
    
    stds = []
    for r in range(grid):
        for c in range(grid):
            cell = gray[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            blurred = cv2.GaussianBlur(cell, (5, 5), 0)
            noise = cell - blurred
            stds.append(float(noise.std()))
    
    stds = np.array(stds)
    cv = stds.std() / (stds.mean() + 1e-6)
    return float(cv), stds.tolist()


def _sharpness_variance(image_bgr, grid=4):
    """Measure sharpness variance — pasted regions have different blur profiles."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ch, cw = h // grid, w // grid
    
    variances = []
    for r in range(grid):
        for c in range(grid):
            cell = gray[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            lap = cv2.Laplacian(cell, cv2.CV_64F)
            variances.append(float(lap.var()))
    
    variances = np.array(variances)
    cv = variances.std() / (variances.mean() + 1e-6)
    return float(cv), variances.tolist()


# ── API ───────────────────────────────────────────────────────────────────────

def load_tampering_model(weights_path=None, backbone="resnet18", device="cpu"):
    """
    Load tampering model.
    ResNet is disabled — it was producing unreliable results (90% fake on genuine cards).
    Forensics-only mode (ELA + noise + sharpness) is used instead.
    """
    print("Forensics-based tampering detection active (ResNet disabled)")
    return (None, "forensics")


def _run_resnet(model, image, device):
    """Run ResNet inference and return fake probability."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    return probs  # [real_prob, fake_prob]


def _run_forensics(image):
    """Run ELA + noise + sharpness and return fake probability."""
    ela_mean, ela_max, _ = _ela_score(image)
    noise_cv, _          = _noise_inconsistency(image)
    sharp_cv, _          = _sharpness_variance(image)

    ela_flag    = ela_mean > 10.0
    ela_score   = min(1.0, max(0.0, (ela_mean - 5.0) / 15.0))
    noise_flag  = noise_cv > 0.7
    noise_score = min(1.0, max(0.0, (noise_cv - 0.3) / 0.8))
    sharp_flag  = sharp_cv > 1.8
    sharp_score = min(1.0, max(0.0, (sharp_cv - 0.8) / 1.5))

    composite = 0.50 * ela_score + 0.30 * noise_score + 0.20 * sharp_score
    flags = sum([ela_flag, noise_flag, sharp_flag])

    return composite, flags, {
        "ela_mean":           round(ela_mean, 3),
        "ela_max":            round(ela_max, 3),
        "ela_flagged":        ela_flag,
        "noise_cv":           round(noise_cv, 3),
        "noise_flagged":      noise_flag,
        "sharpness_cv":       round(sharp_cv, 3),
        "sharpness_flagged":  sharp_flag,
        "flags_triggered":    flags,
    }


def predict_tampering(model_tuple, image, device="cpu"):
    """
    Run tampering detection.
    - "both"      → ResNet (60%) + forensics (40%) combined
    - "forensics" → forensics only
    """
    if image is None or image.size == 0:
        return {"label": "unknown", "confidence": 0.0, "scores": {"real": 0.0, "fake": 0.0}, "skipped": True}

    model, mode = model_tuple

    if mode == "both":
        # ResNet score
        resnet_probs = _run_resnet(model, image, device)
        resnet_fake  = resnet_probs[1]

        # Forensics score
        forensics_fake, flags, forensics_detail = _run_forensics(image)

        # weighted combination — ResNet gets more weight since it's trained
        combined_fake = 0.60 * resnet_fake + 0.40 * forensics_fake
        combined_real = 1.0 - combined_fake

        if combined_fake < 0.35:
            label = "real"
        elif combined_fake > 0.60:
            label = "fake"
        else:
            label = "suspicious"

        return {
            "label":      label,
            "confidence": round(max(combined_real, combined_fake), 4),
            "scores":     {"real": round(combined_real, 4), "fake": round(combined_fake, 4)},
            "method":     "resnet+forensics",
            "resnet_scores":   {"real": round(resnet_probs[0], 4), "fake": round(resnet_probs[1], 4)},
            "forensics_score": round(forensics_fake, 4),
            "forensics":       forensics_detail,
        }

    elif mode == "forensics":
        forensics_fake, flags, forensics_detail = _run_forensics(image)
        real_prob = round(1.0 - forensics_fake, 4)
        fake_prob = round(forensics_fake, 4)

        if forensics_fake < 0.20 and flags == 0:
            label = "real"
        elif forensics_fake > 0.50 or flags >= 2:
            label = "fake"
        else:
            label = "suspicious"

        return {
            "label":      label,
            "confidence": round(max(real_prob, fake_prob), 4),
            "scores":     {"real": real_prob, "fake": fake_prob},
            "method":     "forensics",
            "forensics":  forensics_detail,
        }

    return {"label": "unknown", "confidence": 0.0, "scores": {"real": 0.0, "fake": 0.0}, "skipped": True}
