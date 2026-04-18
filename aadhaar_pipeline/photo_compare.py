"""
Photo comparison between QR-embedded photo and card face crop.
Uses facenet-pytorch (InceptionResnetV1) to compute facial embedding distance.
Falls back to SSIM and MSE if facenet cannot detect a face or fails to load.
"""
import cv2
import numpy as np

# Try importing facenet-pytorch dependencies globally
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _mtcnn = MTCNN(keep_all=False, device=_device)
    # Pretrained weights will download on first usage automatically (~90MB, fast!)
    _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)
    FACENET_AVAILABLE = True
except Exception as e:
    print(f"Facenet not available, falling back to SSIM. Error: {e}")
    FACENET_AVAILABLE = False


def _to_gray_resized(img, size=(200, 200)):
    """Convert image to grayscale 200x200."""
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(bytes(img), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _to_rgb_array(img):
    """Safely convert inputs representing images to RGB numpy arrays."""
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(bytes(img), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compare_photos(qr_photo, card_photo):
    """
    Main photo comparison logic.
    Attempts strict deep-learning facial matching (Facenet) first.
    If the face geometry is completely destroyed or library is missing,
    it falls gracefully back to image structure checks (SSIM/MSE).
    """
    if FACENET_AVAILABLE:
        try:
            return _facenet_compare(qr_photo, card_photo)
        except Exception as e:
            print(f"Facenet comparison failed: {e}. Attempting SSIM fallback.")

    return _fallback_ssim_compare(qr_photo, card_photo)


def _facenet_compare(qr_photo, card_photo):
    from PIL import Image

    img1_rgb = _to_rgb_array(qr_photo)
    img2_rgb = _to_rgb_array(card_photo)

    if img1_rgb is None or img2_rgb is None:
        return {"decision": "UNAVAILABLE", "error": "Could not decode photos.", "match": None, "fraud_contribution": 0}

    # Convert to PIL for facenet
    pil1 = Image.fromarray(img1_rgb)
    pil2 = Image.fromarray(img2_rgb)

    # Extract tightly cropped faces and format to tensors
    face1 = _mtcnn(pil1)
    face2 = _mtcnn(pil2)

    # Fallback to SSIM if MTCNN literally cannot find a human face in the layout 
    if face1 is None or face2 is None:
        raise ValueError("No visible face detected by MTCNN in one or both crops")

    # Stack the MTCNN output tensors securely into device memory
    face1 = face1.unsqueeze(0).to(_device)
    face2 = face2.unsqueeze(0).to(_device)

    # Generate the embeddings via Facenet ResNet
    with torch.no_grad():
        emb1 = _resnet(face1)
        emb2 = _resnet(face2)

    # Compute Euclidean distance
    dist = (emb1 - emb2).norm().item()

    # The typical Facenet threshold is ~1.1 for same person
    # You can tune this! 0.9 is strict, 1.2 is forgiving.
    threshold = 1.05

    if dist < threshold:
        decision, match, fraud_contribution = "MATCH", True, 20
    elif dist < threshold + 0.25:
        decision, match, fraud_contribution = "SUSPICIOUS", None, 10
    else:
        decision, match, fraud_contribution = "NO_MATCH", False, 0

    return {
        "match":              match,
        "confidence":         round(max(0, 1.0 - (dist / 2.0)), 4),
        "ssim":               None, 
        "distance":           round(dist, 4),
        "decision":           decision,
        "fraud_contribution": fraud_contribution,
        "error":              None,
    }


def _fallback_ssim_compare(qr_photo, card_photo):
    try:
        from skimage.metrics import structural_similarity as ssim
        g1 = _to_gray_resized(qr_photo)
        g2 = _to_gray_resized(card_photo)

        if g1 is None or g2 is None:
            return {"match": None, "decision": "UNAVAILABLE", "fraud_contribution": 0}

        score, _ = ssim(g1, g2, full=True)

        if score > 0.75:
            decision, match, fraud_contribution = "MATCH", True, 20
        elif score >= 0.50:
            decision, match, fraud_contribution = "SUSPICIOUS", None, 10
        else:
            decision, match, fraud_contribution = "NO_MATCH", False, 0

        return {
            "match": match,
            "confidence": round(float(score), 4),
            "ssim": round(float(score), 4),
            "decision": decision,
            "fraud_contribution": fraud_contribution,
            "error": "Facenet skipped or failed — SSIM fallback used",
        }
    except ImportError:
        return {"match": None, "decision": "UNAVAILABLE", "fraud_contribution": 0, "error": "No comparison tech available"}
