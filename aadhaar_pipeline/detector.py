import cv2
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from ultralytics import YOLO

# these match the class IDs in our data.yaml
# note: the new final_aadhaar_dataset uses a different order
#   0=aadhaar_number, 1=address, 2=dob, 3=face, 4=name, 5=qr_code
# update this if you retrain on the new dataset
CLASS_NAMES = {
    0: "aadhaar_number",
    1: "address",
    2: "dob",
    3: "face",
    4: "name",
    5: "qr_code",
}

# box colors for visualization
BOX_COLORS = {
    "aadhaar_number": (0, 255, 0),
    "name":           (255, 165, 0),
    "dob":            (0, 165, 255),
    "address":        (255, 0, 255),
    "qr_code":        (0, 255, 255),
    "face_photo":     (255, 255, 0),
}


def load_model(weights_path):
    return YOLO(weights_path)


def detect_regions(model, image, conf=0.25):
    """Run YOLO on the image and return a list of detected regions with crops."""
    results = model.predict(source=image, conf=conf, verbose=False)[0]
    h, w = image.shape[:2]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # make sure we don't go out of bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        detections.append({
            "class_id":   cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "confidence": confidence,
            "bbox":       (x1, y1, x2, y2),
            "crop":       image[y1:y2, x1:x2],
        })

    return detections


def get_crops_by_class(detections):
    """Group all crops by their class name for easy lookup later."""
    crops = {}
    for det in detections:
        crops.setdefault(det["class_name"], []).append(det["crop"])
    return crops


def draw_detections(image, detections):
    """Draw boxes and labels on the image so we can see what YOLO found."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = BOX_COLORS.get(det["class_name"], (200, 200, 200))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis
