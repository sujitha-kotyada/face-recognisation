"""
Real-Time Face Recognition App
===============================
Uses a pre-trained Keras model (keras_model.h5) and OpenCV to detect
and recognise faces from a live webcam feed.
"""

import os
import cv2
import numpy as np
from tf_keras.models import load_model
from tf_keras.layers import DepthwiseConv2D

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
INPUT_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65

# ── Colours & fonts ─────────────────────────────────────────────────────────
BOX_COLOR = (0, 255, 0)          # Green bounding box
TEXT_COLOR = (255, 255, 255)     # White text
BG_COLOR = (0, 255, 0)          # Green text background
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


# ── Helper functions ─────────────────────────────────────────────────────────

def load_labels(path):
    """Read labels.txt and return a list of class names."""
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # Format: "<index> <name>"  →  keep only the name
                parts = line.split(" ", 1)
                labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels


def preprocess_face(face_img, target_size):
    """
    Prepare a BGR face crop for the Keras model.

    Steps:
        1. Convert BGR -> RGB
        2. Resize to (target_size x target_size)
        3. Normalise pixel values to [-1, 1] (Teachable Machine format)
        4. Add batch dimension -> (1, H, W, 3)
    """
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size))
    normalised = (resized.astype("float32") / 127.5) - 1.0
    return np.expand_dims(normalised, axis=0)


class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load the Keras model (compile=False avoids optimizer warnings)
    print("[INFO] Loading model ...")
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
    )

    # Load class labels
    labels = load_labels(LABELS_PATH)
    print(f"[INFO] Loaded {len(labels)} labels: {labels}")

    # Initialise the Haar Cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
        )

        # ── Predict on the FULL frame (Teachable Machine trains on full frames) ──
        processed = preprocess_face(frame, INPUT_SIZE)
        predictions = model.predict(processed, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]

        # Build the label string
        if confidence >= CONFIDENCE_THRESHOLD:
            name = labels[class_idx] if class_idx < len(labels) else "Unknown"
            label = f"{name} - {confidence * 100:.1f}%"
        else:
            label = "Unknown"

        for (x, y, w, h) in faces:

            # ── Draw bounding box ────────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

            # ── Draw label with background ───────────────────────────────
            (text_w, text_h), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            # Background rectangle above the bounding box
            cv2.rectangle(
                frame,
                (x, y - text_h - baseline - 6),
                (x + text_w + 4, y),
                BG_COLOR,
                cv2.FILLED,
            )
            # Text
            cv2.putText(
                frame,
                label,
                (x + 2, y - baseline - 4),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )

        # Show the annotated frame
        cv2.imshow("Face Recognition", frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()