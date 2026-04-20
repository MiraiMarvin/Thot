"""
collect_data.py — Collecte des données d'entraînement via webcam.
Utilise la MediaPipe Tasks API (mediapipe >= 0.10).

Usage : python scripts/collect_data.py

Touches → signes :
  a→A  b→B  c→C  d→D  e→E  h→hello  y→yes  n→no  t→thanks  p→please
'q' pour quitter.
"""
import cv2
import mediapipe as mp
import csv
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_utils import extract_landmarks, normalize_landmarks

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CSV_PATH   = os.path.join(DATA_DIR,   'raw_landmarks.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'hand_landmarker.task')
CAM_INDEX  = 0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

SIGN_MAP = {
    ord('a'): 'A',     ord('b'): 'B',   ord('c'): 'C',
    ord('d'): 'D',     ord('e'): 'E',   ord('h'): 'hello',
    ord('y'): 'yes',   ord('n'): 'no',  ord('t'): 'thanks',
    ord('p'): 'please',
}

HEADER = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]

# Connexions pour dessiner le squelette de la main manuellement
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
# ──────────────────────────────────────────────────────────────────────────────


def download_model(path: str):
    """Télécharge le modèle hand_landmarker.task si absent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Téléchargement du modèle MediaPipe ({MODEL_URL}) …")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  Modèle sauvegardé : {path}")


def ensure_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)


def count_samples(path: str) -> dict:
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            counts[row['label']] = counts.get(row['label'], 0) + 1
    return counts


def draw_hand(frame, landmarks, h, w):
    """Dessine le squelette de la main sur le frame."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 0), 1)


def main():
    download_model(MODEL_PATH)
    ensure_csv(CSV_PATH)

    # Initialisation Tasks API
    BaseOptions          = mp.tasks.BaseOptions
    HandLandmarker       = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode    = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=== Collecte active ===")
    print("Touches :", {chr(k): v for k, v in SIGN_MAP.items()})
    print("'q' pour quitter\n")

    feedback_msg  = ""
    feedback_time = 0.0
    t_start       = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        with open(CSV_PATH, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame    = cv2.flip(frame, 1)
                h, w     = frame.shape[:2]
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Timestamp en millisecondes pour le mode VIDEO
                ts_ms   = int((time.time() - t_start) * 1000)
                results = landmarker.detect_for_video(mp_image, ts_ms)

                hand_ok = len(results.hand_landmarks) > 0

                if hand_ok:
                    draw_hand(frame, results.hand_landmarks[0], h, w)

                # ── HUD ──────────────────────────────────────────────────────
                status = "Main detectee" if hand_ok else "Aucune main"
                color  = (0, 210, 0) if hand_ok else (0, 0, 210)
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                counts = count_samples(CSV_PATH)
                y_off  = 60
                for sign in sorted(set(SIGN_MAP.values())):
                    n = counts.get(sign, 0)
                    cv2.putText(frame, f"{sign:8s}: {n:3d}", (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                    y_off += 22

                if time.time() - feedback_time < 1.5:
                    cv2.putText(frame, feedback_msg, (10, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # ─────────────────────────────────────────────────────────────

                cv2.imshow("Collecte — Sign Language", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key in SIGN_MAP:
                    if hand_ok:
                        raw    = extract_landmarks(results.hand_landmarks[0])
                        normed = normalize_landmarks(raw)
                        label  = SIGN_MAP[key]
                        writer.writerow([label] + normed.tolist())
                        csv_file.flush()
                        feedback_msg  = f"[+] '{label}' enregistre"
                        feedback_time = time.time()
                        print(f"  Sample : {label}  (total {counts.get(label,0)+1})")
                    else:
                        feedback_msg  = "Aucune main detectee !"
                        feedback_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCSV : {CSV_PATH}")
    print("Samples par signe :", count_samples(CSV_PATH))


if __name__ == '__main__':
    main()
