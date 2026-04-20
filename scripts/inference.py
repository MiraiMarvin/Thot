"""
inference.py — Inférence temps réel + synthèse vocale.
Utilise la MediaPipe Tasks API (mediapipe >= 0.10).

Usage : python scripts/inference.py
'q' pour quitter.
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import sys
import collections
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_utils import extract_landmarks, normalize_landmarks
from utils.tts_utils   import TTSEngine

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.joblib')
ENC_PATH   = os.path.join(MODELS_DIR, 'label_encoder.joblib')
MP_MODEL   = os.path.join(MODELS_DIR, 'hand_landmarker.task')
CAM_INDEX  = 0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

SMOOTH_WINDOW     = 7
MIN_STABLE_FRAMES = 15
TTS_COOLDOWN      = 2.5
CONF_THRESHOLD    = 0.65

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

C_GREEN  = (0, 215, 0)
C_ORANGE = (0, 165, 255)
C_GRAY   = (80, 80, 80)
C_WHITE  = (240, 240, 240)
# ──────────────────────────────────────────────────────────────────────────────


def download_model(path: str):
    if os.path.exists(path):
        return
    print(f"Téléchargement du modèle MediaPipe …")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  Modèle sauvegardé : {path}")


def draw_hand(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 0), 1)


def main():
    download_model(MP_MODEL)

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : modèle RF introuvable ({MODEL_PATH}). Lancez train.py d'abord.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENC_PATH)
    print(f"Modèle RF chargé — {len(le.classes_)} classes : {le.classes_.tolist()}")

    tts = TTSEngine()

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    pred_buffer  = collections.deque(maxlen=SMOOTH_WINDOW)
    stable_count = 0
    last_spoken  = ""
    last_speak_t = 0.0
    fps_ema      = 0.0
    t_prev       = time.perf_counter()
    t_start      = time.time()
    sentence     = []          # historique des mots reconnus
    MAX_SENTENCE = 8           # nombre max de mots affichés

    print("Inférence active — 'q' quitter | 'ESPACE' effacer la phrase.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms   = int((time.time() - t_start) * 1000)
            results = landmarker.detect_for_video(mp_image, ts_ms)

            prediction = None
            confidence = 0.0
            hand_found = len(results.hand_landmarks) > 0

            if hand_found:
                lms    = results.hand_landmarks[0]
                draw_hand(frame, lms, h, w)
                raw    = extract_landmarks(lms)
                normed = normalize_landmarks(raw).reshape(1, -1)
                proba  = clf.predict_proba(normed)[0]
                top_i  = int(np.argmax(proba))
                confidence = float(proba[top_i])
                if confidence >= CONF_THRESHOLD:
                    prediction = le.classes_[top_i]

            # ── Lissage ──────────────────────────────────────────────────────
            pred_buffer.append(prediction)
            smoothed = None
            if len(pred_buffer) == SMOOTH_WINDOW:
                best, cnt = collections.Counter(pred_buffer).most_common(1)[0]
                if best is not None and cnt >= (SMOOTH_WINDOW // 2 + 1):
                    smoothed = best

            if smoothed is not None and smoothed == prediction:
                stable_count += 1
            else:
                stable_count = 0

            # ── TTS + historique ─────────────────────────────────────────────
            now = time.time()
            if (stable_count >= MIN_STABLE_FRAMES
                    and smoothed is not None
                    and now - last_speak_t > TTS_COOLDOWN):
                tts.speak(smoothed)
                last_spoken  = smoothed
                last_speak_t = now
                stable_count = 0
                sentence.append(smoothed)
                if len(sentence) > MAX_SENTENCE:
                    sentence.pop(0)

            # ── FPS ──────────────────────────────────────────────────────────
            t_now   = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
            t_prev  = t_now

            # ── HUD ──────────────────────────────────────────────────────────
            cv2.putText(frame, f"FPS {fps_ema:.0f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

            box_y1, box_y2 = h - 75, h - 20
            box_x1, box_x2 = w//2 - 150, w//2 + 150

            if smoothed:
                color = C_GREEN if confidence >= 0.85 else C_ORANGE
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (30,30,30), -1)
                cv2.putText(frame, f"{smoothed}  {confidence*100:.0f}%",
                            (box_x1+15, box_y2-12),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
            elif hand_found:
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (30,30,30), -1)
                cv2.putText(frame, "Confiance faible…", (box_x1+15, box_y2-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_ORANGE, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Montrez votre main", (w//2-130, box_y2-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_GRAY, 1, cv2.LINE_AA)

            # Barre de stabilité
            bar_w = int((min(stable_count, MIN_STABLE_FRAMES) / MIN_STABLE_FRAMES)
                        * (box_x2 - box_x1 - 4))
            if bar_w > 0:
                cv2.rectangle(frame,
                              (box_x1+2, box_y2+2), (box_x1+2+bar_w, box_y2+8),
                              (0, 180, 255), -1)

            # ── Historique de phrase (haut de l'écran) ────────────────────────
            if sentence:
                phrase = ' '.join(sentence)
                # Fond semi-transparent
                cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
                cv2.putText(frame, phrase, (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 50), 2, cv2.LINE_AA)
            # ─────────────────────────────────────────────────────────────────

            cv2.imshow("Sign Language Translator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):   # ESPACE = effacer la phrase
                sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
