"""
inference.py — Inférence temps réel + synthèse vocale.

Usage : python scripts/inference.py

Charge le modèle entraîné, ouvre la webcam, prédit le signe à chaque frame
et déclenche la TTS dès qu'une prédiction est stable sur plusieurs frames.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_utils import extract_landmarks, normalize_landmarks
from utils.tts_utils   import TTSEngine

# ─── Configuration ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.joblib')
ENC_PATH   = os.path.join(MODELS_DIR, 'label_encoder.joblib')
CAM_INDEX  = 0

# Lissage temporel : vote majoritaire sur les N dernières frames
SMOOTH_WINDOW    = 7
# Nombre de frames stables consécutives avant de déclencher la TTS
MIN_STABLE_FRAMES = 15
# Délai minimum (secondes) entre deux synthèses vocales
TTS_COOLDOWN     = 2.5
# Seuil de confiance (probabilité RF) en dessous duquel on n'affiche rien
CONF_THRESHOLD   = 0.65
# ──────────────────────────────────────────────────────────────────────────────

# Palette couleurs BGR
C_GREEN  = (0, 215, 0)
C_ORANGE = (0, 165, 255)
C_GRAY   = (80, 80, 80)
C_WHITE  = (240, 240, 240)


def draw_pill(img, pt1, pt2, color, alpha=0.55):
    """
    Fond semi-transparent arrondi pour le HUD principal.
    Technique : blend d'un rectangle sur une copie du frame.
    """
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = (y2 - y1) // 2
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.circle(overlay,    (x1+r, (y1+y2)//2), r, color, -1)
    cv2.circle(overlay,    (x2-r, (y1+y2)//2), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : modele introuvable ({MODEL_PATH}).")
        print("Lancez train.py d'abord.")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENC_PATH)
    print(f"Modele charge — {len(le.classes_)} classes : {le.classes_.tolist()}")

    tts = TTSEngine()

    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_style  = mp.solutions.drawing_styles
    hands_sol = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Réduire le buffer interne à 1 frame pour minimiser la latence
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Buffers et état
    pred_buffer  = collections.deque(maxlen=SMOOTH_WINDOW)
    stable_count = 0
    last_spoken  = ""
    last_speak_t = 0.0
    fps_ema      = 0.0       # moyenne mobile exponentielle du FPS
    t_prev       = time.perf_counter()

    print("Inference active — appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Traitement MediaPipe (RGB, no-copy pour économiser de la mémoire)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_sol.process(rgb)
        rgb.flags.writeable = True

        prediction  = None
        confidence  = 0.0
        hand_found  = results.multi_hand_landmarks is not None

        if hand_found:
            hlm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hlm, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style(),
            )
            # Extraction → normalisation → prédiction RF
            raw    = extract_landmarks(hlm)
            normed = normalize_landmarks(raw).reshape(1, -1)
            proba  = clf.predict_proba(normed)[0]
            top_i  = int(np.argmax(proba))
            confidence = float(proba[top_i])
            if confidence >= CONF_THRESHOLD:
                prediction = le.classes_[top_i]

        # ── Lissage temporel (vote majoritaire) ──────────────────────────────
        pred_buffer.append(prediction)
        smoothed = None
        if len(pred_buffer) == SMOOTH_WINDOW:
            counts_pred = collections.Counter(pred_buffer)
            best, cnt   = counts_pred.most_common(1)[0]
            if best is not None and cnt >= (SMOOTH_WINDOW // 2 + 1):
                smoothed = best

        # Compteur de stabilité : incrémenté seulement si la prédiction ne change pas
        if smoothed is not None and smoothed == prediction:
            stable_count += 1
        else:
            stable_count = 0

        # ── Déclenchement TTS ────────────────────────────────────────────────
        now = time.time()
        if (stable_count >= MIN_STABLE_FRAMES
                and smoothed is not None
                and now - last_speak_t > TTS_COOLDOWN):
            tts.speak(smoothed)
            last_spoken  = smoothed
            last_speak_t = now
            stable_count = 0   # repart à zéro pour éviter les déclenchements en rafale

        # ── Calcul FPS (EMA) ─────────────────────────────────────────────────
        t_now   = time.perf_counter()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
        t_prev  = t_now

        # ── HUD ──────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # FPS coin supérieur gauche
        cv2.putText(frame, f"FPS {fps_ema:.0f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

        # Bandeau de prédiction centré en bas
        box_y1, box_y2 = h - 75, h - 20
        box_x1, box_x2 = w//2 - 150, w//2 + 150

        if smoothed:
            box_color = C_GREEN if confidence >= 0.85 else C_ORANGE
            draw_pill(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20))
            label_str = f"{smoothed}   {confidence*100:.0f}%"
            cv2.putText(frame, label_str, (box_x1 + 20, box_y2 - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, box_color, 2, cv2.LINE_AA)
        elif hand_found:
            draw_pill(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20))
            cv2.putText(frame, "Confiance faible…", (box_x1 + 15, box_y2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_ORANGE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Montrez votre main", (w//2 - 130, box_y2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_GRAY, 1, cv2.LINE_AA)

        # Barre de stabilité (progression vers MIN_STABLE_FRAMES)
        bar_max = box_x2 - box_x1 - 4
        bar_w   = int((min(stable_count, MIN_STABLE_FRAMES) / MIN_STABLE_FRAMES) * bar_max)
        if bar_w > 0:
            cv2.rectangle(frame,
                          (box_x1 + 2, box_y2 + 2),
                          (box_x1 + 2 + bar_w, box_y2 + 8),
                          (0, 180, 255), -1)
        # ─────────────────────────────────────────────────────────────────────

        cv2.imshow("Sign Language Translator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_sol.close()


if __name__ == '__main__':
    main()
