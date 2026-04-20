"""
collect_data.py — Collecte des données d'entraînement via webcam.

Usage : python scripts/collect_data.py

Correspondance touches → signes (modifiable dans SIGN_MAP) :
  a → A    b → B    c → C    d → D    e → E
  h → hello   y → yes   n → no   t → thanks   p → please

Appuyez sur la touche correspondante quand votre main fait le signe.
Chaque appui enregistre 1 sample normalisé dans data/raw_landmarks.csv.
Objectif recommandé : ≥ 150 samples par signe.
'q' pour quitter.
"""
import cv2
import mediapipe as mp
import csv
import os
import sys
import time

# Permet d'importer utils/ depuis scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_utils import extract_landmarks, normalize_landmarks

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH  = os.path.join(DATA_DIR, 'raw_landmarks.csv')
CAM_INDEX = 0   # modifier si la webcam n'est pas l'index 0

# Mapping touche (ord) → label du signe
SIGN_MAP = {
    ord('a'): 'A',
    ord('b'): 'B',
    ord('c'): 'C',
    ord('d'): 'D',
    ord('e'): 'E',
    ord('h'): 'hello',
    ord('y'): 'yes',
    ord('n'): 'no',
    ord('t'): 'thanks',
    ord('p'): 'please',
}

# En-tête CSV : label + x0,y0,z0 … x20,y20,z20 (63 features normalisées)
HEADER = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]
# ──────────────────────────────────────────────────────────────────────────────


def ensure_csv(path: str):
    """Crée le CSV avec l'en-tête s'il n'existe pas encore."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)


def count_samples(path: str) -> dict:
    """Lit le CSV et retourne {label: nombre_de_samples}."""
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            counts[row['label']] = counts.get(row['label'], 0) + 1
    return counts


def main():
    ensure_csv(CSV_PATH)

    # MediaPipe Hands en mode streaming (static_image_mode=False)
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    hands_sol = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
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

    with open(CSV_PATH, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Effet miroir naturel pour l'utilisateur
            frame = cv2.flip(frame, 1)

            # MediaPipe attend du RGB ; on désactive writeable pour éviter une copie
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_sol.process(rgb)
            rgb.flags.writeable = True

            hand_ok = results.multi_hand_landmarks is not None

            # Dessin des landmarks si une main est détectée
            if hand_ok:
                for hlm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            # ── HUD ──────────────────────────────────────────────────────────
            status = "Main detectee" if hand_ok else "Aucune main"
            color  = (0, 210, 0)    if hand_ok else (0, 0, 210)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Compteur de samples par signe (colonne gauche)
            counts = count_samples(CSV_PATH)
            y_off  = 60
            for sign in sorted(set(SIGN_MAP.values())):
                n = counts.get(sign, 0)
                bar = '|' * (n // 10)
                cv2.putText(frame, f"{sign:8s}: {n:3d}  {bar}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
                y_off += 22

            # Message de feedback temporaire (1.5 s)
            if time.time() - feedback_time < 1.5:
                cv2.putText(frame, feedback_msg, (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # ─────────────────────────────────────────────────────────────────

            cv2.imshow("Collecte — Sign Language", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key in SIGN_MAP:
                if hand_ok:
                    # Extraction + normalisation + écriture CSV
                    raw    = extract_landmarks(results.multi_hand_landmarks[0])
                    normed = normalize_landmarks(raw)
                    label  = SIGN_MAP[key]
                    writer.writerow([label] + normed.tolist())
                    csv_file.flush()
                    feedback_msg  = f"[+] '{label}' enregistre"
                    feedback_time = time.time()
                    print(f"  Sample : {label}  (total {counts.get(label, 0)+1})")
                else:
                    feedback_msg  = "Aucune main — sample ignore"
                    feedback_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    hands_sol.close()
    print(f"\nCSV : {CSV_PATH}")
    print("Samples par signe :", count_samples(CSV_PATH))


if __name__ == '__main__':
    main()
