"""
build_asl_dataset.py — Dataset ASL Alphabet complet (26 lettres) via Kaggle.

Télécharge le dataset 'grassknoted/asl-alphabet' (87k images),
extrait les landmarks MediaPipe sur chaque image et les ajoute à
data/raw_landmarks.csv (sans écraser les signes custom existants).

Prérequis :
  pip install kaggle
  Placer ~/.kaggle/kaggle.json (Kaggle → Account → API → Create New Token)
  Accepter les règles du dataset : kaggle.com/datasets/grassknoted/asl-alphabet

Usage : python scripts/build_asl_dataset.py
"""
import os, sys, csv, time, shutil, urllib.request
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')
CSV_PATH   = os.path.join(DATA_DIR, 'raw_landmarks.csv')
MP_MODEL   = os.path.join(MODELS_DIR, 'hand_landmarker.task')
TMP_ZIP    = os.path.join(DATA_DIR, 'asl-alphabet.zip')

KAGGLE_DATASET    = 'grassknoted/asl-alphabet'
SAMPLES_PER_CLASS = 3000   # images traitées par lettre → 26×3000 = 78000 samples
LETTERS           = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HEADER = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, ROOT)
from utils.hand_utils import extract_landmarks, normalize_landmarks


def check_kaggle():
    """Vérifie que kaggle est installé et les credentials présents."""
    try:
        import kaggle  # noqa
    except ImportError:
        print("Erreur : le package 'kaggle' n'est pas installé.")
        print("  → pip install kaggle")
        sys.exit(1)

    cred = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(cred):
        print("Erreur : ~/.kaggle/kaggle.json introuvable.")
        print("  → Kaggle → Account → API → Create New Token")
        print("  → mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  → chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)


def download_dataset() -> str:
    """Télécharge et décompresse le dataset. Retourne le dossier racine."""
    import kaggle

    dest = os.path.join(DATA_DIR, 'asl_alphabet_train')
    nested = os.path.join(dest, 'asl_alphabet_train')
    if os.path.exists(nested) and os.listdir(nested):
        print(f"Dataset déjà présent : {nested}")
        return nested
    if os.path.exists(dest) and os.listdir(dest):
        print(f"Dataset déjà présent : {dest}")
        return dest

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Téléchargement de '{KAGGLE_DATASET}' (~1 GB)…")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET, path=DATA_DIR, unzip=True, quiet=False
    )
    print("Téléchargement terminé.")

    # Le dataset se décompresse dans asl_alphabet_train/asl_alphabet_train/
    # ou directement dans asl_alphabet_train/
    nested = os.path.join(DATA_DIR, 'asl_alphabet_train', 'asl_alphabet_train')
    if os.path.exists(nested):
        return nested
    return dest


def find_letter_dir(root: str, letter: str) -> str | None:
    """Cherche le dossier de la lettre (insensible à la casse)."""
    for name in os.listdir(root):
        if name.upper() == letter.upper() and os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    return None


def ensure_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(HEADER)


def existing_labels() -> set:
    """Retourne les labels déjà présents dans le CSV."""
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        return {row['label'] for row in reader}


def process_dataset(dataset_root: str):
    import cv2
    import mediapipe as mp

    # Télécharge le modèle MediaPipe si absent
    if not os.path.exists(MP_MODEL):
        print("Téléchargement du modèle MediaPipe…")
        urllib.request.urlretrieve(MODEL_URL, MP_MODEL)

    letters_to_process = LETTERS[:]

    print(f"\nLettres à traiter : {letters_to_process}")
    print(f"Samples par lettre : {SAMPLES_PER_CLASS}\n")

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    # Mode IMAGE (pas de timestamp, pas de contrainte d'ordre)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.4,
    )

    total_ok = 0
    t0       = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        with open(CSV_PATH, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            for letter in letters_to_process:
                letter_dir = find_letter_dir(dataset_root, letter)
                if not letter_dir:
                    print(f"  ⚠  {letter} : dossier introuvable dans {dataset_root}")
                    continue

                images = sorted([
                    f for f in os.listdir(letter_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])[:SAMPLES_PER_CLASS]

                n_ok = 0
                for img_name in images:
                    img_path = os.path.join(letter_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results  = landmarker.detect(mp_image)

                    if not results.hand_landmarks:
                        continue

                    raw    = extract_landmarks(results.hand_landmarks[0])
                    normed = normalize_landmarks(raw)
                    writer.writerow([letter] + normed.tolist())
                    n_ok  += 1

                csv_file.flush()
                total_ok += n_ok
                pct = n_ok / max(len(images), 1) * 100
                bar = '█' * (n_ok // 10) + '░' * ((len(images) - n_ok) // 10)
                print(f"  {letter}  {n_ok:3d}/{len(images)}  ({pct:.0f}%)  {bar}")

    elapsed = time.time() - t0
    print(f"\nTotal : {total_ok} samples ajoutés en {elapsed:.0f}s")
    print(f"CSV   : {CSV_PATH}")


def main():
    print("=" * 55)
    print("  Build ASL Dataset — 26 lettres + signes custom")
    print("=" * 55)

    check_kaggle()
    dataset_root = download_dataset()
    ensure_csv_header()
    process_dataset(dataset_root)

    print("\nÉtapes suivantes :")
    print("  python scripts/preprocess.py")
    print("  python scripts/train.py")
    print("  python scripts/inference_web.py")


if __name__ == '__main__':
    main()
