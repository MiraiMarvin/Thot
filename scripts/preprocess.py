"""
preprocess.py — Validation, augmentation et export des features.

Usage : python scripts/preprocess.py

Charge data/raw_landmarks.csv, applique la data augmentation,
puis exporte :
  data/X.npy       → features float32  (N, 63)
  data/y.npy       → labels  str       (N,)
  data/classes.npy → noms des classes  (K,)

Augmentations appliquées :
  - Flip horizontal (miroir main gauche/droite) : nège les coordonnées X
  - Bruit gaussien léger : simule des variations de détection
"""
import numpy as np
import pandas as pd
import os
import sys

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'raw_landmarks.csv')
X_PATH   = os.path.join(DATA_DIR, 'X.npy')
Y_PATH   = os.path.join(DATA_DIR, 'y.npy')
CLS_PATH = os.path.join(DATA_DIR, 'classes.npy')

MIN_SAMPLES_PER_CLASS = 30
NOISE_STD   = 0.01   # écart-type du bruit gaussien (coordonnées normalisées)
NOISE_COPIES = 2     # nombre de copies bruitées par sample original
# ──────────────────────────────────────────────────────────────────────────────


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  {n_dropped} ligne(s) corrompue(s) supprimée(s)")
    return df


def augment_flip(X: np.ndarray) -> np.ndarray:
    """
    Flip horizontal : inverse le signe des coordonnées X de chaque landmark.
    Les landmarks sont stockés [x0,y0,z0, x1,y1,z1, ...] — on nège les indices
    0,3,6,… (toutes les colonnes X).
    Simule la main miroir, utile si l'utilisateur change de main.
    """
    X_flip = X.copy()
    x_indices = np.arange(0, 63, 3)   # colonnes x0, x1, … x20
    X_flip[:, x_indices] *= -1
    return X_flip


def augment_noise(X: np.ndarray, n_copies: int, std: float) -> np.ndarray:
    """
    Génère n_copies copies de X avec un bruit gaussien additif.
    Simule des variations légères de position des landmarks.
    """
    copies = []
    for _ in range(n_copies):
        noise = np.random.normal(0, std, X.shape).astype(np.float32)
        copies.append(np.clip(X + noise, -1.0, 1.0))
    return np.vstack(copies)


def report_balance(y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    print("\nDistribution des classes (après augmentation) :")
    for cls, cnt in zip(classes, counts):
        bar = '█' * (cnt // 20)
        flag = "  ⚠  INSUFFISANT" if cnt < MIN_SAMPLES_PER_CLASS else ""
        print(f"  {cls:10s} : {cnt:5d}  {bar}{flag}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Erreur : {CSV_PATH} introuvable. Lancez collect_data.py d'abord.")
        sys.exit(1)

    print(f"Chargement de {CSV_PATH} …")
    df = load_and_clean(CSV_PATH)
    n_raw = len(df)
    print(f"  {n_raw} samples bruts valides")

    y_raw = df['label'].values.astype(str)
    X_raw = df.drop(columns=['label']).values.astype(np.float32)

    if X_raw.shape[1] != 63:
        print(f"Erreur : attendu 63 features, obtenu {X_raw.shape[1]}.")
        sys.exit(1)

    # ── Augmentation ─────────────────────────────────────────────────────────
    print("\nAugmentation des données …")

    # 1. Flip horizontal
    X_flip = augment_flip(X_raw)
    print(f"  Flip horizontal     : +{len(X_flip)} samples")

    # 2. Bruit gaussien sur les données originales
    X_noise = augment_noise(X_raw, NOISE_COPIES, NOISE_STD)
    print(f"  Bruit gaussien x{NOISE_COPIES}  : +{len(X_noise)} samples")

    # Assemblage final
    X = np.vstack([X_raw, X_flip, X_noise]).astype(np.float32)
    y = np.concatenate([y_raw, y_raw, np.tile(y_raw, NOISE_COPIES)])

    print(f"\n  Total : {n_raw} → {len(X)} samples ({len(X)//n_raw}x)")

    report_balance(y)

    # Vérification plage
    vmin, vmax = X.min(), X.max()
    print(f"\nPlage des features : [{vmin:.4f}, {vmax:.4f}]")

    classes = np.unique(y)
    np.save(X_PATH,   X)
    np.save(Y_PATH,   y)
    np.save(CLS_PATH, classes)

    print(f"\nFichiers exportés :")
    print(f"  X       → {X_PATH}   {X.shape}")
    print(f"  y       → {Y_PATH}   {y.shape}")
    print(f"  classes → {CLS_PATH} {classes.tolist()}")
    print("\nPrêt pour train.py")


if __name__ == '__main__':
    main()
