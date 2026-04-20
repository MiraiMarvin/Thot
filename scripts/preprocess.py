"""
preprocess.py — Validation et export des features d'entraînement.

Usage : python scripts/preprocess.py

Charge data/raw_landmarks.csv (produit par collect_data.py),
vérifie la qualité des données et exporte :
  data/X.npy       → features float32  (N, 63)
  data/y.npy       → labels  str       (N,)
  data/classes.npy → noms des classes  (K,)

Les landmarks sont déjà normalisés à la collecte (hand_utils.normalize_landmarks).
Ce script se charge uniquement de la validation et de la mise en forme finale.
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

MIN_SAMPLES_PER_CLASS = 30   # avertissement si une classe est sous ce seuil
# ──────────────────────────────────────────────────────────────────────────────


def load_and_clean(path: str) -> pd.DataFrame:
    """Charge le CSV et supprime les lignes corrompues (NaN, inf)."""
    df = pd.read_csv(path)
    n_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  {n_dropped} ligne(s) corrompue(s) supprimee(s)")
    return df


def report_balance(y: np.ndarray):
    """Affiche la distribution des classes et avertit en cas de déséquilibre."""
    classes, counts = np.unique(y, return_counts=True)
    print("\nDistribution des classes :")
    for cls, cnt in zip(classes, counts):
        bar = '█' * (cnt // 5)
        flag = "  ⚠  INSUFFISANT" if cnt < MIN_SAMPLES_PER_CLASS else ""
        print(f"  {cls:10s} : {cnt:4d}  {bar}{flag}")
    ratio = counts.max() / (counts.min() + 1e-9)
    if ratio > 3:
        print(f"\n  ⚠  Déséquilibre classes max/min={ratio:.1f}x — collectez plus de données.")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Erreur : {CSV_PATH} introuvable.")
        print("Lancez collect_data.py pour collecter des données d'abord.")
        sys.exit(1)

    print(f"Chargement de {CSV_PATH} …")
    df = load_and_clean(CSV_PATH)
    print(f"  {len(df)} samples valides")

    if len(df) == 0:
        print("Erreur : aucun sample valide dans le CSV.")
        sys.exit(1)

    y = df['label'].values.astype(str)
    X = df.drop(columns=['label']).values.astype(np.float32)

    # Vérification de la dimensionnalité attendue (21 landmarks × 3 axes)
    if X.shape[1] != 63:
        print(f"Erreur : attendu 63 features, obtenu {X.shape[1]}.")
        print("Vérifiez que collect_data.py utilise bien normalize_landmarks().")
        sys.exit(1)

    report_balance(y)

    # Vérification de la plage des valeurs normalisées (doit être dans [-1, 1])
    vmin, vmax = X.min(), X.max()
    print(f"\nPlage des features : [{vmin:.4f}, {vmax:.4f}]")
    if vmax > 2.0 or vmin < -2.0:
        print("  ⚠  Valeurs hors plage — la normalisation a peut-être échoué.")

    classes = np.unique(y)
    np.save(X_PATH,   X)
    np.save(Y_PATH,   y)
    np.save(CLS_PATH, classes)

    print(f"\nFichiers exportes :")
    print(f"  X       → {X_PATH}   {X.shape}")
    print(f"  y       → {Y_PATH}   {y.shape}")
    print(f"  classes → {CLS_PATH} {classes.tolist()}")
    print("\nPret pour train.py")


if __name__ == '__main__':
    main()
