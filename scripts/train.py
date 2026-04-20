"""
train.py — Entraînement du classificateur Random Forest.

Usage : python scripts/train.py

Charge data/X.npy et data/y.npy (produits par preprocess.py),
évalue le modèle par cross-validation stratifiée, entraîne sur 100%
des données, puis exporte :
  models/rf_model.joblib      → le classifieur
  models/label_encoder.joblib → l'encodeur de labels
"""
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
X_PATH     = os.path.join(DATA_DIR,   'X.npy')
Y_PATH     = os.path.join(DATA_DIR,   'y.npy')
MODEL_PATH = os.path.join(MODELS_DIR, 'rf_model.joblib')
ENC_PATH   = os.path.join(MODELS_DIR, 'label_encoder.joblib')

RF_PARAMS = {
    'n_estimators'   : 200,        # 200 arbres : bon compromis précision/vitesse
    'max_depth'      : None,       # arbres complets → meilleure expressivité
    'min_samples_split': 4,
    'min_samples_leaf' : 2,
    'n_jobs'         : -1,         # parallélise sur tous les cœurs CPU
    'random_state'   : 42,
    'class_weight'   : 'balanced', # compense le déséquilibre éventuel des classes
}
CV_FOLDS = 5
# ──────────────────────────────────────────────────────────────────────────────


def main():
    for p in (X_PATH, Y_PATH):
        if not os.path.exists(p):
            print(f"Erreur : {p} introuvable. Lancez preprocess.py d'abord.")
            sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    print(f"Données chargées : X={X.shape}, classes={np.unique(y).tolist()}")

    # Encode les labels string → entiers (requis par sklearn)
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Cross-validation : estimation des performances réelles ───────────────
    print(f"\nCross-validation {CV_FOLDS} folds (stratifiée) …")
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    clf_cv = RandomForestClassifier(**RF_PARAMS)
    scores = cross_validate(
        clf_cv, X, y_enc, cv=skf,
        scoring=['accuracy', 'f1_macro'],
        n_jobs=-1
    )
    print(f"  Accuracy : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"  F1-macro : {scores['test_f1_macro'].mean():.3f} ± {scores['test_f1_macro'].std():.3f}")

    if scores['test_accuracy'].mean() < 0.80:
        print("  ⚠  Accuracy faible — collectez plus de samples ou vérifiez la qualité des gestes.")

    # ── Entraînement final sur 100 % des données ─────────────────────────────
    print("\nEntraînement final …")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X, y_enc)

    # Rapport sur les données d'entraînement (overfit attendu ~100%)
    y_pred = clf.predict(X)
    print("\nRapport entraînement (overfit intentionnel sur train set) :")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # ── Top features ─────────────────────────────────────────────────────────
    feat_names  = [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]
    importances = clf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    print("Top 10 features :")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feat_names[idx]:5s} : {importances[idx]:.4f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENC_PATH)
    print(f"\nModele sauvegarde   : {MODEL_PATH}")
    print(f"Encodeur sauvegarde : {ENC_PATH}")
    print("\nPret pour inference.py")


if __name__ == '__main__':
    main()
