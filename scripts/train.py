"""
train.py — Entraînement et comparaison RF vs SVM.

Usage : python scripts/train.py

Compare Random Forest et SVM par cross-validation,
entraîne le meilleur sur 100% des données et exporte :
  models/best_model.joblib      → meilleur classifieur
  models/label_encoder.joblib   → encodeur de labels
  models/rf_model.joblib        → Random Forest (toujours sauvegardé)
"""
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
X_PATH     = os.path.join(DATA_DIR,   'X.npy')
Y_PATH     = os.path.join(DATA_DIR,   'y.npy')
BEST_PATH  = os.path.join(MODELS_DIR, 'best_model.joblib')
RF_PATH    = os.path.join(MODELS_DIR, 'rf_model.joblib')   # compat inference.py
ENC_PATH   = os.path.join(MODELS_DIR, 'label_encoder.joblib')
CV_FOLDS   = 5

RF_PARAMS = {
    'n_estimators'    : 200,
    'max_depth'       : None,
    'min_samples_split': 4,
    'min_samples_leaf' : 2,
    'n_jobs'          : -1,
    'random_state'    : 42,
    'class_weight'    : 'balanced',
}

# SVM avec StandardScaler dans un pipeline (SVM sensible à l'échelle)
SVM_PARAMS = {
    'C'          : 10,
    'kernel'     : 'rbf',
    'gamma'      : 'scale',
    'probability': True,    # nécessaire pour predict_proba dans inference.py
    'class_weight': 'balanced',
}
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(name: str, clf, X, y_enc, skf) -> float:
    """Cross-valide un classifieur et retourne l'accuracy moyenne."""
    scores = cross_validate(clf, X, y_enc, cv=skf,
                            scoring=['accuracy', 'f1_macro'], n_jobs=-1)
    acc = scores['test_accuracy'].mean()
    f1  = scores['test_f1_macro'].mean()
    print(f"  {name:20s}  accuracy={acc:.3f} ± {scores['test_accuracy'].std():.3f}"
          f"   f1={f1:.3f} ± {scores['test_f1_macro'].std():.3f}")
    return acc


def main():
    for p in (X_PATH, Y_PATH):
        if not os.path.exists(p):
            print(f"Erreur : {p} introuvable. Lancez preprocess.py d'abord.")
            sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    print(f"Données : X={X.shape}  classes={np.unique(y).tolist()}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # ── Comparaison des modèles ───────────────────────────────────────────────
    print(f"\nCross-validation {CV_FOLDS} folds :")

    rf  = RandomForestClassifier(**RF_PARAMS)
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc',    SVC(**SVM_PARAMS)),
    ])

    acc_rf  = evaluate("Random Forest",  rf,  X, y_enc, skf)
    acc_svm = evaluate("SVM (RBF)",      svm, X, y_enc, skf)

    # ── Choix du meilleur modèle ──────────────────────────────────────────────
    if acc_svm > acc_rf + 0.01:   # SVM retenu seulement s'il est clairement meilleur
        best_name = "SVM"
        best_clf  = svm
    else:
        best_name = "Random Forest"
        best_clf  = rf

    print(f"\n  → Meilleur modèle : {best_name}")

    # ── Entraînement final sur 100% ───────────────────────────────────────────
    print("\nEntraînement final …")
    best_clf.fit(X, y_enc)
    rf.fit(X, y_enc)   # toujours entraîner RF pour compatibilité inference.py

    y_pred = best_clf.predict(X)
    print("\nRapport (train set) :")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # Top features RF (même si SVM est meilleur, utile pour comprendre les données)
    if hasattr(rf, 'feature_importances_'):
        feat_names  = [f'{ax}{i}' for i in range(21) for ax in ('x', 'y', 'z')]
        importances = rf.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:8]
        print("Top 8 features (RF) :")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {feat_names[idx]:5s} : {importances[idx]:.4f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    joblib.dump(best_clf, BEST_PATH)
    joblib.dump(rf,       RF_PATH)    # compat inference.py (charge rf_model.joblib)
    joblib.dump(le,       ENC_PATH)

    # inference.py charge rf_model.joblib — on y met le meilleur modèle
    joblib.dump(best_clf, RF_PATH)

    print(f"\nModèle sauvegardé : {RF_PATH}  ({best_name})")
    print(f"Encodeur          : {ENC_PATH}")
    print("\nPrêt pour inference.py")


if __name__ == '__main__':
    main()
