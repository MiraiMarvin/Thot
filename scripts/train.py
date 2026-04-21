"""
train.py — Entraînement RF vs MLP, export du meilleur modèle.

Usage : python scripts/train.py

Compare Random Forest et MLP par cross-validation,
entraîne le meilleur sur 100% des données et exporte :
  models/best_model.joblib      → meilleur classifieur
  models/label_encoder.joblib   → encodeur de labels
  models/rf_model.joblib        → alias du meilleur modèle (lu par inference.py)
"""
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
X_PATH     = os.path.join(DATA_DIR,   'X.npy')
Y_PATH     = os.path.join(DATA_DIR,   'y.npy')
BEST_PATH  = os.path.join(MODELS_DIR, 'best_model.joblib')
RF_PATH    = os.path.join(MODELS_DIR, 'rf_model.joblib')
ENC_PATH   = os.path.join(MODELS_DIR, 'label_encoder.joblib')
CV_FOLDS   = 5

RF_PARAMS = {
    'n_estimators'     : 300,
    'max_depth'        : None,
    'min_samples_split': 4,
    'min_samples_leaf' : 2,
    'n_jobs'           : -1,
    'random_state'     : 42,
    'class_weight'     : 'balanced',
}

# MLP dans un pipeline avec StandardScaler (obligatoire pour les réseaux de neurones)
MLP_PARAMS = {
    'hidden_layer_sizes': (256, 128, 64),
    'activation'        : 'relu',
    'solver'            : 'adam',
    'alpha'             : 1e-4,       # régularisation L2
    'batch_size'        : 256,
    'learning_rate_init': 1e-3,
    'max_iter'          : 300,
    'early_stopping'    : True,
    'validation_fraction': 0.1,
    'n_iter_no_change'  : 15,
    'random_state'      : 42,
}
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(name: str, clf, X, y_enc, skf) -> float:
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
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp',    MLPClassifier(**MLP_PARAMS)),
    ])

    acc_rf  = evaluate("Random Forest",  rf,  X, y_enc, skf)
    acc_mlp = evaluate("MLP (256-128-64)", mlp, X, y_enc, skf)

    # ── Choix du meilleur modèle ──────────────────────────────────────────────
    if acc_mlp > acc_rf + 0.005:
        best_name = "MLP"
        best_clf  = mlp
    else:
        best_name = "Random Forest"
        best_clf  = rf

    print(f"\n  → Meilleur modèle : {best_name}")

    # ── Entraînement final sur 100% ───────────────────────────────────────────
    print("\nEntraînement final …")
    best_clf.fit(X, y_enc)

    y_pred = best_clf.predict(X)
    print("\nRapport (train set) :")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    joblib.dump(best_clf, BEST_PATH)
    joblib.dump(best_clf, RF_PATH)   # inference.py charge rf_model.joblib
    joblib.dump(le,       ENC_PATH)

    print(f"\nModèle sauvegardé : {RF_PATH}  ({best_name})")
    print(f"Encodeur          : {ENC_PATH}")
    print("\nPrêt pour inference.py")


if __name__ == '__main__':
    main()
