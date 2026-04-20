"""Utilitaires d'extraction et normalisation des landmarks de main."""
import numpy as np


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Retourne un vecteur plat (63,) contenant (x, y, z) des 21 landmarks
    dans les coordonnées normalisées MediaPipe (0–1 relatives au frame).
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """
    Normalise les landmarks par rapport au poignet (landmark 0) pour
    obtenir une représentation invariante à la position et à l'échelle.

    Étapes :
      1. Translate : soustrait le poignet pour le mettre à l'origine (0,0,0)
      2. Scale     : divise par la distance max entre le poignet et un autre point

    Retourne un vecteur float32 (63,) utilisable directement comme feature.
    """
    coords = raw.reshape(21, 3)
    wrist  = coords[0].copy()
    coords = coords - wrist                         # translation → poignet = origine

    max_dist = np.linalg.norm(coords, axis=1).max()
    if max_dist > 1e-6:
        coords /= max_dist                          # scale → distances comprises dans [-1, 1]

    return coords.flatten().astype(np.float32)
