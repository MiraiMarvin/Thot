# Thot — Sign Language Translator

Traducteur de langue des signes ASL en temps réel via webcam.  
Reconnaissance des 26 lettres de l'alphabet + synthèse vocale + hologramme 3D.

---

## Prérequis

- Python 3.10+
- Une webcam
- Un compte [Kaggle](https://www.kaggle.com) (pour télécharger le dataset)

---

## Installation

```bash
git clone https://github.com/<ton-username>/Thot.git
cd Thot

python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

---

## Première utilisation (pipeline complet)

Ces étapes sont à faire **une seule fois** pour construire et entraîner le modèle.

### 1. Configurer Kaggle

Crée ton token API sur [kaggle.com](https://www.kaggle.com/settings) → *Account* → *API* → *Create New Token*.  
Place le fichier `kaggle.json` dans `~/.kaggle/` :

```bash
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Télécharger et extraire les landmarks du dataset

```bash
python3 scripts/build_asl_dataset.py
```

> Télécharge ~1 GB d'images Kaggle, extrait les landmarks MediaPipe et génère `data/raw_landmarks.csv`.  
> Durée : 20–30 min selon ta machine.

### 3. Préparer les features

```bash
python3 scripts/preprocess.py
```

> Génère `data/X.npy` et `data/y.npy`.

### 4. Entraîner le modèle

```bash
python3 scripts/train.py
```

> Compare Random Forest et MLP, sauvegarde le meilleur dans `models/rf_model.joblib`.

---

## Lancement

```bash
source venv/bin/activate
python3 scripts/inference_web.py
```

Ouvre **http://localhost:5001** dans ton navigateur.

---

## Utilisation

| Action | Description |
|--------|-------------|
| Montre ta main | L'IA reconnaît la lettre ASL et la lit à voix haute |
| Barre de stabilité | Se remplit pendant que tu maintiens le signe |
| Bouton 🎤 | Parle en anglais → l'hologramme montre le signe correspondant |
| `Espace` | Efface la phrase en cours |

---

## Optionnel — Voix naturelle ElevenLabs

Crée un fichier `.env` à la racine :

```
ELEVENLABS_API_KEY=ta_clé_ici
```

Sans clé, le projet utilise `pyttsx3` (voix offline, gratuite).

---

## Structure du projet

```
Thot/
├── scripts/
│   ├── build_asl_dataset.py   # Étape 1 : extraction des landmarks depuis les images
│   ├── preprocess.py          # Étape 2 : augmentation et export des features
│   ├── train.py               # Étape 3 : entraînement du modèle
│   ├── collect_data.py        # Collecte de signes custom via webcam
│   └── inference_web.py       # Lancement de l'interface web
├── utils/
│   ├── hand_utils.py          # Extraction et normalisation des landmarks
│   └── tts_utils.py           # Moteur TTS (ElevenLabs / pyttsx3 / gTTS)
├── templates/
│   └── index.html             # Interface web AR (Three.js + Socket.IO)
├── models/                    # Modèles générés (non versionnés)
├── data/                      # Données générées (non versionnées)
├── requirements.txt
└── .env                       # Clé API ElevenLabs (non versionnée)
```

---

## Fichiers non versionnés (trop lourds pour GitHub)

Ces fichiers sont générés localement via le pipeline ci-dessus :

| Fichier | Taille | Généré par |
|---------|--------|------------|
| `data/asl_alphabet_train/` | ~1.2 GB | `build_asl_dataset.py` |
| `data/raw_landmarks.csv` | ~20 MB | `build_asl_dataset.py` |
| `data/X.npy`, `data/y.npy` | ~6 MB | `preprocess.py` |
| `models/*.joblib` | ~48 MB | `train.py` |
| `models/hand_landmarker.task` | ~7.5 MB | `inference_web.py` (auto-téléchargé) |
| `venv/` | ~650 MB | `pip install` |
