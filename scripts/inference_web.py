"""
inference_web.py — Interface web temps réel avec animations.

Usage : python scripts/inference_web.py
Puis ouvrez http://localhost:5000 dans votre navigateur.
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os, sys, collections, time, threading, urllib.request
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hand_utils import extract_landmarks, normalize_landmarks
from utils.tts_utils   import TTSEngine

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')
MODEL_PATH    = os.path.join(MODELS_DIR, 'rf_model.joblib')
ENC_PATH      = os.path.join(MODELS_DIR, 'label_encoder.joblib')
MP_MODEL      = os.path.join(MODELS_DIR, 'hand_landmarker.task')
CAM_INDEX     = 0

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

SMOOTH_WINDOW     = 7
MIN_STABLE_FRAMES = 15
TTS_COOLDOWN      = 2.5
CONF_THRESHOLD    = 0.65
MAX_SENTENCE      = 10

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config['SECRET_KEY'] = 'signlang-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

_frame_lock  = threading.Lock()
_latest_frame = None

_state = {
    'prediction': '', 'confidence': 0, 'speaking': False,
    'sentence': [], 'hand_detected': False, 'fps': 0,
    'stable_progress': 0.0,
}
_state_lock = threading.Lock()


def download_model(path: str):
    if os.path.exists(path):
        return
    print("Téléchargement du modèle MediaPipe …")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  Modèle sauvegardé : {path}")


def draw_hand(frame, landmarks, h, w):
    """Dessine le squelette avec un style néon violet."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 90, 255), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = (255, 120, 200) if i == 0 else (210, 160, 255)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (80, 40, 120), 1, cv2.LINE_AA)


def camera_loop():
    global _latest_frame

    download_model(MP_MODEL)

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : {MODEL_PATH} introuvable. Lancez train.py d'abord.")
        return

    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENC_PATH)
    tts = TTSEngine()

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    pred_buffer  = collections.deque(maxlen=SMOOTH_WINDOW)
    stable_count = 0
    last_speak_t = 0.0
    fps_ema      = 0.0
    t_prev       = time.perf_counter()
    t_start      = time.time()
    sentence     = []

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms   = int((time.time() - t_start) * 1000)
            results = landmarker.detect_for_video(mp_image, ts_ms)

            prediction = None
            confidence = 0.0
            hand_found = len(results.hand_landmarks) > 0

            if hand_found:
                lms    = results.hand_landmarks[0]
                draw_hand(frame, lms, h, w)
                raw    = extract_landmarks(lms)
                normed = normalize_landmarks(raw).reshape(1, -1)
                proba  = clf.predict_proba(normed)[0]
                top_i  = int(np.argmax(proba))
                confidence = float(proba[top_i])
                if confidence >= CONF_THRESHOLD:
                    prediction = le.classes_[top_i]

            # Lissage
            pred_buffer.append(prediction)
            smoothed = None
            if len(pred_buffer) == SMOOTH_WINDOW:
                best, cnt = collections.Counter(pred_buffer).most_common(1)[0]
                if best is not None and cnt >= (SMOOTH_WINDOW // 2 + 1):
                    smoothed = best

            if smoothed is not None and smoothed == prediction:
                stable_count += 1
            else:
                stable_count = 0

            # TTS + mise à jour phrase
            speaking = False
            now = time.time()
            if (stable_count >= MIN_STABLE_FRAMES
                    and smoothed is not None
                    and now - last_speak_t > TTS_COOLDOWN):
                tts.speak(smoothed)
                last_speak_t = now
                stable_count = 0
                speaking     = True
                sentence.append(smoothed)
                if len(sentence) > MAX_SENTENCE:
                    sentence.pop(0)

            # FPS EMA
            t_now   = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
            t_prev  = t_now

            # Frame partagée
            with _frame_lock:
                _latest_frame = frame.copy()

            # Événement SocketIO
            payload = {
                'prediction'     : smoothed or '',
                'confidence'     : round(confidence * 100),
                'speaking'       : speaking,
                'sentence'       : sentence.copy(),
                'hand_detected'  : hand_found,
                'fps'            : round(fps_ema),
                'stable_progress': round(min(stable_count / MIN_STABLE_FRAMES, 1.0), 3),
            }
            with _state_lock:
                _state.update(payload)

            socketio.emit('state', payload)

    cap.release()


def generate_mjpeg():
    """Générateur MJPEG (~30 FPS) pour le stream vidéo."""
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('clear_sentence')
def handle_clear():
    with _state_lock:
        _state['sentence'] = []


if __name__ == '__main__':
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

    print("\n" + "─" * 50)
    print("  Interface web : http://localhost:5001")
    print("  Ouvrez ce lien dans votre navigateur.")
    print("─" * 50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False,
                 allow_unsafe_werkzeug=True)
