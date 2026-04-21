"""
inference_web.py — Interface web AR avec Three.js + overlays holographiques.

Usage : python scripts/inference_web.py
Puis ouvrez http://localhost:5001
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os, sys, collections, time, threading, urllib.request
from flask import Flask, render_template, Response, jsonify
import pandas as pd
from flask_socketio import SocketIO
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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

SMOOTH_WINDOW     = 3
MIN_STABLE_FRAMES = 5

# Phrases custom : label détecté → ce que Charlotte dit
CUSTOM_PHRASES = {
    "seven": "SIX SEVEEEEENN",
}
TTS_COOLDOWN      = 1.5
CONF_THRESHOLD    = 0.60
MAX_SENTENCE      = 10

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Doigts avec traînées (indices MediaPipe)
FINGERTIP_IDS = [4, 8, 12, 16, 20]
TRAIL_LEN     = 12
TRAIL_COLORS  = {
    4:  (180, 80,  255),   # pouce   – violet
    8:  (50,  220, 255),   # index   – cyan
    12: (50,  255, 150),   # majeur  – vert
    16: (255, 180, 50),    # annulaire – or
    20: (255, 80,  180),   # auriculaire – rose
}
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config['SECRET_KEY'] = 'signlang-ar'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

_frame_lock   = threading.Lock()
_latest_frame = None
_trails       = {tip: collections.deque(maxlen=TRAIL_LEN) for tip in FINGERTIP_IDS}

_state = {
    'prediction': '', 'confidence': 0, 'speaking': False,
    'sentence': [], 'hand_detected': False, 'fps': 0,
    'stable_progress': 0.0, 'landmarks': [],
}
_state_lock = threading.Lock()


def download_model(path: str):
    if os.path.exists(path):
        return
    print("Téléchargement du modèle MediaPipe …")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"  Modèle sauvegardé : {path}")


# ── Rendu AR OpenCV ────────────────────────────────────────────────────────────

def draw_trails(frame):
    overlay = frame.copy()
    for tip_id, trail in _trails.items():
        color = TRAIL_COLORS[tip_id]
        pts   = list(trail)
        n     = len(pts)
        for i, pt in enumerate(pts):
            ratio  = (i + 1) / n
            radius = max(1, int(5 * ratio))
            c      = tuple(int(ch * ratio) for ch in color)
            cv2.circle(overlay, pt, radius, c, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)


def draw_joints_glow(frame, pts):
    """Halo lumineux autour des articulations."""
    overlay = frame.copy()
    for pt in pts:
        cv2.circle(overlay, pt, 12, (130, 70, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
    for i, pt in enumerate(pts):
        color = (255, 110, 200) if i == 0 else (200, 140, 255)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (80, 30, 110), 1, cv2.LINE_AA)


def draw_holo_label(frame, pts, label: str, confidence: float, stable_progress: float):
    """
    Label holographique flottant au-dessus du poignet.
    Taille proportionnelle à l'échelle de la main (profondeur simulée).
    """
    if not label or len(pts) < 10:
        return

    wrist = pts[0]
    mcp9  = pts[9]

    # Échelle relative à la distance poignet–MCP9
    scale_ref = np.linalg.norm(np.array(wrist) - np.array(mcp9))
    font_scale = max(0.5, min(1.4, scale_ref / 55.0))
    thickness  = max(1, int(font_scale * 2))

    # Position : au-dessus du poignet
    offset_y = int(scale_ref * 1.8)
    label_x  = wrist[0] - int(len(label) * 12 * font_scale * 0.5)
    label_y  = wrist[1] - offset_y

    # Barre de confiance (arc autour du label)
    bar_w = int(150 * font_scale * confidence / 100)
    bar_x = label_x - 5
    bar_y = label_y + 8

    # ── Fond translucide ──────────────────────────────────────────────────────
    text_w = int(len(label) * 22 * font_scale)
    pad    = int(10 * font_scale)
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (label_x - pad, label_y - int(35 * font_scale)),
                  (label_x + text_w + pad, label_y + int(12 * font_scale)),
                  (15, 5, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ── Glow ──────────────────────────────────────────────────────────────────
    glow = frame.copy()
    cv2.putText(glow, label,
                (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (130, 70, 255), thickness + 5, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.2, frame, 0.8, 0, frame)

    # ── Texte principal ───────────────────────────────────────────────────────
    cv2.putText(frame, label,
                (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (230, 210, 255), thickness, cv2.LINE_AA)

    # ── Barre de confiance ────────────────────────────────────────────────────
    if bar_w > 0:
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + int(150 * font_scale), bar_y + max(2, int(3 * font_scale))),
                      (40, 20, 80), -1)
        # Couleur : vert si conf haute, orange sinon
        bar_col = (50, 220, 150) if confidence > 80 else (50, 160, 255)
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + max(2, int(3 * font_scale))),
                      bar_col, -1)

    # ── Ligne de connexion vers poignet ───────────────────────────────────────
    cv2.line(frame,
             (label_x + text_w // 2, label_y + int(6 * font_scale)),
             wrist,
             (100, 60, 180), 1, cv2.LINE_AA)

    # ── Arc de stabilité autour du poignet ───────────────────────────────────
    if stable_progress > 0:
        angle = int(360 * stable_progress)
        cv2.ellipse(frame, wrist, (18, 18), -90, 0, angle,
                    (50, 220, 150), 2, cv2.LINE_AA)


def draw_hand(frame, landmarks, h, w, label='', confidence=0.0, stable_progress=0.0):
    """Rendu AR complet : trails + squelette + glow + label holographique."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Mise à jour des traînées
    for tip_id in FINGERTIP_IDS:
        _trails[tip_id].append(pts[tip_id])

    # Couche 1 : connexions (ligne double épaisseur pour effet néon)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (60, 30, 120), 4, cv2.LINE_AA)   # halo
        cv2.line(frame, pts[a], pts[b], (160, 90, 255), 2, cv2.LINE_AA)  # fil

    # Couche 3 : articulations avec glow
    draw_joints_glow(frame, pts)

    # Couche 4 : label holographique flottant
    if label:
        draw_holo_label(frame, pts, label, confidence, stable_progress)


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
            lms_raw    = []

            if hand_found:
                lms = results.hand_landmarks[0]

                # Landmarks pour Three.js
                lms_raw = [{'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)}
                           for lm in lms]

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

            stable_progress = min(stable_count / MIN_STABLE_FRAMES, 1.0)

            # TTS
            speaking = False
            now = time.time()
            if (stable_count >= MIN_STABLE_FRAMES
                    and smoothed is not None
                    and now - last_speak_t > TTS_COOLDOWN):
                tts.speak(CUSTOM_PHRASES.get(smoothed, smoothed))
                last_speak_t = now
                stable_count = 0
                speaking     = True
                sentence.append(smoothed)
                if len(sentence) > MAX_SENTENCE:
                    sentence.pop(0)

            # Rendu AR sur le frame
            if hand_found:
                draw_hand(frame, results.hand_landmarks[0], h, w,
                          label=smoothed or '',
                          confidence=confidence * 100,
                          stable_progress=stable_progress)

            # FPS
            t_now   = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
            t_prev  = t_now

            with _frame_lock:
                _latest_frame = frame.copy()

            payload = {
                'prediction'     : smoothed or '',
                'confidence'     : round(confidence * 100),
                'speaking'       : speaking,
                'sentence'       : sentence.copy(),
                'hand_detected'  : hand_found,
                'fps'            : round(fps_ema),
                'stable_progress': round(stable_progress, 3),
                'landmarks'      : lms_raw,
            }
            with _state_lock:
                _state.update(payload)

            socketio.emit('state', payload)

    cap.release()


def generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


_welcome_cache: bytes | None = None

WELCOME_TEXT = (
    "Welcome to SignTranslate AR. "
    "I'm Charlotte, your assistant. "
    "This platform translates sign language into text in real time. "
    "Show your hand, and your signs will come to life."
)

def _generate_welcome_audio() -> bytes:
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    if api_key:
        try:
            from elevenlabs.client import ElevenLabs
            CHARLOTTE_VOICE_ID = "XB0fDUnXU5powFXDhCwa"
            print('[TTS] Génération audio via ElevenLabs (Charlotte)…')
            client = ElevenLabs(api_key=api_key)
            audio_iter = client.text_to_speech.convert(
                voice_id=CHARLOTTE_VOICE_ID,
                text=WELCOME_TEXT,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
            )
            audio = b"".join(audio_iter)
            print(f'[TTS] OK — {len(audio)} octets')
            return audio
        except Exception as e:
            print(f'[ElevenLabs] erreur : {e} — fallback gTTS')
    import io
    from gtts import gTTS
    print('[TTS] Génération audio via gTTS (Google)…')
    tts = gTTS(text=WELCOME_TEXT, lang='en', slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


@app.route('/welcome_audio')
def welcome_audio():
    global _welcome_cache
    if _welcome_cache is None:
        try:
            _welcome_cache = _generate_welcome_audio()
        except Exception as e:
            print(f'[ElevenLabs] ERREUR : {e}')
            return Response(status=204)
    return Response(_welcome_cache, mimetype='audio/mpeg',
                    headers={'Cache-Control': 'public, max-age=86400'})


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sign_poses')
def sign_poses():
    """
    Retourne les poses moyennes (21 landmarks × 3 axes) par signe,
    calculées sur les données collectées.
    Format : { "hello": [[x0,y0,z0], ..., [x20,y20,z20]], ... }
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_landmarks.csv')
    if not os.path.exists(csv_path):
        return jsonify({})
    try:
        df       = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c != 'label']
        poses    = {}
        for sign in df['label'].unique():
            avg = df[df['label'] == sign][feat_cols].values.mean(axis=0)
            poses[str(sign)] = avg.reshape(21, 3).tolist()
        return jsonify(poses)
    except Exception as e:
        print(f"sign_poses error: {e}")
        return jsonify({})


@socketio.on('clear_sentence')
def handle_clear():
    with _state_lock:
        _state['sentence'] = []


if __name__ == '__main__':
    threading.Thread(target=camera_loop, daemon=True).start()
    print("\n" + "─" * 50)
    print("  Interface AR : http://localhost:5001")
    print("  Ouvrez ce lien dans votre navigateur.")
    print("─" * 50 + "\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False,
                 allow_unsafe_werkzeug=True)
