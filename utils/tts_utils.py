"""
Moteur TTS non-bloquant.
Priorité : ElevenLabs (Charlotte) → gTTS (Google) → pyttsx3 (offline).
"""
import threading
import platform
import os
import tempfile


def _load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    except ImportError:
        pass
    return os.environ.get('ELEVENLABS_API_KEY')


CHARLOTTE_VOICE_ID = "XB0fDUnXU5powFXDhCwa"


class TTSEngine:
    def __init__(self):
        self._lock   = threading.Lock()
        self._busy   = False
        self._engine = None          # pyttsx3, initialisé seulement si gTTS échoue
        self._el_key = _load_api_key()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 150)
            self._engine.setProperty('volume', 1.0)
        except Exception:
            self._engine = None

    def speak(self, text: str):
        if self._busy:
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        with self._lock:
            self._busy = True
            try:
                if self._el_key:
                    self._elevenlabs(text)
                else:
                    self._gtts(text)
            except Exception as e:
                print(f"[TTS] gTTS erreur : {e} — fallback pyttsx3")
                try:
                    if self._engine is None:
                        self._init_pyttsx3()
                    if self._engine:
                        self._engine.say(text)
                        self._engine.runAndWait()
                except Exception:
                    pass
            finally:
                self._busy = False

    def _elevenlabs(self, text: str):
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self._el_key)
        audio_iter = client.text_to_speech.convert(
            voice_id=CHARLOTTE_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_iter)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name

        sys = platform.system()
        if sys == 'Darwin':
            os.system(f'afplay "{tmp}"')
        elif sys == 'Windows':
            os.system(f'start /wait "" "{tmp}"')
        else:
            os.system(f'mpg123 "{tmp}"')

        os.unlink(tmp)

    def _gtts(self, text: str):
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tts.save(f.name)
            tmp = f.name

        sys = platform.system()
        if sys == 'Darwin':
            os.system(f'afplay "{tmp}"')
        elif sys == 'Windows':
            os.system(f'start /wait "" "{tmp}"')
        else:
            os.system(f'mpg123 "{tmp}"')

        os.unlink(tmp)
