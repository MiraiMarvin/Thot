"""
Moteur TTS non-bloquant.
Priorité : ElevenLabs (Adam) → pyttsx3 (offline) → gTTS fallback.
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


ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"


class TTSEngine:
    def __init__(self):
        self._lock   = threading.Lock()
        self._busy   = False
        self._engine = None
        self._el_key = _load_api_key()
        if not self._el_key:
            self._init_pyttsx3()

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
                elif self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    self._gtts_fallback(text)
            except Exception as e:
                print(f"[TTS] Erreur ElevenLabs : {e} — fallback pyttsx3")
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
            voice_id=ADAM_VOICE_ID,
            text=text,
            model_id="eleven_monolingual_v1",
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

    def _gtts_fallback(self, text: str):
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
