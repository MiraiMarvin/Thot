"""
Moteur TTS non-bloquant.

Priorité : pyttsx3 (offline, faible latence, Windows + Mac).
Fallback  : gTTS + lecture système (nécessite connexion internet).
"""
import threading
import platform


class TTSEngine:
    def __init__(self):
        self._lock  = threading.Lock()
        self._busy  = False
        self._engine = None
        self._init_pyttsx3()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 150)   # mots/minute
            self._engine.setProperty('volume', 1.0)
        except Exception:
            self._engine = None  # pyttsx3 non disponible → fallback gTTS

    def speak(self, text: str):
        """Lance la synthèse dans un thread daemon (non-bloquant)."""
        if self._busy:
            return  # ignore si une synthèse est déjà en cours
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        with self._lock:
            self._busy = True
            try:
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    self._gtts_fallback(text)
            finally:
                self._busy = False

    def _gtts_fallback(self, text: str):
        """Synthèse gTTS : génère un MP3 temporaire et le lit via la commande système."""
        import os
        import tempfile
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
            os.system(f'mpg123 "{tmp}"')   # Linux : sudo apt install mpg123

        os.unlink(tmp)
