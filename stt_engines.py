"""
Production-grade Speech-to-Text engines for Telugu.

Engine priority:
  1. Sarvam AI  (saarika:v2.5)  — best Telugu quality, purpose-built
  2. Google Cloud STT (Chirp v2) — enterprise fallback
  3. Free Google (SpeechRecognition) — dev/testing only, no API key needed

Each engine returns a list of STTResult objects.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from config import Config

logger = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────

@dataclass
class STTResult:
    """Single recognition alternative."""
    text: str
    confidence: float = 0.0
    engine: str = ""


@dataclass
class RecognitionResult:
    """Full result from an engine (may contain multiple alternatives)."""
    results: List[STTResult] = field(default_factory=list)
    error: Optional[str] = None
    engine_name: str = ""
    latency_ms: int = 0


# ─── Sarvam AI Engine ───────────────────────────────────────────

class SarvamEngine:
    """
    Sarvam AI saarika:v2.5 — purpose-built for Indian languages.

    Key advantages:
      - Accepts WebM directly (no ffmpeg conversion!)
      - Native Telugu (te-IN) support
      - <250 ms median latency
      - INR 30/hour pricing (charged per second)

    API: POST https://api.sarvam.ai/speech-to-text
    Auth: api-subscription-key header
    Body: multipart/form-data (file + model + language_code)
    """

    NAME = "sarvam"
    MAX_RETRIES = 2
    TIMEOUT = 15  # seconds

    def __init__(self):
        self.api_key = Config.SARVAM_API_KEY
        self.api_url = Config.SARVAM_API_URL
        self.model = Config.SARVAM_MODEL

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def recognize(self, audio_bytes: bytes, filename: str = "audio.webm") -> RecognitionResult:
        """Send audio to Sarvam AI and return recognized Telugu text."""
        if not self.available:
            return RecognitionResult(error="Sarvam API key not configured", engine_name=self.NAME)

        start = time.time()

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={"api-subscription-key": self.api_key},
                    files={"file": (filename, io.BytesIO(audio_bytes), "audio/webm")},
                    data={
                        "model": self.model,
                        "language_code": "te-IN",
                    },
                    timeout=self.TIMEOUT,
                )

                latency = int((time.time() - start) * 1000)

                if resp.status_code == 429:
                    logger.warning("Sarvam rate limited (attempt %d/%d)", attempt, self.MAX_RETRIES)
                    time.sleep(1 * attempt)  # simple backoff
                    continue

                if resp.status_code != 200:
                    body = resp.text[:200]
                    logger.error("Sarvam API error %d: %s", resp.status_code, body)
                    return RecognitionResult(
                        error=f"Sarvam API error ({resp.status_code})",
                        engine_name=self.NAME,
                        latency_ms=latency,
                    )

                data = resp.json()
                transcript = data.get("transcript", "").strip()

                if not transcript:
                    return RecognitionResult(
                        error="No speech detected",
                        engine_name=self.NAME,
                        latency_ms=latency,
                    )

                logger.info("Sarvam recognized: '%s' (%d ms)", transcript, latency)

                return RecognitionResult(
                    results=[STTResult(
                        text=transcript,
                        confidence=data.get("language_probability", 0.8),
                        engine=self.NAME,
                    )],
                    engine_name=self.NAME,
                    latency_ms=latency,
                )

            except requests.exceptions.Timeout:
                logger.warning("Sarvam timeout (attempt %d/%d)", attempt, self.MAX_RETRIES)
                continue

            except requests.exceptions.RequestException as e:
                logger.error("Sarvam request error: %s", e)
                return RecognitionResult(
                    error=f"Network error: {e}",
                    engine_name=self.NAME,
                    latency_ms=int((time.time() - start) * 1000),
                )

        return RecognitionResult(
            error="Sarvam API failed after retries",
            engine_name=self.NAME,
            latency_ms=int((time.time() - start) * 1000),
        )


# ─── Google Cloud STT Engine ────────────────────────────────────

class GoogleCloudEngine:
    """
    Google Cloud Speech-to-Text V2 with Chirp model.

    Requires:
      - GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON)
      - GOOGLE_CLOUD_PROJECT env var
      - Speech-to-Text API enabled in GCP console

    API: POST https://speech.googleapis.com/v2/projects/{PROJECT}/locations/us/recognizers/_:recognize
    Auth: OAuth2 Bearer token from service account
    Body: JSON with base64-encoded audio + config
    """

    NAME = "google-cloud"
    TIMEOUT = 20

    def __init__(self):
        self.project = Config.GOOGLE_CLOUD_PROJECT
        self._token: Optional[str] = None
        self._token_expiry: float = 0

    @property
    def available(self) -> bool:
        return Config.has_google()

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 token from service account credentials."""
        if self._token and time.time() < self._token_expiry:
            return self._token

        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            import os

            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
            if not creds_path:
                return None

            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(Request())
            self._token = credentials.token
            self._token_expiry = time.time() + 3500  # tokens last 1 hour
            return self._token

        except Exception as e:
            logger.error("Google auth error: %s", e)
            return None

    def recognize(self, audio_bytes: bytes, filename: str = "audio.webm") -> RecognitionResult:
        """Send audio to Google Cloud STT and return recognized Telugu text."""
        if not self.available:
            return RecognitionResult(error="Google Cloud not configured", engine_name=self.NAME)

        start = time.time()

        # Convert to WAV first (Google needs proper audio format)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            wav_buf = io.BytesIO()
            audio.export(wav_buf, format="wav")
            wav_bytes = wav_buf.getvalue()
        except Exception as e:
            logger.error("Audio conversion error: %s", e)
            return RecognitionResult(error=f"Audio conversion failed: {e}", engine_name=self.NAME)

        token = self._get_access_token()
        if not token:
            return RecognitionResult(error="Google auth failed", engine_name=self.NAME)

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        url = (
            f"https://speech.googleapis.com/v2/projects/{self.project}"
            f"/locations/us/recognizers/_:recognize"
        )

        body = {
            "config": {
                "languageCodes": ["te-IN"],
                "model": "chirp_2",
                "autoDecodingConfig": {},
            },
            "content": audio_b64,
        }

        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.TIMEOUT,
            )

            latency = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                logger.error("Google API error %d: %s", resp.status_code, resp.text[:200])
                return RecognitionResult(
                    error=f"Google API error ({resp.status_code})",
                    engine_name=self.NAME,
                    latency_ms=latency,
                )

            data = resp.json()
            results = []

            for result_block in data.get("results", []):
                for alt in result_block.get("alternatives", []):
                    text = alt.get("transcript", "").strip()
                    if text:
                        results.append(STTResult(
                            text=text,
                            confidence=alt.get("confidence", 0.7),
                            engine=self.NAME,
                        ))

            if not results:
                return RecognitionResult(error="No speech detected", engine_name=self.NAME, latency_ms=latency)

            logger.info("Google recognized: '%s' (%d ms)", results[0].text, latency)
            return RecognitionResult(results=results, engine_name=self.NAME, latency_ms=latency)

        except requests.exceptions.RequestException as e:
            logger.error("Google request error: %s", e)
            return RecognitionResult(
                error=f"Google request failed: {e}",
                engine_name=self.NAME,
                latency_ms=int((time.time() - start) * 1000),
            )


# ─── Google Cloud STT V1 (API Key) ──────────────────────────────

class GoogleAPIKeyEngine:
    """
    Google Cloud Speech-to-Text V1 using a simple API key.

    Much easier setup than V2 (no service account needed):
      1. Go to https://console.cloud.google.com/apis/credentials
      2. Create an API key
      3. Enable "Cloud Speech-to-Text API"
      4. Set GOOGLE_STT_API_KEY in .env

    Pricing: 60 min free/month, then $0.006 per 15 seconds.
    """

    NAME = "google-stt"
    TIMEOUT = 20

    def __init__(self):
        self.api_key = Config.GOOGLE_STT_API_KEY

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def recognize(self, audio_bytes: bytes, filename: str = "audio.webm") -> RecognitionResult:
        if not self.available:
            return RecognitionResult(error="Google STT API key not configured", engine_name=self.NAME)

        start = time.time()

        # Convert to WAV (Google V1 needs LINEAR16)
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent

            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            # Gentle silence trim + padding (helps single letters)
            nonsilent = detect_nonsilent(audio, min_silence_len=200, silence_thresh=-50)
            if nonsilent:
                s = max(0, nonsilent[0][0] - 200)
                e = min(len(audio), nonsilent[-1][1] + 200)
                audio = audio[s:e]

            if len(audio) < 800:
                pad = AudioSegment.silent(duration=300, frame_rate=16000)
                audio = pad + audio + pad

            wav_buf = io.BytesIO()
            audio.export(wav_buf, format="wav")
            wav_bytes = wav_buf.getvalue()
        except Exception as e:
            logger.error("Audio conversion error: %s", e)
            return RecognitionResult(error=f"Audio conversion failed: {e}", engine_name=self.NAME)

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"

        body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "te-IN",
                "alternativeLanguageCodes": ["en-IN"],
                "maxAlternatives": 5,
                "enableAutomaticPunctuation": False,
                "model": "default",
            },
            "audio": {
                "content": audio_b64,
            },
        }

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=self.TIMEOUT,
            )

            latency = int((time.time() - start) * 1000)

            if resp.status_code != 200:
                error_msg = resp.text[:300]
                logger.error("Google STT API error %d: %s", resp.status_code, error_msg)
                return RecognitionResult(
                    error=f"Google STT API error ({resp.status_code})",
                    engine_name=self.NAME,
                    latency_ms=latency,
                )

            data = resp.json()
            results = []

            for result_block in data.get("results", []):
                for alt in result_block.get("alternatives", []):
                    text = alt.get("transcript", "").strip()
                    if text:
                        results.append(STTResult(
                            text=text,
                            confidence=alt.get("confidence", 0.7),
                            engine=self.NAME,
                        ))

            if not results:
                return RecognitionResult(
                    error="No speech detected",
                    engine_name=self.NAME,
                    latency_ms=latency,
                )

            logger.info("Google STT recognized: '%s' (%d ms)", results[0].text, latency)
            return RecognitionResult(results=results, engine_name=self.NAME, latency_ms=latency)

        except requests.exceptions.RequestException as e:
            logger.error("Google STT request error: %s", e)
            return RecognitionResult(
                error=f"Google STT request failed: {e}",
                engine_name=self.NAME,
                latency_ms=int((time.time() - start) * 1000),
            )


# ─── Free Google Engine (dev/testing fallback) ──────────────────

class FreeGoogleEngine:
    """
    Uses the SpeechRecognition library's free (unofficial) Google API.
    No API key needed — works out of the box for development/testing.

    ⚠️  NOT for production:
      - Unofficial, undocumented endpoint
      - Can be rate-limited or shut down by Google
      - Lower quality than Sarvam AI

    Requires: pip install SpeechRecognition pydub
    Also requires: ffmpeg (for WebM → WAV conversion)
    """

    NAME = "free-google"

    @property
    def available(self) -> bool:
        try:
            import speech_recognition  # noqa: F401
            return True
        except ImportError:
            return False

    def recognize(self, audio_bytes: bytes, filename: str = "audio.webm") -> RecognitionResult:
        if not self.available:
            return RecognitionResult(error="SpeechRecognition not installed", engine_name=self.NAME)

        import speech_recognition as sr
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        start = time.time()

        try:
            # Convert WebM → WAV
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            # Gentle silence trim
            nonsilent = detect_nonsilent(audio, min_silence_len=200, silence_thresh=-50)
            if nonsilent:
                s = max(0, nonsilent[0][0] - 200)
                e = min(len(audio), nonsilent[-1][1] + 200)
                audio = audio[s:e]

            # Ensure min 1 second + 500ms padding
            if len(audio) < 1000:
                audio = AudioSegment.silent(duration=200) + audio + AudioSegment.silent(duration=200)
            pad = AudioSegment.silent(duration=500, frame_rate=16000)
            audio = pad + audio + pad

            wav_buf = io.BytesIO()
            audio.export(wav_buf, format="wav")
            wav_buf.seek(0)

            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 150
            recognizer.dynamic_energy_threshold = False

            with sr.AudioFile(wav_buf) as source:
                audio_data = recognizer.record(source)

            results = []

            # Pass 1: Telugu
            try:
                all_res = recognizer.recognize_google(audio_data, language="te-IN", show_all=True)
                if all_res and "alternative" in all_res:
                    for alt in all_res["alternative"]:
                        t = alt.get("transcript", "").strip()
                        if t:
                            results.append(STTResult(
                                text=t,
                                confidence=alt.get("confidence", 0.5),
                                engine=self.NAME,
                            ))
            except sr.UnknownValueError:
                pass

            # Pass 2: Indian English (catches romanised output)
            try:
                en_res = recognizer.recognize_google(audio_data, language="en-IN", show_all=True)
                if en_res and "alternative" in en_res:
                    existing = {r.text.lower() for r in results}
                    for alt in en_res["alternative"]:
                        t = alt.get("transcript", "").strip()
                        if t and t.lower() not in existing:
                            results.append(STTResult(
                                text=t,
                                confidence=alt.get("confidence", 0.4),
                                engine=f"{self.NAME}-en",
                            ))
            except sr.UnknownValueError:
                pass

            latency = int((time.time() - start) * 1000)

            if not results:
                return RecognitionResult(
                    error="Could not recognise speech. Speak louder and hold for 1–2 seconds.",
                    engine_name=self.NAME,
                    latency_ms=latency,
                )

            logger.info("FreeGoogle recognized: '%s' (%d ms)", results[0].text, latency)
            return RecognitionResult(results=results, engine_name=self.NAME, latency_ms=latency)

        except Exception as e:
            logger.error("FreeGoogle error: %s", e)
            return RecognitionResult(
                error=f"Recognition failed: {e}",
                engine_name=self.NAME,
                latency_ms=int((time.time() - start) * 1000),
            )


# ─── Multi-Engine Orchestrator ──────────────────────────────────

class STTOrchestrator:
    """
    Runs recognition through engines in priority order.
    Returns merged results from the first engine that succeeds.
    Falls back to the next engine on failure.
    """

    def __init__(self):
        self.engines = []

        # Add engines in priority order
        sarvam = SarvamEngine()
        if sarvam.available:
            self.engines.append(sarvam)
            logger.info("STT engine registered: Sarvam AI (saarika:v2.5)")

        google_key = GoogleAPIKeyEngine()
        if google_key.available:
            self.engines.append(google_key)
            logger.info("STT engine registered: Google STT (API key)")

        google = GoogleCloudEngine()
        if google.available:
            self.engines.append(google)
            logger.info("STT engine registered: Google Cloud (Chirp v2)")

        # Dev fallback — always available if SpeechRecognition is installed
        free = FreeGoogleEngine()
        if free.available:
            self.engines.append(free)
            if len(self.engines) == 1:
                logger.warning(
                    "Running with FREE Google engine only (dev mode). "
                    "For production, set SARVAM_API_KEY."
                )
            else:
                logger.info("STT engine registered: Free Google (dev fallback)")

        if not self.engines:
            logger.error(
                "No STT engines available! Install SpeechRecognition + pydub, "
                "or set SARVAM_API_KEY."
            )

    def get_engine(self, name: str):
        """Get a specific engine by name, or None."""
        for e in self.engines:
            if e.NAME == name:
                return e
        return None

    def list_engines(self) -> list[dict]:
        """Return engine info for the /api/engines endpoint."""
        return [
            {"name": e.NAME, "available": e.available}
            for e in self.engines
        ]

    def recognize(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        engine_name: str | None = None,
    ) -> RecognitionResult:
        """
        Run STT recognition.

        If engine_name is given, use ONLY that engine (no fallback).
        Otherwise try each engine in priority order, falling back on failure.
        """
        if not self.engines:
            return RecognitionResult(
                error="No STT engines configured. Please set SARVAM_API_KEY in environment."
            )

        # ── Explicit engine selection ────────────────────────────
        if engine_name:
            engine = self.get_engine(engine_name)
            if not engine:
                available = [e.NAME for e in self.engines]
                return RecognitionResult(
                    error=f"Engine '{engine_name}' not found. Available: {available}"
                )
            return engine.recognize(audio_bytes, filename)

        # ── Auto: try in priority order ──────────────────────────
        last_error = None

        for engine in self.engines:
            result = engine.recognize(audio_bytes, filename)

            if result.results:
                return result  # success — return immediately

            last_error = result
            logger.warning(
                "Engine '%s' failed: %s — trying next engine",
                engine.NAME,
                result.error,
            )

        return last_error or RecognitionResult(error="All engines failed")
