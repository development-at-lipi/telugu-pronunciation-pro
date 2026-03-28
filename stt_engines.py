"""
Production-grade Speech-to-Text engines for Telugu.

Engine priority:
  1. Sarvam AI  (saarika:v2.5)  — best Telugu quality, purpose-built
  2. Google Cloud STT (Chirp v2) — enterprise fallback

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

        google = GoogleCloudEngine()
        if google.available:
            self.engines.append(google)
            logger.info("STT engine registered: Google Cloud (Chirp v2)")

        if not self.engines:
            logger.warning(
                "No STT engines configured! Set SARVAM_API_KEY or "
                "GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT"
            )

    def recognize(self, audio_bytes: bytes, filename: str = "audio.webm") -> RecognitionResult:
        """
        Try each engine in priority order.
        Returns the first successful result, or the last error.
        """
        if not self.engines:
            return RecognitionResult(
                error="No STT engines configured. Please set SARVAM_API_KEY in environment."
            )

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
