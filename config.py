"""
Application configuration — loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Sarvam AI ────────────────────────────────────────────────
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_API_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_MODEL: str = "saarika:v2.5"

    # ── Google Cloud STT (V2 — service account) ───────────────────
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    # GOOGLE_APPLICATION_CREDENTIALS is read automatically by google libs

    # ── Google Cloud STT (V1 — simple API key) ──────────────────
    GOOGLE_STT_API_KEY: str = os.getenv("GOOGLE_STT_API_KEY", "")

    # ── App ──────────────────────────────────────────────────────
    PORT: int = int(os.getenv("FLASK_PORT", "5060"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_AUDIO_SECONDS: int = 30  # Sarvam REST limit

    @classmethod
    def has_sarvam(cls) -> bool:
        return bool(cls.SARVAM_API_KEY)

    @classmethod
    def has_google(cls) -> bool:
        """Google Cloud V2 (service account)."""
        return bool(cls.GOOGLE_CLOUD_PROJECT) and bool(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        )

    @classmethod
    def has_google_apikey(cls) -> bool:
        """Google Cloud V1 (simple API key)."""
        return bool(cls.GOOGLE_STT_API_KEY)
