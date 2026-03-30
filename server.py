"""
Telugu Pronunciation Checker — Production API Server.

STT engines (in priority order):
  1. Sarvam AI  (saarika:v2.5)  — best Telugu quality
  2. Google Cloud STT (Chirp v2) — enterprise fallback

Endpoints:
  GET  /                     → frontend
  GET  /api/letters           → letter/word lists
  GET  /api/health            → health check + engine status
  POST /api/verify            → pronunciation verification
"""
from __future__ import annotations

import logging
import os
import sys

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from config import Config
from stt_engines import STTOrchestrator
from telugu_matcher import TELUGU_LETTERS, compare

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("server")

# ── App ──────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app)

orchestrator = STTOrchestrator()


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — shows which engines are configured."""
    return jsonify({
        "status": "ok",
        "engines": orchestrator.list_engines(),
        "engine_count": len(orchestrator.engines),
    })


@app.route("/api/engines", methods=["GET"])
def list_engines():
    """List available STT engines so the frontend can offer a selector."""
    return jsonify({
        "engines": orchestrator.list_engines(),
        "default": orchestrator.engines[0].NAME if orchestrator.engines else None,
    })


@app.route("/api/letters", methods=["GET"])
def get_letters():
    category = request.args.get("category", "vowels")
    letters = TELUGU_LETTERS.get(category, TELUGU_LETTERS["vowels"])
    return jsonify({
        "category": category,
        "letters": letters,
        "categories": list(TELUGU_LETTERS.keys()),
    })


@app.route("/api/verify", methods=["POST"])
def verify_pronunciation():
    """
    Verify Telugu pronunciation.

    Expects multipart/form-data:
      - audio:    audio file (WebM from browser)
      - expected: Telugu text to verify against
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    expected_text = request.form.get("expected", "").strip()
    if not expected_text:
        return jsonify({"error": "No expected text provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if len(audio_bytes) < 100:
        return jsonify({"error": "Audio file too small — recording failed"}), 400

    # ── Run STT (optionally force a specific engine) ──────────

    preferred_engine = request.form.get("engine", "").strip() or None
    recognition = orchestrator.recognize(
        audio_bytes,
        audio_file.filename or "audio.webm",
        engine_name=preferred_engine,
    )

    if not recognition.results:
        return jsonify({
            "pass": False,
            "error": recognition.error or "Could not recognise speech. Speak louder and hold for 1–2 seconds.",
            "expected": expected_text,
            "recognized": None,
            "score": 0,
            "engine": recognition.engine_name,
            "latency_ms": recognition.latency_ms,
            "details": [],
        })

    # ── Compare each result with expected text ───────────────

    best_result = None
    best_score = -1

    for stt_result in recognition.results:
        match = compare(expected_text, stt_result.text)
        if match.score > best_score:
            best_score = match.score
            best_result = {
                "recognized_text": stt_result.text,
                "engine_confidence": stt_result.confidence,
                "match": match,
            }

    passed = best_result["match"].match if best_result else False

    return jsonify({
        "pass": passed,
        "expected": expected_text,
        "recognized": best_result["recognized_text"] if best_result else None,
        "score": best_result["match"].score if best_result else 0,
        "reason": best_result["match"].reason if best_result else "No match",
        "engine": recognition.engine_name,
        "latency_ms": recognition.latency_ms,
        "details": [
            {
                "text": r.text,
                "confidence": r.confidence,
                "engine": r.engine,
                "match": {
                    "match": compare(expected_text, r.text).match,
                    "score": compare(expected_text, r.text).score,
                    "reason": compare(expected_text, r.text).reason,
                },
            }
            for r in recognition.results
        ],
    })


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = Config.PORT
    logger.info("Starting Telugu Pronunciation Checker (Production) on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
