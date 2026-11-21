"""Configurazione centrale di AutoClipper.

Tutte le impostazioni sono costanti di modulo. I valori LLM possono essere
sovrascritti tramite variabili d'ambiente:

    LLM_PROVIDER   ollama | openai | gemini      (default: ollama)
    LLM_API_KEY    chiave API (obbligatoria per openai/gemini)
    LLM_MODEL      nome modello (default dipendente dal provider)
    LLM_BASE_URL   base URL compatibile OpenAI (opzionale)
    OLLAMA_URL     endpoint Ollama (default: http://localhost:11434/api/generate)
    WHISPER_LANGUAGE  codice lingua (es. "it"); vuoto = auto-detect
    WHISPER_DEVICE    cuda | cpu (default: auto)
    WHISPER_COMPUTE_TYPE  float16 | int8 | ...
    SMART_CROP     1/0 (default: 1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Cartelle
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
INPUT_DIR: Final[Path] = BASE_DIR / "input"
TEMP_DIR: Final[Path] = BASE_DIR / "temp"
OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
PROCESSED_LOGS_DIR: Final[Path] = BASE_DIR / "processed_logs"
HISTORY_FILE: Final[Path] = PROCESSED_LOGS_DIR / "history.json"


def ensure_directories() -> None:
    """Crea le cartelle di lavoro se non esistono."""
    for directory in (INPUT_DIR, TEMP_DIR, OUTPUT_DIR, PROCESSED_LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

# --------------------------------------------------------------------------- #
# Helper per variabili d'ambiente
# --------------------------------------------------------------------------- #


class ConfigError(RuntimeError):
    """Configurazione non valida o incompleta."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Variabile d'ambiente {name} non è un intero: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Variabile d'ambiente {name} non è un numero: {raw!r}") from exc


# --------------------------------------------------------------------------- #
# Whisper
# --------------------------------------------------------------------------- #
MODEL_SIZE: Final[str] = os.getenv("WHISPER_MODEL", "medium").strip() or "medium"

# Auto-detect CUDA if available, otherwise CPU
_raw_device = (os.getenv("WHISPER_DEVICE") or "").strip().lower()
if _raw_device in ("cuda", "cpu"):
    DEVICE: Final[str] = _raw_device
else:
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"

# float16 on GPU is much faster; int8 on CPU saves RAM
_raw_compute = (os.getenv("WHISPER_COMPUTE_TYPE") or "").strip().lower()
if _raw_compute:
    COMPUTE_TYPE: Final[str] = _raw_compute
else:
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

WHISPER_LANGUAGE: Final[str | None] = (os.getenv("WHISPER_LANGUAGE") or "").strip() or None

# --------------------------------------------------------------------------- #
# LLM (dual-mode: Ollama locale oppure API compatibile OpenAI / Gemini)
# --------------------------------------------------------------------------- #
SUPPORTED_LLM_PROVIDERS: Final[tuple[str, ...]] = ("ollama", "openai", "gemini")

LLM_PROVIDER: Final[str] = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    raise ConfigError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} non supportato. "
        f"Valori ammessi: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
    )

_DEFAULT_MODELS: Final[dict[str, str]] = {
    "ollama": "qwen2.5:7b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}
_DEFAULT_BASE_URLS: Final[dict[str, str]] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

LLM_API_KEY: Final[str] = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL: Final[str] = (os.getenv("LLM_MODEL") or "").strip() or _DEFAULT_MODELS[LLM_PROVIDER]
LLM_BASE_URL: Final[str] = (
    (os.getenv("LLM_BASE_URL") or "").strip() or _DEFAULT_BASE_URLS.get(LLM_PROVIDER, "")
)

OLLAMA_URL: Final[str] = (
    os.getenv("OLLAMA_URL") or "http://localhost:11434/api/generate"
).strip()
OLLAMA_MODEL: Final[str] = "qwen2.5:7b"
OLLAMA_NUM_CTX: Final[int] = _env_int("OLLAMA_NUM_CTX", 16384)

LLM_TEMPERATURE: Final[float] = _env_float("LLM_TEMPERATURE", 0.4)
LLM_TIMEOUT_SEC: Final[int] = _env_int("LLM_TIMEOUT_SEC", 600)
LLM_MAX_RETRIES: Final[int] = 3

LLM_MAX_TRANSCRIPT_CHARS: Final[int] = _env_int("LLM_MAX_TRANSCRIPT_CHARS", 24000)
LLM_CHUNK_OVERLAP_CHARS: Final[int] = _env_int("LLM_CHUNK_OVERLAP_CHARS", 2000)

# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
TARGET_WIDTH: Final[int] = 1080
TARGET_HEIGHT: Final[int] = 1920
MIN_CLIP_SEC: Final[int] = 28
MAX_CLIP_SEC: Final[int] = 58

MIN_CLIPS: Final[int] = 3
MAX_CLIPS: Final[int] = 5

# Smart crop (face tracking) – attivo di default
SMART_CROP: Final[bool] = (os.getenv("SMART_CROP", "1").strip().lower() in ("1", "true", "yes", "on"))


def validate_llm_config() -> None:
    """Verifica che la configurazione LLM sia utilizzabile."""
    if LLM_PROVIDER != "ollama" and not LLM_API_KEY:
        raise ConfigError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} richiede la variabile d'ambiente LLM_API_KEY."
        )
    if LLM_PROVIDER != "ollama" and not LLM_BASE_URL:
        raise ConfigError("LLM_BASE_URL non impostato per il provider selezionato.")
