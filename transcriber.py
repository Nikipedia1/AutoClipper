"""Trascrizione con faster-whisper e timestamp per singola parola."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Errore durante la trascrizione."""


def transcript_path(video_id: str) -> Path:
    return config.TEMP_DIR / f"{video_id}_transcript.json"


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    logger.info(
        "Caricamento modello Whisper '%s' (device=%s, compute_type=%s). "
        "Al primo avvio il modello viene scaricato.",
        config.MODEL_SIZE,
        config.DEVICE,
        config.COMPUTE_TYPE,
    )
    return WhisperModel(
        config.MODEL_SIZE,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def load_transcript(video_id: str) -> dict[str, Any]:
    """Carica una trascrizione già salvata in temp/."""
    path = transcript_path(video_id)
    if not path.exists():
        raise FileNotFoundError(f"Trascrizione non trovata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def transcribe_video(
    video_path: str | Path,
    video_id: str,
    *,
    force: bool = False,
    language: str | None = None,
) -> dict[str, Any]:
    """Trascrive il video e salva `temp/{video_id}_transcript.json`.

    Struttura del risultato:
        {
          "video_id": str, "language": str, "language_probability": float,
          "duration": float,
          "words":    [{"word": str, "start": float, "end": float, "probability": float}, ...],
          "segments": [{"id": int, "start": float, "end": float, "text": str}, ...]
        }

    I tempi sono in secondi arrotondati al centesimo.

    Args:
        video_path: file video/audio da trascrivere.
        video_id: identificativo usato per il nome del file di output.
        force: se True ritrascrive anche se esiste già la trascrizione.
        language: codice lingua (None = config.WHISPER_LANGUAGE oppure auto-detect).
    """
    out_path = transcript_path(video_id)
    if out_path.exists() and not force:
        logger.info("Trascrizione già presente, riuso: %s", out_path.name)
        return load_transcript(video_id)

    media = Path(video_path)
    if not media.exists():
        raise FileNotFoundError(f"File media non trovato: {media}")

    model = _get_model()
    segments_iter, info = model.transcribe(
        str(media),
        word_timestamps=True,
        language=language or config.WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    logger.info(
        "Lingua rilevata: %s (p=%.2f), durata audio %.1fs",
        info.language,
        info.language_probability,
        info.duration,
    )

    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    next_log_at = 60.0

    for segment in segments_iter:
        text = segment.text.strip()
        if text:
            segments.append(
                {
                    "id": int(segment.id),
                    "start": round(float(segment.start), 2),
                    "end": round(float(segment.end), 2),
                    "text": text,
                }
            )
        for w in segment.words or []:
            token = w.word.strip()
            if not token:
                continue
            words.append(
                {
                    "word": token,
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                    "probability": round(float(w.probability), 3),
                }
            )
        if segment.end >= next_log_at:
            logger.info("Trascritti %.0fs / %.0fs", segment.end, info.duration)
            next_log_at = segment.end + 60.0

    if not words:
        raise TranscriptionError(
            "Nessuna parola rilevata: il video potrebbe non contenere parlato."
        )

    result: dict[str, Any] = {
        "video_id": video_id,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration": round(float(info.duration), 2),
        "words": words,
        "segments": segments,
    }
    _write_json_atomic(out_path, result)
    logger.info("Trascrizione salvata: %s (%d parole)", out_path.name, len(words))
    return result
