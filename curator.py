"""Selezione delle clip più virali tramite LLM, con output validato da Pydantic."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config

logger = logging.getLogger(__name__)

Platform = Literal["youtube_shorts", "tiktok", "instagram_reels"]


class CurationError(RuntimeError):
    """La curazione non ha prodotto un risultato valido."""


# --------------------------------------------------------------------------- #
# Schema Pydantic
# --------------------------------------------------------------------------- #


class ViralClip(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    clip_index: int = Field(ge=1)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    hook_reason: str = Field(min_length=1)
    viral_score: int = Field(ge=1, le=100)
    title: str = Field(min_length=1, max_length=150)
    social_caption: str = Field(min_length=1, max_length=2200)
    hashtags: list[str] = Field(min_length=1, max_length=15)
    recommended_platforms: list[Platform] = Field(min_length=1, max_length=3)

    @field_validator("viral_score", mode="before")
    @classmethod
    def _round_score(cls, value: Any) -> Any:
        if isinstance(value, float):
            return int(round(value))
        return value

    @field_validator("hashtags", mode="before")
    @classmethod
    def _normalize_hashtags(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = re.split(r"[\s,]+", value)
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            tag = "".join(item.split()).lstrip("#")
            if not tag:
                continue
            tag = f"#{tag}"
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())
            cleaned.append(tag)
        return cleaned[:15]

    @field_validator("recommended_platforms", mode="before")
    @classmethod
    def _normalize_platforms(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return value
        result: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = re.sub(r"[\s\-]+", "_", item.strip().lower())
            if item not in result:
                result.append(item)
        return result

    @model_validator(mode="after")
    def _check_range(self) -> "ViralClip":
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec deve essere maggiore di start_sec")
        return self

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


class ClipSelection(BaseModel):
    clips: list[ViralClip] = Field(min_length=1, max_length=10)


# --------------------------------------------------------------------------- #
# Chiamate LLM
# --------------------------------------------------------------------------- #


def _call_ollama(system: str, user: str) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }
    response = requests.post(config.OLLAMA_URL, json=payload, timeout=config.LLM_TIMEOUT_SEC)
    response.raise_for_status()
    text = response.json().get("response")
    if not isinstance(text, str) or not text.strip():
        raise CurationError("Risposta Ollama vuota o malformata")
    return text


def _call_openai_compatible(system: str, user: str) -> str:
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT_SEC)
    response.raise_for_status()
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise CurationError(f"Risposta API malformata: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise CurationError("Risposta API vuota")
    return content


def _call_llm(system: str, user: str) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(system, user)
    return _call_openai_compatible(system, user)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_EXAMPLE: dict[str, Any] = {
    "clips": [
        {
            "clip_index": 1,
            "start_sec": 123.4,
            "end_sec": 158.9,
            "hook_reason": "Opens with a bold, counter-intuitive claim that creates curiosity.",
            "viral_score": 87,
            "title": "Short catchy title",
            "social_caption": "Engaging caption with a call to action.",
            "hashtags": ["#topic", "#shorts", "#viral"],
            "recommended_platforms": ["youtube_shorts", "tiktok", "instagram_reels"],
        }
    ]
}


def _build_system_prompt(min_clips: int, max_clips: int) -> str:
    example = json.dumps(_EXAMPLE, ensure_ascii=False, indent=2)
    return (
        "You are an expert short-form video editor and viral content strategist "
        "(TikTok, YouTube Shorts, Instagram Reels).\n"
        "You receive a timestamped transcript and must pick the most viral moments.\n\n"
        "RULES:\n"
        f"- Return between {min_clips} and {max_clips} clips, ranked by viral potential.\n"
        f"- Each clip must last between {config.MIN_CLIP_SEC} and {config.MAX_CLIP_SEC} seconds "
        "(end_sec - start_sec).\n"
        "- start_sec and end_sec are absolute seconds taken from the transcript timestamps.\n"
        "- A clip must start at the beginning of a sentence with a strong hook in the first "
        "3 seconds, make sense on its own, and end on a complete thought.\n"
        "- Clips must NOT overlap.\n"
        "- viral_score is an integer from 1 to 100.\n"
        "- Write title, social_caption and hook_reason in the same language as the transcript.\n"
        "- hashtags: a list of strings starting with '#', no spaces inside a tag.\n"
        "- recommended_platforms: only values from "
        '["youtube_shorts", "tiktok", "instagram_reels"].\n\n'
        "OUTPUT: reply with ONE valid JSON object only. No markdown, no comments, no extra text. "
        "Use exactly this structure:\n"
        f"{example}"
    )


def _build_user_prompt(
    transcript_text: str,
    title: str,
    language: str,
    min_clips: int,
    max_clips: int,
    part: tuple[int, int] | None,
) -> str:
    part_note = ""
    if part is not None:
        part_note = (
            f"This is excerpt {part[0]} of {part[1]} of a longer video. Choose only from this "
            "excerpt. Timestamps are absolute.\n"
        )
    return (
        f"Video title: {title or 'n/a'}\n"
        f"Transcript language: {language}\n"
        f"{part_note}\n"
        "Timestamped transcript ([start-end] in seconds):\n"
        f"{transcript_text}\n\n"
        f"Return between {min_clips} and {max_clips} clips as JSON."
    )


# --------------------------------------------------------------------------- #
# Parsing e normalizzazione
# --------------------------------------------------------------------------- #


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first == -1 or last <= first:
            raise
        return json.loads(text[first : last + 1])


def _parse_selection(raw: str) -> ClipSelection:
    data = _extract_json(raw)
    if isinstance(data, list):
        data = {"clips": data}
    elif isinstance(data, dict) and "clips" not in data and "start_sec" in data:
        data = {"clips": [data]}
    return ClipSelection.model_validate(data)


def _overlaps(a: ViralClip, b: ViralClip) -> bool:
    return a.start_sec < b.end_sec and b.start_sec < a.end_sec


def _snap_clip(clip: ViralClip, starts: list[float], ends: list[float]) -> ViralClip | None:
    """Allinea i confini ai timestamp delle parole e rispetta MIN/MAX durata."""
    min_s, max_s = float(config.MIN_CLIP_SEC), float(config.MAX_CLIP_SEC)
    start = min(starts, key=lambda s: abs(s - clip.start_sec))
    valid_ends = [e for e in ends if min_s <= e - start <= max_s]
    if not valid_ends:
        return None
    end = min(valid_ends, key=lambda e: abs(e - clip.end_sec))
    return clip.model_copy(update={"start_sec": round(start, 2), "end_sec": round(end, 2)})


def _dedupe_by_score(clips: list[ViralClip]) -> list[ViralClip]:
    accepted: list[ViralClip] = []
    for clip in sorted(clips, key=lambda c: c.viral_score, reverse=True):
        if not any(_overlaps(clip, other) for other in accepted):
            accepted.append(clip)
    return accepted


def _normalize(clips: list[ViralClip], starts: list[float], ends: list[float]) -> list[ViralClip]:
    snapped: list[ViralClip] = []
    for clip in clips:
        fixed = _snap_clip(clip, starts, ends)
        if fixed is None:
            logger.debug("Clip scartata (durata non ottenibile): %s", clip.title)
            continue
        snapped.append(fixed)
    return _dedupe_by_score(snapped)


# --------------------------------------------------------------------------- #
# Trascrizione -> testo per il prompt
# --------------------------------------------------------------------------- #


def _segments_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    for w in words:
        bucket.append(w)
        token = str(w.get("word", ""))
        if len(bucket) >= 15 or token.endswith((".", "?", "!")):
            segments.append(_bucket_to_segment(bucket))
            bucket = []
    if bucket:
        segments.append(_bucket_to_segment(bucket))
    return segments


def _bucket_to_segment(bucket: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": bucket[0]["start"],
        "end": bucket[-1]["end"],
        "text": " ".join(str(w.get("word", "")) for w in bucket),
    }


def _transcript_lines(transcript: dict[str, Any], words: list[dict[str, Any]]) -> list[str]:
    segments = transcript.get("segments") or _segments_from_words(words)
    return [
        f"[{float(s['start']):.2f}-{float(s['end']):.2f}] {str(s['text']).strip()}"
        for s in segments
        if str(s.get("text", "")).strip()
    ]


def _chunk_lines(lines: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    total = len(lines)
    while start < total:
        size = 0
        end = start
        while end < total and (end == start or size + len(lines[end]) + 1 <= max_chars):
            size += len(lines[end]) + 1
            end += 1
        chunks.append("\n".join(lines[start:end]))
        if end >= total:
            break
        new_start = end
        overlap = 0
        while new_start - 1 > start and overlap + len(lines[new_start - 1]) + 1 <= overlap_chars:
            new_start -= 1
            overlap += len(lines[new_start]) + 1
        start = new_start
    return chunks


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


def _short(exc: BaseException, limit: int = 1200) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _request_clips(
    system: str,
    user: str,
    starts: list[float],
    ends: list[float],
    min_clips: int,
    max_clips: int,
) -> list[ViralClip]:
    last_error = ""
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        prompt = user
        if last_error:
            prompt = (
                f"{user}\n\nIMPORTANT: your previous answer was rejected: {last_error}\n"
                "Reply again with ONE valid JSON object following the exact structure given. "
                f"Each clip must last {config.MIN_CLIP_SEC}-{config.MAX_CLIP_SEC} seconds and "
                "use timestamps from the transcript."
            )
        try:
            raw = _call_llm(system, prompt)
            selection = _parse_selection(raw)
            clips = _normalize(selection.clips, starts, ends)
            if len(clips) < min_clips:
                raise CurationError(
                    f"only {len(clips)} usable non-overlapping clips after validation; "
                    f"at least {min_clips} required"
                )
            return clips[:max_clips]
        except (ValueError, CurationError) as exc:  # include JSONDecodeError e ValidationError
            last_error = _short(exc)
            logger.warning(
                "Tentativo %d/%d fallito: %s", attempt, config.LLM_MAX_RETRIES, last_error
            )
        except requests.RequestException as exc:
            last_error = f"network/API error: {_short(exc, 300)}"
            logger.warning(
                "Tentativo %d/%d fallito: %s", attempt, config.LLM_MAX_RETRIES, last_error
            )
            time.sleep(min(2**attempt, 10))

    raise CurationError(
        f"Output LLM non valido dopo {config.LLM_MAX_RETRIES} tentativi. Ultimo errore: {last_error}"
    )


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #


def curate_clips(transcript: dict[str, Any], video_title: str = "") -> ClipSelection:
    """Seleziona le migliori 3-5 clip virali da una trascrizione.

    Se la trascrizione supera `config.LLM_MAX_TRANSCRIPT_CHARS` viene spezzata in
    blocchi sovrapposti, curati singolarmente e poi uniti scegliendo le clip con
    punteggio più alto e senza sovrapposizioni.

    Raises:
        config.ConfigError: configurazione LLM incompleta.
        CurationError: nessuna clip valida ottenuta.
    """
    config.validate_llm_config()

    words = [
        w for w in (transcript.get("words") or []) if "start" in w and "end" in w
    ]
    if not words:
        raise CurationError("La trascrizione non contiene parole con timestamp")

    starts = [float(w["start"]) for w in words]
    ends = [float(w["end"]) for w in words]
    total_duration = max(float(transcript.get("duration") or 0.0), ends[-1])

    max_possible = int(total_duration // config.MIN_CLIP_SEC)
    if max_possible < 1:
        raise CurationError(
            f"Video troppo corto ({total_duration:.1f}s) per clip da almeno {config.MIN_CLIP_SEC}s"
        )
    min_clips = max(1, min(config.MIN_CLIPS, max_possible))
    max_clips = config.MAX_CLIPS

    language = str(transcript.get("language") or "unknown")
    lines = _transcript_lines(transcript, words)
    chunks = _chunk_lines(lines, config.LLM_MAX_TRANSCRIPT_CHARS, config.LLM_CHUNK_OVERLAP_CHARS)

    if len(chunks) == 1:
        system = _build_system_prompt(min_clips, max_clips)
        user = _build_user_prompt(chunks[0], video_title, language, min_clips, max_clips, None)
        final = _request_clips(system, user, starts, ends, min_clips, max_clips)
    else:
        per_chunk_max = 3
        system = _build_system_prompt(1, per_chunk_max)
        collected: list[ViralClip] = []
        for index, chunk in enumerate(chunks, start=1):
            logger.info("Curazione blocco %d/%d", index, len(chunks))
            user = _build_user_prompt(
                chunk, video_title, language, 1, per_chunk_max, (index, len(chunks))
            )
            try:
                collected.extend(_request_clips(system, user, starts, ends, 1, per_chunk_max))
            except CurationError as exc:
                logger.warning("Blocco %d/%d saltato: %s", index, len(chunks), exc)
        final = _dedupe_by_score(collected)[:max_clips]
        if not final:
            raise CurationError("Nessuna clip valida ottenuta da nessun blocco")

    final = sorted(final, key=lambda c: c.viral_score, reverse=True)
    if len(final) < config.MIN_CLIPS:
        logger.warning("Ottenute solo %d clip (target %d)", len(final), config.MIN_CLIPS)

    reindexed = [clip.model_copy(update={"clip_index": i}) for i, clip in enumerate(final, start=1)]
    return ClipSelection(clips=reindexed)


def curate_video(video_id: str, video_title: str = "") -> ClipSelection:
    """Carica `temp/{video_id}_transcript.json`, cura le clip e salva `temp/{video_id}_clips.json`."""
    path: Path = config.TEMP_DIR / f"{video_id}_transcript.json"
    if not path.exists():
        raise FileNotFoundError(f"Trascrizione non trovata: {path}")
    transcript = json.loads(path.read_text(encoding="utf-8"))

    selection = curate_clips(transcript, video_title)

    out_path = config.TEMP_DIR / f"{video_id}_clips.json"
    out_path.write_text(selection.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Clip salvate: %s (%d clip)", out_path.name, len(selection.clips))
    return selection
