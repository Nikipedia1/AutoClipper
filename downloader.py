"""Download di video YouTube (MP4 1080p H.264 + AAC) con controllo anti-duplicati."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import yt_dlp
from yt_dlp.utils import DownloadError

import config

logger = logging.getLogger(__name__)

# Preferisce H.264 (avc1) + AAC (mp4a) fino a 1080p, con fallback progressivi.
VIDEO_FORMAT: str = (
    "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]"
    "/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
    "/best[height<=1080][ext=mp4]"
    "/best[height<=1080]"
)


class VideoInfo(TypedDict):
    video_id: str
    title: str
    video_path: str
    duration: float


class DownloaderError(Exception):
    """Errore generico del downloader."""


class AlreadyProcessedError(DownloaderError):
    """Il video è già presente in processed_logs/history.json."""

    def __init__(self, video_id: str, title: str = "") -> None:
        self.video_id = video_id
        self.title = title
        suffix = f" ({title})" if title else ""
        super().__init__(f"Video già processato: {video_id}{suffix}")


class DownloadFailedError(DownloaderError):
    """Il download non è andato a buon fine."""


# --------------------------------------------------------------------------- #
# Storico anti-duplicati
# --------------------------------------------------------------------------- #


def _load_history() -> dict[str, dict[str, Any]]:
    path = config.HISTORY_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        corrupted = path.with_name(path.name + ".corrupt")
        logger.warning("history.json illeggibile (%s): spostato in %s", exc, corrupted.name)
        try:
            os.replace(path, corrupted)
        except OSError:
            pass
        return {}

    if isinstance(data, list):  # formato legacy: lista di id
        return {str(video_id): {} for video_id in data}
    if isinstance(data, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in data.items()}
    return {}


def _save_history(history: dict[str, dict[str, Any]]) -> None:
    path = config.HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(history, tmp, ensure_ascii=False, indent=2)
        tmp_name = tmp.name
    os.replace(tmp_name, path)  # scrittura atomica (anche su Windows)


def is_processed(video_id: str) -> bool:
    """True se il video_id è già nello storico."""
    return video_id in _load_history()


def mark_as_processed(video_id: str, title: str = "", url: str = "") -> None:
    """Registra il video come processato.

    Da chiamare dall'orchestratore SOLO a pipeline completata con successo,
    così un'esecuzione fallita a metà può essere rilanciata.
    """
    history = _load_history()
    history[video_id] = {
        "title": title,
        "url": url,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_history(history)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise DownloaderError(
            "FFmpeg non trovato nel PATH (necessario per unire video e audio). "
            "Su Windows: `winget install Gyan.FFmpeg`, poi riapri il terminale."
        )


def _build_ydl_opts() -> dict[str, Any]:
    return {
        "format": VIDEO_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(config.INPUT_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "windowsfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "overwrites": False,
    }


def _resolve_video_path(info: dict[str, Any], video_id: str) -> Path:
    for item in info.get("requested_downloads") or []:
        filepath = item.get("filepath")
        if filepath and Path(filepath).exists():
            return Path(filepath)

    expected = config.INPUT_DIR / f"{video_id}.mp4"
    if expected.exists():
        return expected

    ignored = {".part", ".ytdl", ".json", ".tmp"}
    for candidate in sorted(config.INPUT_DIR.glob(f"{video_id}.*")):
        if candidate.suffix.lower() not in ignored:
            return candidate

    raise DownloadFailedError(f"File scaricato non trovato in {config.INPUT_DIR} per id {video_id}")


def download_video(url: str, *, force: bool = False) -> VideoInfo:
    """Scarica un video YouTube in input/ e restituisce i metadati.

    Args:
        url: URL del video.
        force: se True ignora lo storico anti-duplicati.

    Returns:
        dict con `video_id`, `title`, `video_path` (str) e `duration` (secondi).

    Raises:
        AlreadyProcessedError: il video_id è già in processed_logs/history.json.
        DownloadFailedError: download fallito o URL non supportato (playlist, live).
        DownloaderError: FFmpeg mancante.
        ValueError: URL vuoto.
    """
    url = url.strip()
    if not url:
        raise ValueError("URL vuoto")

    _ensure_ffmpeg()
    opts = _build_ydl_opts()

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            meta = ydl.extract_info(url, download=False)
            if not meta:
                raise DownloadFailedError("yt-dlp non ha restituito metadati")
            if meta.get("_type") == "playlist":
                raise DownloadFailedError(
                    "URL di playlist/canale non supportato: fornisci l'URL di un singolo video"
                )
            if meta.get("is_live"):
                raise DownloadFailedError("Le dirette in corso non sono supportate")

            video_id = str(meta["id"])
            title = str(meta.get("title") or video_id)

            if not force and is_processed(video_id):
                raise AlreadyProcessedError(video_id, title)

            existing = config.INPUT_DIR / f"{video_id}.mp4"
            if existing.exists():
                logger.info("File già presente, salto il download: %s", existing.name)
                info = meta
                video_path = existing
            else:
                logger.info("Download di '%s' (%s)...", title, video_id)
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise DownloadFailedError("Download fallito: nessun risultato")
                video_path = _resolve_video_path(info, video_id)
    except DownloadError as exc:
        raise DownloadFailedError(f"yt-dlp: {exc}") from exc

    duration = float(info.get("duration") or meta.get("duration") or 0.0)
    return VideoInfo(
        video_id=video_id,
        title=title,
        video_path=str(video_path),
        duration=duration,
    )
