"""Rendering di clip verticali 9:16 con FFmpeg (taglio, crop, sottotitoli ASS, loudnorm)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import config

logger = logging.getLogger(__name__)

LOUDNORM_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"
DEFAULT_TIMEOUT_SEC = 3600


class RenderError(RuntimeError):
    """Errore durante il rendering con FFmpeg."""


# --------------------------------------------------------------------------- #
# Escape per il parser dei filtri FFmpeg
# --------------------------------------------------------------------------- #


def escape_filter_value(value: str) -> str:
    """Prepara un valore (es. un percorso) da usare come opzione di un filtro FFmpeg.

    FFmpeg applica due livelli di parsing al valore di un'opzione di filtro:
      1. opzioni del filtro: `\\`, `:` e `'` sono speciali  -> vanno preceduti da `\\`
      2. filtergraph: `, ; [ ] ' \\` sono speciali          -> si racchiude tra apici singoli
         (un apice interno si scrive chiudendo l'apice, `\\'`, riaprendolo)

    Esempio Windows: `C:/Users/me/clip.ass`  ->  `'C\\:/Users/me/clip.ass'`
    Poiché il comando è passato come lista ad argv (nessuna shell), non serve un terzo livello.
    """
    level1 = value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return "'" + level1.replace("'", "'\\''") + "'"


def _build_ass_filter(ass_path: Path) -> tuple[str, Path | None]:
    """Costruisce il filtro `ass=` e, se serve, una copia temporanea con nome ASCII.

    Su Windows libass può non aprire file con caratteri non-ASCII nel percorso:
    in quel caso si usa una copia con nome ASCII nella stessa cartella e FFmpeg
    viene eseguito con quella cartella come working directory.

    Returns:
        (filtro, copia_temporanea_o_None)
    """
    absolute = ass_path.resolve()
    if str(absolute).isascii():
        return f"ass=filename={escape_filter_value(absolute.as_posix())}", None

    staged = absolute.parent / f"_burn_{uuid.uuid4().hex[:8]}.ass"
    shutil.copyfile(absolute, staged)
    return f"ass=filename={escape_filter_value(staged.name)}", staged


# --------------------------------------------------------------------------- #
# Filtro video
# --------------------------------------------------------------------------- #


def _probe_dimensions(video_path: Path) -> tuple[int, int] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            return None
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, ValueError, TypeError):
        return None


def _build_video_filter(video_path: Path) -> str:
    width_t, height_t = config.TARGET_WIDTH, config.TARGET_HEIGHT
    dims = _probe_dimensions(video_path)
    if dims is not None:
        src_w, src_h = dims
        # Sorgente almeno largo quanto il formato target: crop verticale centrale
        # (larghezza pari, equivalente a crop=ih*9/16:ih) e poi scale.
        if src_w * height_t >= src_h * width_t:
            crop = f"crop=trunc(ih*{width_t}/{height_t}/2)*2:ih"
            return f"{crop},scale={width_t}:{height_t}:flags=lanczos,setsar=1"
    # Sorgente più stretta del 9:16 (o dimensioni ignote): riempi il frame e ritaglia.
    return (
        f"scale={width_t}:{height_t}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width_t}:{height_t},setsar=1"
    )


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #


def render_vertical_clip(
    video_path: Path,
    ass_path: Path,
    start_sec: float,
    end_sec: float,
    output_mp4_path: Path,
    *,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> Path:
    """Taglia [start_sec, end_sec], porta il video in 1080x1920, brucia i sottotitoli e normalizza l'audio.

    Il taglio usa `-ss` (inizio) e `-t` (durata = end - start) come opzioni di input:
    è equivalente a `-ss/-to` ma con semantica identica in tutte le versioni di FFmpeg.
    I timestamp dei sottotitoli devono essere relativi all'inizio della clip
    (come prodotti da `subtitle_gen.generate_ass_subtitles`).

    Raises:
        FileNotFoundError: video o file ASS mancanti.
        ValueError: intervallo non valido.
        RenderError: FFmpeg assente, fallito o in timeout.
    """
    video_path = Path(video_path)
    ass_path = Path(ass_path)
    output_mp4_path = Path(output_mp4_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video non trovato: {video_path}")
    if not ass_path.is_file():
        raise FileNotFoundError(f"File ASS non trovato: {ass_path}")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"Intervallo non valido: start={start_sec}, end={end_sec}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RenderError(
            "FFmpeg non trovato nel PATH. Su Windows: `winget install Gyan.FFmpeg` "
            "e riapri il terminale."
        )

    duration = end_sec - start_sec
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_mp4_path.with_name(f"{output_mp4_path.stem}.partial.mp4")

    ass_filter, staged = _build_ass_filter(ass_path)
    video_filter = ",".join([_build_video_filter(video_path), ass_filter])

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path.resolve()),
        "-vf",
        video_filter,
        "-af",
        LOUDNORM_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(partial.resolve()),
    ]

    logger.info(
        "Rendering %s [%.2fs -> %.2fs] -> %s", video_path.name, start_sec, end_sec, output_mp4_path.name
    )
    logger.debug("Filtro video: %s", video_filter)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
            stdin=subprocess.DEVNULL,
            cwd=str(staged.parent) if staged is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        _safe_unlink(partial)
        raise RenderError(f"FFmpeg ha superato il timeout di {timeout_sec}s") from exc
    except OSError as exc:
        _safe_unlink(partial)
        raise RenderError(f"Impossibile avviare FFmpeg: {exc}") from exc
    finally:
        if staged is not None:
            _safe_unlink(staged)

    if proc.returncode != 0:
        _safe_unlink(partial)
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        raise RenderError(f"FFmpeg terminato con codice {proc.returncode}:\n{stderr_tail}")

    if not partial.exists() or partial.stat().st_size == 0:
        _safe_unlink(partial)
        raise RenderError("FFmpeg non ha prodotto alcun output")

    os.replace(partial, output_mp4_path)
    return output_mp4_path


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Impossibile eliminare %s: %s", path, exc)
