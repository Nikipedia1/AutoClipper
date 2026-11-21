"""AutoClipper - orchestratore CLI.

Esempio:
    python main.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX" --max-clips 3 --force

Pipeline: download -> trascrizione -> selezione AI -> smart crop (face tracking) -> sottotitoli .ass -> rendering .mp4
-> metadati .meta.json -> pulizia temp/ -> registrazione in processed_logs/history.json

Codici di uscita: 0 = almeno una clip prodotta, 1 = errore, 3 = video già processato,
130 = interrotto dall'utente.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from curator import CurationError, ViralClip, curate_video
from downloader import (
    AlreadyProcessedError,
    DownloaderError,
    download_video,
    mark_as_processed,
)
from editor import RenderError, render_vertical_clip
from smart_crop import SmartCropError, crop_with_face_tracking
from subtitle_gen import generate_ass_subtitles
from transcriber import TranscriptionError, transcribe_video

logger = logging.getLogger("autoclipper")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_ALREADY_PROCESSED = 3
EXIT_INTERRUPTED = 130


def _slugify(text: str, max_len: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-").lower()
    return slug[:max_len].strip("-") or "clip"


def _clip_name(video_id: str, clip: ViralClip) -> str:
    return f"{video_id}_clip{clip.clip_index:02d}_{_slugify(clip.title)}"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} non è un intero") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("deve essere >= 1")
    return number


def _write_meta(
    meta_path: Path,
    clip: ViralClip,
    mp4_path: Path,
    video_info: dict[str, Any],
    url: str,
) -> None:
    meta = {
        "video_path": str(mp4_path.resolve()),
        "title": clip.title,
        "caption": clip.social_caption,
        "hashtags": clip.hashtags,
        "viral_score": clip.viral_score,
        "platforms": list(clip.recommended_platforms),
        "clip_index": clip.clip_index,
        "start_sec": clip.start_sec,
        "end_sec": clip.end_sec,
        "duration_sec": clip.duration_sec,
        "hook_reason": clip.hook_reason,
        "source": {
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "url": url,
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_temp(video_id: str) -> int:
    removed = 0
    escaped = glob.escape(video_id)
    for pattern in (f"{escaped}_*", f"{escaped}.*"):
        for path in config.TEMP_DIR.glob(pattern):
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Impossibile eliminare %s: %s", path.name, exc)
    return removed


def _process_clip(
    clip: ViralClip,
    words: list[dict[str, Any]],
    video_info: dict[str, Any],
    url: str,
    *,
    smart_crop: bool = True,
) -> Path:
    """Genera .ass, rende l'.mp4 e salva il .meta.json. Restituisce il percorso dell'mp4."""
    name = _clip_name(video_info["video_id"], clip)
    ass_path = config.TEMP_DIR / f"{name}.ass"
    mp4_path = config.OUTPUT_DIR / f"{name}.mp4"
    meta_path = config.OUTPUT_DIR / f"{name}.meta.json"
    source_video = Path(video_info["video_path"])

    generate_ass_subtitles(words, clip.start_sec, clip.end_sec, ass_path)

    if smart_crop:
        temp_cropped = config.TEMP_DIR / f"{name}_smart.mp4"
        try:
            crop_with_face_tracking(
                source_video,
                clip.start_sec,
                clip.end_sec,
                temp_cropped,
            )
            render_vertical_clip(
                temp_cropped,
                ass_path,
                0.0,
                clip.duration_sec,
                mp4_path,
            )
        finally:
            temp_cropped.unlink(missing_ok=True)
    else:
        render_vertical_clip(
            source_video,
            ass_path,
            clip.start_sec,
            clip.end_sec,
            mp4_path,
        )

    _write_meta(meta_path, clip, mp4_path, video_info, url)
    return mp4_path


def run_pipeline(
    url: str,
    max_clips: int,
    force: bool,
    keep_temp: bool,
    language: str | None,
    smart_crop: bool = True,
) -> int:
    config.ensure_directories()

    logger.info("1/5 Download")
    try:
        info = download_video(url, force=force)
    except AlreadyProcessedError as exc:
        logger.warning("%s. Usa --force per rielaborarlo.", exc)
        return EXIT_ALREADY_PROCESSED

    video_id = info["video_id"]
    logger.info("Video: %s (%s, %.0fs)", info["title"], video_id, info["duration"])
    logger.info("Smart crop (face tracking): %s", "ON" if smart_crop else "OFF")

    logger.info("2/5 Trascrizione (device=%s, compute=%s)", config.DEVICE, config.COMPUTE_TYPE)
    transcript = transcribe_video(info["video_path"], video_id, force=force, language=language)
    words: list[dict[str, Any]] = transcript["words"]

    logger.info("3/5 Selezione clip virali")
    selection = curate_video(video_id, info["title"])
    clips = selection.clips[:max_clips]
    logger.info("Clip selezionate: %d", len(clips))

    logger.info("4/5 Sottotitoli e rendering")
    rendered: list[Path] = []
    failed: list[int] = []
    for clip in clips:
        logger.info(
            "Clip %d/%d: '%s' (%.2fs-%.2fs, score %d)",
            clip.clip_index, len(clips), clip.title,
            clip.start_sec, clip.end_sec, clip.viral_score,
        )
        try:
            rendered.append(
                _process_clip(clip, words, info, url, smart_crop=smart_crop)
            )
        except (RenderError, SmartCropError, ValueError, OSError) as exc:
            logger.error("Clip %d fallita: %s", clip.clip_index, exc)
            failed.append(clip.clip_index)

    if not rendered:
        logger.error("Nessuna clip prodotta. File temporanei conservati in %s", config.TEMP_DIR)
        return EXIT_FAILURE

    logger.info("5/5 Finalizzazione")
    mark_as_processed(video_id, info["title"], url)
    if keep_temp:
        logger.info("--keep-temp attivo: file temporanei conservati")
    else:
        logger.info("Rimossi %d file temporanei", _cleanup_temp(video_id))

    if failed:
        logger.warning(
            "Clip non riuscite: %s. Rilancia con --force per riprovare.",
            ", ".join(str(i) for i in failed),
        )
    logger.info("Completato: %d clip in %s", len(rendered), config.OUTPUT_DIR)
    for path in rendered:
        logger.info("  - %s", path.name)
    return EXIT_OK


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autoclipper",
        description="Scarica un video YouTube e ne ricava clip verticali 9:16 sottotitolate.",
    )
    parser.add_argument("--url", required=True, help="URL del video YouTube")
    parser.add_argument(
        "--max-clips", type=_positive_int, default=config.MAX_CLIPS,
        help=f"numero massimo di clip da produrre (default: {config.MAX_CLIPS})",
    )
    parser.add_argument("--force", action="store_true",
                        help="ignora lo storico anti-duplicati e ritrascrive")
    parser.add_argument("--keep-temp", action="store_true",
                        help="non eliminare i file temporanei in temp/")
    parser.add_argument("--language", default=None,
                        help="codice lingua per Whisper (es. it, en)")
    parser.add_argument("--no-smart-crop", action="store_true",
                        help="disabilita il face-tracking e usa il crop centrale classico")
    parser.add_argument("-v", "--verbose", action="store_true", help="log dettagliato")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    if args.max_clips > config.MAX_CLIPS:
        logger.warning(
            "--max-clips=%d supera il massimo selezionabile (%d): verranno prodotte al più %d clip.",
            args.max_clips, config.MAX_CLIPS, config.MAX_CLIPS,
        )

    try:
        use_smart = config.SMART_CROP and not args.no_smart_crop
        return run_pipeline(
            url=args.url,
            max_clips=args.max_clips,
            force=args.force,
            keep_temp=args.keep_temp,
            language=args.language,
            smart_crop=use_smart,
        )
    except KeyboardInterrupt:
        logger.warning("Interrotto dall'utente")
        return EXIT_INTERRUPTED
    except (
        DownloaderError, TranscriptionError, CurationError,
        config.ConfigError, FileNotFoundError, ValueError,
    ) as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
