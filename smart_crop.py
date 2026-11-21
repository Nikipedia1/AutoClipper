"""Smart crop 9:16 con face-tracking dinamico (MediaPipe + OpenCV + EMA).

Pipeline:
  1. Apre il video originale e si posiziona su start_sec.
  2. Per ogni frame del segmento [start_sec, end_sec]:
       - rileva volti con mediapipe.solutions.face_detection
       - sceglie il volto con bounding-box più grande
       - calcola la coordinata X del centro del volto
       - applica smoothing esponenziale (EMA) per evitare micro-vibrazioni
       - ritaglia un riquadro 9:16 centrato su quella X (clamp ai bordi)
  3. Scrive un video muto intermedio (OpenCV VideoWriter).
  4. Rimonta l'audio originale del segmento con FFmpeg e produce
     il file finale in temp_cropped_video (MP4 H.264 + AAC).

Il file risultante è già in 1080×1920 e può essere passato direttamente
a editor.render_vertical_clip (che brucerà i sottotitoli ASS).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Costanti
# --------------------------------------------------------------------------- #
EMA_ALPHA: float = 0.18          # più basso = più fluido (0.10–0.25 consigliato)
MIN_DETECTION_CONFIDENCE: float = 0.55
MODEL_SELECTION: int = 1         # 0 = short-range, 1 = full-range
TARGET_W: int = config.TARGET_WIDTH   # 1080
TARGET_H: int = config.TARGET_HEIGHT  # 1920
ASPECT: float = TARGET_W / TARGET_H   # 9/16


class SmartCropError(RuntimeError):
    """Errore durante il face-tracking / crop intelligente."""


# --------------------------------------------------------------------------- #
# Helper FFmpeg
# --------------------------------------------------------------------------- #


def _ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SmartCropError(
            "FFmpeg non trovato nel PATH. Su Windows: `winget install Gyan.FFmpeg` "
            "e riapri il terminale."
        )
    return ffmpeg


def _mux_audio(
    muted_video: Path,
    source_video: Path,
    start_sec: float,
    duration: float,
    output_path: Path,
) -> None:
    """Copia il video muto e aggiunge l'audio originale del segmento."""
    ffmpeg = _ensure_ffmpeg()
    partial = output_path.with_name(f"{output_path.stem}.partial_{uuid.uuid4().hex[:8]}.mp4")

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
        str(source_video.resolve()),
        "-i",
        str(muted_video.resolve()),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-shortest",
        "-movflags",
        "+faststart",
        str(partial.resolve()),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        _safe_unlink(partial)
        raise SmartCropError("FFmpeg timeout durante il mux audio") from exc
    except OSError as exc:
        _safe_unlink(partial)
        raise SmartCropError(f"Impossibile avviare FFmpeg: {exc}") from exc

    if proc.returncode != 0:
        _safe_unlink(partial)
        stderr_tail = (proc.stderr or "").strip()[-1500:]
        raise SmartCropError(f"FFmpeg mux fallito (code {proc.returncode}):\n{stderr_tail}")

    if not partial.exists() or partial.stat().st_size == 0:
        _safe_unlink(partial)
        raise SmartCropError("FFmpeg non ha prodotto alcun output audio-mux")

    os.replace(partial, output_path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Impossibile eliminare %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Face detection + EMA
# --------------------------------------------------------------------------- #


def _largest_face_center_x(
    detections,
    frame_w: int,
) -> float | None:
    """Restituisce la X normalizzata [0,1] del centro del volto più grande, o None."""
    if not detections:
        return None

    best_area = -1.0
    best_cx = None
    for det in detections:
        # relative_bounding_box: xmin, ymin, width, height  (tutti normalizzati 0-1)
        box = det.location_data.relative_bounding_box
        area = box.width * box.height
        if area > best_area:
            best_area = area
            best_cx = box.xmin + box.width / 2.0

    if best_cx is None:
        return None
    return float(np.clip(best_cx, 0.0, 1.0))


def _crop_frame(
    frame: np.ndarray,
    center_x_norm: float,
    out_w: int = TARGET_W,
    out_h: int = TARGET_H,
) -> np.ndarray:
    """Ritaglia il frame a 9:16 centrato su center_x_norm e lo ridimensiona a out_w×out_h."""
    h, w = frame.shape[:2]
    crop_w = int(round(h * ASPECT))          # larghezza del crop 9:16 alla stessa altezza
    if crop_w > w:
        # Sorgente più stretta del 9:16: scale + letterbox (ma in pratica non succede
        # con video YouTube landscape). Fallback: scale-to-fill + crop centrale.
        scale = max(out_w / w, out_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = max(0, (nw - out_w) // 2)
        y0 = max(0, (nh - out_h) // 2)
        return resized[y0 : y0 + out_h, x0 : x0 + out_w]

    # Calcola la posizione X del crop (in pixel)
    cx = center_x_norm * w
    x0 = int(round(cx - crop_w / 2.0))
    x0 = max(0, min(x0, w - crop_w))
    cropped = frame[:, x0 : x0 + crop_w]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #


def crop_with_face_tracking(
    input_video: Path,
    start_sec: float,
    end_sec: float,
    temp_cropped_video: Path,
    *,
    ema_alpha: float = EMA_ALPHA,
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
) -> Path:
    """Esegue face-tracking dinamico + crop 9:16 sul segmento [start_sec, end_sec].

    Args:
        input_video: percorso del video sorgente (landscape, tipicamente 16:9).
        start_sec: inizio del segmento (secondi assoluti).
        end_sec: fine del segmento (secondi assoluti).
        temp_cropped_video: percorso di destinazione del video 1080×1920 con audio.
        ema_alpha: coefficiente di smoothing esponenziale (0 < alpha ≤ 1).
        min_detection_confidence: soglia MediaPipe Face Detection.

    Returns:
        Il percorso del file scritto (temp_cropped_video).

    Raises:
        FileNotFoundError: video sorgente assente.
        ValueError: intervallo non valido.
        SmartCropError: errori di lettura, scrittura o FFmpeg.
    """
    input_video = Path(input_video)
    temp_cropped_video = Path(temp_cropped_video)

    if not input_video.is_file():
        raise FileNotFoundError(f"Video non trovato: {input_video}")
    if end_sec <= start_sec:
        raise ValueError(f"Intervallo non valido: start={start_sec}, end={end_sec}")
    if not (0.0 < ema_alpha <= 1.0):
        raise ValueError("ema_alpha deve essere in (0, 1]")

    duration = end_sec - start_sec
    temp_cropped_video.parent.mkdir(parents=True, exist_ok=True)

    # --- Apertura video -------------------------------------------------------
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise SmartCropError(f"Impossibile aprire il video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise SmartCropError("Dimensioni video non valide")

    # Seek all'inizio del segmento (preciso al frame)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps))) if total_frames else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # --- VideoWriter (muto) ---------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    muted_path = temp_cropped_video.with_name(
        f"{temp_cropped_video.stem}_muted_{uuid.uuid4().hex[:8]}.mp4"
    )
    writer = cv2.VideoWriter(
        str(muted_path),
        fourcc,
        fps,
        (TARGET_W, TARGET_H),
    )
    if not writer.isOpened():
        cap.release()
        raise SmartCropError(f"Impossibile creare VideoWriter: {muted_path}")

    # --- Face Detection -------------------------------------------------------
    mp_face = mp.solutions.face_detection
    smoothed_x: float | None = None          # ultima X EMA (normalizzata 0-1)
    frames_written = 0
    frames_with_face = 0

    try:
        with mp_face.FaceDetection(
            model_selection=MODEL_SELECTION,
            min_detection_confidence=min_detection_confidence,
        ) as detector:
            while True:
                if end_frame is not None and frames_written >= (end_frame - start_frame):
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                # Converti BGR → RGB per MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = detector.process(rgb)
                rgb.flags.writeable = True

                raw_x = _largest_face_center_x(results.detections, src_w)

                if raw_x is not None:
                    frames_with_face += 1
                    if smoothed_x is None:
                        smoothed_x = raw_x
                    else:
                        # EMA: s_t = α * x_t + (1-α) * s_{t-1}
                        smoothed_x = ema_alpha * raw_x + (1.0 - ema_alpha) * smoothed_x
                # se nessun volto: mantiene l'ultima posizione valida

                # Fallback iniziale / totale assenza di volti → centro
                center = smoothed_x if smoothed_x is not None else 0.5

                cropped = _crop_frame(frame, center)
                writer.write(cropped)
                frames_written += 1

                if frames_written % 90 == 0:
                    logger.debug(
                        "Smart-crop: %d frame scritti (volto rilevato: %d)",
                        frames_written,
                        frames_with_face,
                    )
    finally:
        writer.release()
        cap.release()

    if frames_written == 0:
        _safe_unlink(muted_path)
        raise SmartCropError("Nessun frame letto dal segmento richiesto")

    logger.info(
        "Smart-crop completato: %d frame, volto rilevato su %d (%.0f%%), EMA=%.2f",
        frames_written,
        frames_with_face,
        100.0 * frames_with_face / max(1, frames_written),
        ema_alpha,
    )

    # --- Mux audio originale --------------------------------------------------
    try:
        _mux_audio(muted_path, input_video, start_sec, duration, temp_cropped_video)
    finally:
        _safe_unlink(muted_path)

    logger.info("Video smart-crop salvato: %s", temp_cropped_video.name)
    return temp_cropped_video
