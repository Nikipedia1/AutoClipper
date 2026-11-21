"""Generazione di sottotitoli ASS (Advanced SubStation Alpha) con evidenziazione karaoke.

Per ogni blocco di 3-4 parole viene emesso un evento `Dialogue` per ciascuna parola:
il testo mostra l'intero blocco e SOLO la parola pronunciata in quell'istante ha
il colore di evidenziazione, mentre le altre restano bianche.
I timestamp in uscita sono relativi all'inizio della clip (t - start_sec).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stile
# --------------------------------------------------------------------------- #
FONT_NAME = "Arial Black"  # alternativa: "Montserrat ExtraBold" (da installare su Windows)
FONT_SIZE = 68
OUTLINE = 4
SHADOW = 2
ALIGNMENT = 2  # centro-basso
MARGIN_V = 380
MARGIN_LR = 60

# Colori ASS (formato &HAABBGGRR&)
COLOR_WHITE = "&H00FFFFFF&"
COLOR_YELLOW = "&H0000FFFF&"
COLOR_LIME = "&H0032FF32&"
COLOR_OUTLINE = "&H00000000&"
COLOR_SHADOW = "&H80000000&"  # nero semi-trasparente

# --------------------------------------------------------------------------- #
# Raggruppamento parole
# --------------------------------------------------------------------------- #
MAX_WORDS_PER_BLOCK = 4
PAUSE_BREAK_SEC = 0.35
MAX_CHARS_PER_BLOCK = 26
END_HOLD_SEC = 0.12  # la riga resta visibile un attimo dopo l'ultima parola
BOUNDARY_TOLERANCE_SEC = 0.05
_SENTENCE_END = (".", "?", "!", "…")
_COLOR_RE = re.compile(r"^&H[0-9A-Fa-f]{6,8}&?$")


@dataclass(frozen=True)
class _Word:
    text: str
    start: float  # secondi relativi all'inizio della clip
    end: float


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _escape_ass_text(text: str) -> str:
    """Rimuove i caratteri che ASS interpreterebbe come override/escape."""
    text = text.replace("\\", "").replace("{", "(").replace("}", ")")
    return " ".join(text.split())  # elimina newline, tab e spazi doppi


def _normalize_color(color: str) -> str:
    color = color.strip()
    if not _COLOR_RE.match(color):
        raise ValueError(f"Colore ASS non valido: {color!r} (atteso es. '&H0000FFFF&')")
    return color if color.endswith("&") else color + "&"


def _style_color(color: str) -> str:
    """Nelle righe `Style:` il colore è senza '&' finale."""
    return color.rstrip("&")


def _cs_to_timestamp(centiseconds: int) -> str:
    hours, rest = divmod(max(0, centiseconds), 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _collect_words(
    words: list[dict[str, Any]], start_sec: float, end_sec: float
) -> list[_Word]:
    clip_len = end_sec - start_sec
    collected: list[_Word] = []
    for raw in words:
        try:
            text = _escape_ass_text(str(raw["word"]))
            w_start = float(raw["start"])
            w_end = float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text:
            continue
        if w_end < w_start:
            w_end = w_start
        if w_start < start_sec - BOUNDARY_TOLERANCE_SEC or w_start >= end_sec:
            continue
        rel_start = max(0.0, w_start - start_sec)
        rel_end = min(clip_len, max(w_end - start_sec, rel_start))
        collected.append(_Word(text=text, start=rel_start, end=rel_end))
    collected.sort(key=lambda w: (w.start, w.end))
    return collected


def _group_words(
    words: list[_Word], max_words: int, pause_sec: float
) -> list[list[_Word]]:
    blocks: list[list[_Word]] = []
    current: list[_Word] = []
    chars = 0
    for word in words:
        if current:
            gap = word.start - current[-1].end
            sentence_end = current[-1].text.endswith(_SENTENCE_END)
            too_long = chars + 1 + len(word.text) > MAX_CHARS_PER_BLOCK
            if len(current) >= max_words or gap > pause_sec or sentence_end or too_long:
                blocks.append(current)
                current = []
                chars = 0
        chars += len(word.text) + (1 if current else 0)
        current.append(word)
    if current:
        blocks.append(current)
    return blocks


def _build_header(
    font_name: str, font_size: int, margin_v: int, highlight: str
) -> list[str]:
    style = ",".join(
        [
            "Default",
            font_name,
            str(font_size),
            _style_color(COLOR_WHITE),  # PrimaryColour
            _style_color(highlight),  # SecondaryColour
            _style_color(COLOR_OUTLINE),  # OutlineColour
            _style_color(COLOR_SHADOW),  # BackColour (ombra)
            "0",  # Bold (Arial Black è già pesante)
            "0",  # Italic
            "0",  # Underline
            "0",  # StrikeOut
            "100",  # ScaleX
            "100",  # ScaleY
            "0",  # Spacing
            "0",  # Angle
            "1",  # BorderStyle: contorno + ombra
            str(OUTLINE),
            str(SHADOW),
            str(ALIGNMENT),
            str(MARGIN_LR),
            str(MARGIN_LR),
            str(margin_v),
            "1",  # Encoding
        ]
    )
    return [
        "[Script Info]",
        "Title: AutoClipper subtitles",
        "ScriptType: v4.00+",
        f"PlayResX: {config.TARGET_WIDTH}",
        f"PlayResY: {config.TARGET_HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #


def generate_ass_subtitles(
    words: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    output_ass_path: Path,
    *,
    highlight_color: str = COLOR_YELLOW,
    font_name: str = FONT_NAME,
    font_size: int = FONT_SIZE,
    margin_v: int = MARGIN_V,
    max_words_per_block: int = MAX_WORDS_PER_BLOCK,
    pause_threshold_sec: float = PAUSE_BREAK_SEC,
) -> Path:
    """Genera un file `.ass` con sottotitoli karaoke per la clip [start_sec, end_sec].

    Args:
        words: lista di dict con `word`, `start`, `end` (secondi assoluti nel video
            originale), come prodotta da `transcriber.transcribe_video`.
        start_sec: inizio della clip nel video originale.
        end_sec: fine della clip nel video originale.
        output_ass_path: percorso del file `.ass` da scrivere.
        highlight_color: colore della parola attiva (`COLOR_YELLOW` o `COLOR_LIME`).

    Returns:
        Il percorso del file scritto.

    Raises:
        ValueError: intervallo, colore o parametri non validi.
    """
    if end_sec <= start_sec:
        raise ValueError(f"Intervallo non valido: start={start_sec}, end={end_sec}")
    if max_words_per_block < 1:
        raise ValueError("max_words_per_block deve essere >= 1")

    highlight = _normalize_color(highlight_color)
    base = COLOR_WHITE
    clip_len = end_sec - start_sec

    clip_words = _collect_words(words, start_sec, end_sec)
    if not clip_words:
        logger.warning(
            "Nessuna parola tra %.2fs e %.2fs: il file ASS non conterrà eventi.",
            start_sec,
            end_sec,
        )
    blocks = _group_words(clip_words, max_words_per_block, pause_threshold_sec)

    lines = _build_header(font_name, font_size, margin_v, highlight)

    for b_idx, block in enumerate(blocks):
        next_start = blocks[b_idx + 1][0].start if b_idx + 1 < len(blocks) else clip_len
        block_end = min(block[-1].end + END_HOLD_SEC, next_start, clip_len)

        for i, word in enumerate(block):
            start_cs = round(word.start * 100)
            event_end = block[i + 1].start if i + 1 < len(block) else block_end
            end_cs = max(round(event_end * 100), start_cs + 1)

            parts: list[str] = []
            for j, other in enumerate(block):
                if j == i:
                    parts.append(f"{{\\c{highlight}}}{other.text}{{\\c{base}}}")
                else:
                    parts.append(other.text)
            text = " ".join(parts)

            lines.append(
                f"Dialogue: 0,{_cs_to_timestamp(start_cs)},{_cs_to_timestamp(end_cs)},"
                f"Default,,0,0,0,,{text}"
            )

    output_ass_path = Path(output_ass_path)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    output_ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "ASS generato: %s (%d blocchi, %d parole)", output_ass_path.name, len(blocks), len(clip_words)
    )
    return output_ass_path
