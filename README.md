# AutoClipper

**AI-powered pipeline that turns long YouTube videos into ready-to-publish vertical Shorts / TikTok / Reels.**

Download → Transcribe (faster-whisper) → LLM viral-clip selection → **Smart face-tracking crop (MediaPipe)** → Karaoke ASS subtitles → Render 1080×1920 → Multi-platform publishing via n8n + Postiz.

---

## Features

| Stage | Technology |
|-------|------------|
| Download | `yt-dlp` – H.264 + AAC 1080p, anti-duplicate history |
| Transcription | `faster-whisper` (medium / int8 / CPU) with word-level timestamps |
| Viral curation | Local Ollama **or** OpenAI / Gemini – Pydantic-validated JSON |
| Smart crop | **MediaPipe Face Detection + EMA smoothing** → dynamic 9:16 |
| Subtitles | Advanced SubStation Alpha karaoke (yellow highlight) |
| Render | FFmpeg – crop / scale / burn ASS / loudnorm |
| Automation | Full n8n workflow (RSS → execute → Postiz publish) |

---

## Quick start (Windows / Linux / macOS)

### 1. Prerequisites

- Python 3.10 – 3.12
- FFmpeg in PATH  
  ```bash
  # Windows
  winget install Gyan.FFmpeg
  # Ubuntu/Debian
  sudo apt install ffmpeg
  # macOS
  brew install ffmpeg
  ```

### 2. Install

```bash
git clone https://github.com/Nikipedia1/AutoClipper.git
cd AutoClipper
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure LLM (choose one)

**A. Local Ollama (recommended, free, private)**

```bash
# Install Ollama → https://ollama.com
ollama pull qwen2.5:7b
# default is already set in config.py
```

**B. OpenAI / Gemini**

```bash
export LLM_PROVIDER=openai          # or gemini
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini        # optional
```

### 4. Run

```bash
python main.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX" --max-clips 3

# Useful flags
python main.py --url "..." --force --keep-temp --language it -v
```

Output appears in `output/`:
- `*_clip01_title.mp4`
- `*_clip01_title.meta.json` (title, caption, hashtags, platforms, scores)

---

## Smart Crop (face tracking)

The new module `smart_crop.py` replaces the static center crop:

```python
from smart_crop import crop_with_face_tracking

crop_with_face_tracking(
    input_video=Path("input/video.mp4"),
    start_sec=42.0,
    end_sec=95.0,
    temp_cropped_video=Path("temp/smart.mp4"),
)
```

- Detects faces every frame with MediaPipe
- Picks the largest face
- Exponential Moving Average (EMA) smoothing prevents jitter
- Falls back to last known position / center when no face is found
- Outputs a clean 1080×1920 muted video + original audio remuxed

To enable it inside the pipeline, see the comment in `main.py` / `editor.py` (or ask for the one-line patch).

---

## Full automation with n8n + Postiz

1. Import `n8n_autoclipper_workflow.json` into your n8n instance.
2. Set the YouTube channel ID in the RSS node.
3. Point the **Execute Command** node to your `main.py`.
4. Create a Postiz API key + integration IDs (YouTube, TikTok, Instagram).
5. The workflow:
   - checks the channel RSS every 30 min
   - runs AutoClipper on new videos
   - uploads the resulting clips to Postiz
   - schedules them on YouTube Shorts + TikTok + Instagram Reels
   - moves finished files to `output/published/`

---

## Docker

```bash
docker compose --profile ollama up -d   # optional local LLM
docker compose run --rm autoclipper --url "https://..." --max-clips 3
```

---

## Project layout

```
AutoClipper/
├── main.py                 # CLI orchestrator
├── config.py               # all settings + env overrides
├── downloader.py           # yt-dlp + history anti-dupe
├── transcriber.py          # faster-whisper word timestamps
├── curator.py              # LLM viral clip selection (Pydantic)
├── smart_crop.py           # MediaPipe face-tracking 9:16
├── subtitle_gen.py         # ASS karaoke subtitles
├── editor.py               # FFmpeg render + burn
├── n8n_autoclipper_workflow.json
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` / `openai` / `gemini` |
| `LLM_API_KEY` | – | Required for openai / gemini |
| `LLM_MODEL` | provider default | Model name |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | |
| `WHISPER_LANGUAGE` | auto | e.g. `it`, `en` |
| `LLM_TEMPERATURE` | `0.4` | |
| `LLM_MAX_TRANSCRIPT_CHARS` | `24000` | Chunk size for long videos |

---

## Cloudflare notes

- The core pipeline is a local / Docker Python process (needs GPU/CPU + FFmpeg).
- For **zero-public-IP** exposure of n8n or Postiz you can use **Cloudflare Tunnel**:
  ```bash
  cloudflared tunnel --url http://localhost:5678   # n8n
  ```
- Output videos can be uploaded to **Cloudflare R2** (S3-compatible) and then referenced by Postiz `upload-from-url`.
- A static documentation site for this repo can be deployed on **Cloudflare Pages** directly from the GitHub repository.

---

## License

MIT – see [LICENSE](LICENSE).

---

Made for creators who want professional vertical clips without the manual editing grind.
