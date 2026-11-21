# AutoClipper – production image
FROM python:3.11-slim-bookworm

# System deps: FFmpeg + build tools for mediapipe / opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime directories
RUN mkdir -p input temp output processed_logs

ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=ollama

ENTRYPOINT ["python", "main.py"]
