"""AutoClipper Web Interface + Monitoring Dashboard.

Avvio:
    uvicorn webapp:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import config
from downloader import _load_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("webapp")

app = FastAPI(title="AutoClipper", version="1.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": config.DEVICE,
        "compute_type": config.COMPUTE_TYPE,
        "cuda_available": False,
        "gpu_name": None,
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def _list_output_clips() -> list[dict[str, Any]]:
    clips = []
    out = config.OUTPUT_DIR
    if not out.exists():
        return clips
    for meta_path in sorted(out.glob("*.meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mp4 = Path(meta.get("video_path", ""))
            if not mp4.exists():
                mp4 = out / meta_path.name.replace(".meta.json", ".mp4")
            clips.append({
                "meta_path": str(meta_path),
                "mp4_path": str(mp4) if mp4.exists() else None,
                "mp4_name": mp4.name if mp4.exists() else None,
                "title": meta.get("title", ""),
                "viral_score": meta.get("viral_score"),
                "duration_sec": meta.get("duration_sec"),
                "platforms": meta.get("platforms", []),
                "created_at": meta.get("created_at"),
                "source_title": meta.get("source", {}).get("title", ""),
            })
        except Exception as exc:
            logger.warning("Meta non leggibile %s: %s", meta_path.name, exc)
    return clips[:50]


def _run_job(job_id: str, url: str, max_clips: int, force: bool, language: str | None, smart_crop: bool) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    cmd = [sys.executable, "main.py", "--url", url, "--max-clips", str(max_clips)]
    if force:
        cmd.append("--force")
    if language:
        cmd.extend(["--language", language])
    if not smart_crop:
        cmd.append("--no-smart-crop")

    try:
        proc = subprocess.run(
            cmd, cwd=str(config.BASE_DIR), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3600,
        )
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "success" if proc.returncode == 0 else "failed"
            JOBS[job_id]["returncode"] = proc.returncode
            JOBS[job_id]["stdout"] = (proc.stdout or "")[-4000:]
            JOBS[job_id]["stderr"] = (proc.stderr or "")[-2000:]
            JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["stderr"] = str(exc)
            JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    history = _load_history()
    clips = _list_output_clips()
    gpu = _gpu_info()
    with JOBS_LOCK:
        recent_jobs = list(JOBS.values())[-10:][::-1]
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "history": history,
            "history_count": len(history),
            "clips": clips,
            "gpu": gpu,
            "jobs": recent_jobs,
            "smart_crop_default": config.SMART_CROP,
        },
    )


@app.post("/run")
async def run_job(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    max_clips: int = Form(3),
    force: bool = Form(False),
    language: str = Form(""),
    smart_crop: bool = Form(True),
):
    url = url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "URL non valido")
    max_clips = max(1, min(max_clips, config.MAX_CLIPS))

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "url": url,
            "max_clips": max_clips,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    background_tasks.add_task(
        _run_job, job_id, url, max_clips, force, language.strip() or None, smart_crop
    )
    return RedirectResponse(url="/?job=" + job_id, status_code=303)


@app.get("/api/jobs")
async def api_jobs():
    with JOBS_LOCK:
        return list(JOBS.values())[-20:][::-1]


@app.get("/api/status")
async def api_status():
    return {
        "device": config.DEVICE,
        "compute_type": config.COMPUTE_TYPE,
        "smart_crop": config.SMART_CROP,
        "llm_provider": config.LLM_PROVIDER,
        "history_count": len(_load_history()),
        "output_clips": len(list(config.OUTPUT_DIR.glob("*.mp4"))) if config.OUTPUT_DIR.exists() else 0,
        "gpu": _gpu_info(),
    }


@app.get("/download/{filename}")
async def download_clip(filename: str):
    safe = Path(filename).name
    path = config.OUTPUT_DIR / safe
    if not path.is_file() or path.suffix.lower() not in (".mp4", ".json"):
        raise HTTPException(404, "File non trovato")
    return FileResponse(path, filename=safe)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp:app", host="0.0.0.0", port=8080, reload=True)
