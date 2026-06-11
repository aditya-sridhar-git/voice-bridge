"""
VoiceBridge FastAPI server.
Wraps the existing 5-stage pipeline with a REST API.

Endpoints:
  POST /api/process          - upload audio, queue pipeline job
  GET  /api/jobs/{id}        - job status + metadata
  GET  /api/jobs/{id}/logs   - SSE stream of pipeline logs
  GET  /api/jobs/{id}/output - download result WAV
  GET  /api/jobs             - list all jobs
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VoiceBridge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ───────────────────────────────────────────────────────
# job_id -> {status, logs, result, error, input_path, output_path, created_at}
jobs: Dict[str, dict] = {}
JOBS_DIR = ROOT / "runs" / "_api_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


# ── Log capture handler ───────────────────────────────────────────────────────
class JobLogHandler(logging.Handler):
    """Captures log records into the job's log list."""

    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def emit(self, record: logging.LogRecord):
        jobs[self.job_id]["logs"].append({
            "time": time.strftime("%H:%M:%S"),
            "level": record.levelname,
            "msg": self.format(record),
        })


# ── Background pipeline runner ────────────────────────────────────────────────
def _run_pipeline_thread(job_id: str, input_path: str, accent: str, whisper_model: str):
    """Runs in a daemon thread. Updates jobs[job_id] in place."""
    job = jobs[job_id]
    job["status"] = "running"

    # Attach log handler to root logger for this thread
    handler = JobLogHandler(job_id)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    output_path = str(JOBS_DIR / f"{job_id}_output.wav")

    try:
        from pipeline.run_pipeline import run_pipeline, PipelineResult

        result: PipelineResult = run_pipeline(
            input_audio=input_path,
            accent_pair=accent,
            output_path=output_path,
            whisper_model=whisper_model,
            device="cpu",
        )

        job["status"] = "done"
        job["output_path"] = result.output_audio   # use what pipeline actually wrote
        job["result"] = {
            "emotion_label": result.emotion_label,
            "emotion_score": result.emotion_score,
            "accent": result.accent_pair,
            "rewrites": result.rewrites_count,
            "run_dir": str(result.run_dir),
            "duration_s": result.duration_s,
        }

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        logging.exception(f"[job {job_id}] Pipeline failed")

    finally:
        root_logger.removeHandler(handler)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_audio(
    audio: UploadFile = File(...),
    accent: str = Form("indian_american"),
    whisper_model: str = Form("base"),
):
    """Upload audio file and start pipeline job. Returns job_id immediately."""
    job_id = uuid.uuid4().hex[:8]

    # Save uploaded file to a temp location
    suffix = Path(audio.filename).suffix or ".wav"
    input_path = str(JOBS_DIR / f"{job_id}_input{suffix}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Register job
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "accent": accent,
        "filename": audio.filename,
        "input_path": input_path,
        "output_path": None,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    # Launch in background thread (pipeline is CPU-bound, not async)
    t = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, input_path, accent, whisper_model),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "accent": job["accent"],
        "result": job["result"],
        "error": job["error"],
        "log_count": len(job["logs"]),
        "created_at": job["created_at"],
    }


@app.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str, since: int = 0):
    """
    SSE endpoint. Streams new log lines as they arrive.
    Client passes `since` = number of log lines already received.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        idx = since
        while True:
            job = jobs[job_id]
            logs = job["logs"]
            while idx < len(logs):
                entry = logs[idx]
                yield f"data: {json.dumps(entry)}\n\n"
                idx += 1

            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'__end__': True, 'status': job['status']})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/output")
async def download_output(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job not done yet (status={job['status']})")
    if not job["output_path"] or not Path(job["output_path"]).exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(
        job["output_path"],
        media_type="audio/wav",
        filename=f"voicebridge_{job_id}.wav",
    )


@app.get("/api/jobs")
async def list_jobs():
    return [
        {"id": j["id"], "status": j["status"], "filename": j["filename"], "created_at": j["created_at"]}
        for j in sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)
    ]


@app.get("/api/health")
async def health():
    return {"ok": True, "jobs": len(jobs)}


# ── Serve built React frontend ────────────────────────────────────────────────
frontend_dist = ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
