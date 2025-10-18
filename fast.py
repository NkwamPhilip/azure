# main.py
import os
import uuid
import shutil
import zipfile
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional Redis import (installed in Dockerfile)
USE_REDIS = os.getenv("USE_REDIS", "true").lower() in ("1", "true", "yes")

rdb = None
if USE_REDIS:
    try:
        import redis  # type: ignore
        REDIS_HOST = os.getenv("REDIS_HOST", "redis")
        REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        rdb.ping()
        print(f"[main] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"[main] Warning: cannot connect to Redis: {e}. Falling back to in-memory tracking.")
        rdb = None
        USE_REDIS = False

app = FastAPI(title="MRIQC Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent directories (mount these from docker-compose)
BASE_DIR = Path(os.getenv("MNT_BASE", "/mnt"))
UPLOAD_ROOT = BASE_DIR / "mriqc_upload"
OUTPUT_ROOT = BASE_DIR / "mriqc_output"
RESULT_ROOT = BASE_DIR / "mriqc_results"

for p in (UPLOAD_ROOT, OUTPUT_ROOT, RESULT_ROOT):
    p.mkdir(parents=True, exist_ok=True)

# in-memory fallback storage
_jobs: Dict[str, Dict] = {}

# ---------- helper functions for job status tracking ----------
def set_status(job_id: str, status: dict):
    key = f"mriqc:{job_id}"
    if USE_REDIS and rdb:
        rdb.set(key, json.dumps(status))
    else:
        _jobs[job_id] = status

def get_status(job_id: str):
    key = f"mriqc:{job_id}"
    if USE_REDIS and rdb:
        raw = rdb.get(key)
        return json.loads(raw) if raw else None
    return _jobs.get(job_id)

def clear_status(job_id: str):
    key = f"mriqc:{job_id}"
    if USE_REDIS and rdb:
        rdb.delete(key)
    else:
        _jobs.pop(job_id, None)

# ---------- HTTP endpoints ----------
@app.get("/health")
def health():
    """Simple health check"""
    return {"status": "ok"}

@app.post("/submit-job")
async def submit_job(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form(...),
    modalities: str = Form(...),
    session_id: str = Form("baseline"),
    n_procs: int = Form(12),
    mem_gb: int = Form(48),
):
    """
    Receive a BIDS ZIP and start MRIQC as an async background task.
    Return a job_id that the client can poll with /job-status/{job_id}
    """
    # Basic validation
    if not participant_label:
        raise HTTPException(status_code=400, detail="participant_label required")

    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = UPLOAD_ROOT / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_upload_dir / "bids_dataset.zip"

    # Save uploaded zip
    try:
        with open(zip_path, "wb") as out_f:
            while True:
                chunk = await bids_zip.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Extract
    try:
        extract_dir = job_upload_dir / "bids"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    # Set pending status and start background MRIQC process
    set_status(job_id, {"status": "pending", "result": None})
    output_dir = OUTPUT_ROOT / job_id
    asyncio.create_task(run_mriqc_job(job_id, extract_dir, output_dir, participant_label, modalities, n_procs, mem_gb, session_id))
    return {"job_id": job_id}

@app.get("/job-status/{job_id}")
def job_status(job_id: str):
    s = get_status(job_id)
    if not s:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return s

@app.get("/download/{job_id}")
def download_result(job_id: str):
    result_zip = RESULT_ROOT / f"{job_id}.zip"
    if not result_zip.exists():
        raise HTTPException(status_code=404, detail="Result not ready or not found")
    return FileResponse(result_zip, filename=f"mriqc_results_{job_id}.zip", media_type="application/zip")

@app.delete("/delete-job/{job_id}")
def delete_job(job_id: str):
    """
    Delete upload, output and packaged result for a job (called after user downloads).
    """
    try:
        shutil.rmtree(UPLOAD_ROOT / job_id, ignore_errors=True)
        shutil.rmtree(OUTPUT_ROOT / job_id, ignore_errors=True)
        result_zip = RESULT_ROOT / f"{job_id}.zip"
        if result_zip.exists():
            result_zip.unlink()
        clear_status(job_id)
        return {"status": "deleted"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- MRIQC runner ----------
async def run_mriqc_job(job_id: str, bids_dir: Path, output_dir: Path,
                        participant_label: str, modalities: str,
                        n_procs: int, mem_gb: int, session_id: str):
    """
    Launch MRIQC in Docker. Important:
      - docker *options* go BEFORE the image name
      - MRIQC CLI args (like --session-id) go AFTER the image name
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build docker command
        docker_options = [
            "docker", "run", "--rm",
            "--memory", f"{mem_gb}g",
            "--memory-swap", f"{mem_gb}g",
            "--cpus", str(n_procs),
            "-v", f"{str(bids_dir)}:/data:ro",
            "-v", f"{str(output_dir)}:/out",
        ]

        # Image + MRIQC CLI args (MRIQC flags go after image)
        image_and_args = [
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", participant_label,
        ]

        # modalitiy tokens: MRIQC expects -m MODE1 MODE2 ...
        if modalities:
            modality_tokens = modalities.strip().split()
            image_and_args += ["-m"] + modality_tokens

        image_and_args += [
            "--nprocs", str(n_procs),
            "--omp-nthreads", "4",
            "--no-sub",
            "--verbose-reports",
            "--session-id", session_id
        ]

        cmd = docker_options + image_and_args

        set_status(job_id, {"status": "running"})
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # capture stderr and mark failed
            set_status(job_id, {"status": "failed", "error": stderr.decode()})
            return

        # Package outputs into results zip
        zip_path = RESULT_ROOT / f"{job_id}.zip"
        shutil.make_archive(str(zip_path).replace(".zip", ""), 'zip', root_dir=output_dir)
        set_status(job_id, {"status": "complete", "result": str(zip_path)})

    except Exception as e:
        set_status(job_id, {"status": "failed", "error": str(e)})

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
