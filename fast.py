import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import subprocess
import os
import shutil
import zipfile
import uvicorn
from typing import Optional, Deque, List, Dict
from collections import deque
import logging
import uuid
import json
from datetime import datetime
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRIQC-Backend")

app = FastAPI()

# Configure directories (using persistent storage)
UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"
JOBS_FOLDER = "/tmp/mriqc_jobs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

# In-memory job tracking (for simplicity)
jobs: Dict[str, Dict] = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ready", 
        "resources": {
            "memory_gb": 16,
            "cpus": 4,
            "disk_space": shutil.disk_usage("/").free // (2**30)
        }
    }

#########################################
# Job Management Endpoints (for Streamlit)
#########################################

@app.post("/submit-job")
async def submit_job(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    session_id: str = Form(""),
    n_procs: str = Form("4"),
    mem_gb: str = Form("16")
):
    """Submit a job for MRIQC processing"""
    
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = Path(JOBS_FOLDER) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    zip_path = job_dir / "bids_data.zip"
    try:
        with open(zip_path, "wb") as f:
            while contents := await bids_zip.read(1024 * 1024):
                f.write(contents)
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Store job info
    jobs[job_id] = {
        "status": "submitted",
        "participant_label": participant_label,
        "modalities": modalities,
        "session_id": session_id,
        "n_procs": n_procs,
        "mem_gb": mem_gb,
        "submitted_at": datetime.now().isoformat(),
        "job_dir": str(job_dir)
    }
    
    # Start processing in background
    asyncio.create_task(process_mriqc_job(job_id))
    
    return {"job_id": job_id, "status": "submitted"}

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download results for a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")
    
    result_zip = Path(job["job_dir"]) / "mriqc_results.zip"
    if not result_zip.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        result_zip,
        filename=f"mriqc_results_{job_id}.zip",
        media_type="application/zip"
    )

@app.delete("/delete-job/{job_id}")
async def delete_job(job_id: str):
    """Clean up job data"""
    if job_id in jobs:
        job_dir = Path(jobs[job_id]["job_dir"])
        if job_dir.exists():
            shutil.rmtree(job_dir)
        del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}

#########################################
# Background Job Processing
#########################################

async def process_mriqc_job(job_id: str):
    """Process MRIQC job in background"""
    try:
        job = jobs[job_id]
        job_dir = Path(job["job_dir"])
        
        # Update status
        job["status"] = "processing"
        job["started_at"] = datetime.now().isoformat()
        
        # Extract BIDS data
        zip_path = job_dir / "bids_data.zip"
        extract_dir = job_dir / "bids_data"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            job["status"] = "failed"
            job["error"] = "Invalid ZIP file format"
            return
        
        # Find BIDS root
        bids_root = find_bids_root(extract_dir)
        if not bids_root:
            job["status"] = "failed"
            job["error"] = "No valid BIDS dataset found"
            return
        
        # Prepare output directory
        output_dir = job_dir / "mriqc_output"
        output_dir.mkdir(exist_ok=True)
        
        # Build MRIQC command
        cmd = [
            "docker", "run", "--rm",
            f"--memory={job['mem_gb']}g",
            f"--cpus={job['n_procs']}",
            "-v", f"{bids_root}:/data:ro",
            "-v", f"{output_dir}:/out",
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", job["participant_label"],
            "-m", *job["modalities"].split(),
            "--nprocs", job["n_procs"],
            "--no-sub",
            "--verbose-reports"
        ]
        
        # Add session ID if provided
        if job["session_id"]:
            cmd += ["--session-id", job["session_id"]]
        
        logger.info(f"Running MRIQC for job {job_id}: {' '.join(cmd)}")
        
        # Execute MRIQC
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
        except subprocess.TimeoutExpired:
            job["status"] = "failed"
            job["error"] = "Processing timed out after 2 hours"
            return
        
        # Save logs
        log_path = job_dir / "mriqc_log.txt"
        with open(log_path, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.write("=== STDOUT ===\n")
            log_file.write(result.stdout)
            log_file.write("\n=== STDERR ===\n")
            log_file.write(result.stderr)
        
        if result.returncode != 0:
            job["status"] = "failed"
            job["error"] = f"MRIQC failed with return code {result.returncode}"
            job["stderr"] = result.stderr[-1000:]
            return
        
        # Package results
        result_zip_path = job_dir / "mriqc_results.zip"
        shutil.make_archive(
            base_name=str(result_zip_path).replace(".zip", ""),
            format="zip",
            root_dir=output_dir
        )
        
        # Update job status
        job["status"] = "complete"
        job["completed_at"] = datetime.now().isoformat()
        job["result_path"] = str(result_zip_path)
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error processing job {job_id}: {str(e)}")
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Unexpected error: {str(e)}"

#########################################
# Original MRIQC Endpoint (for backward compatibility)
#########################################

@app.post("/run-mriqc")
async def run_mriqc_endpoint(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    n_procs: str = Form("4"),
    mem_gb: str = Form("16")
):
    """Legacy endpoint - redirects to job system"""
    
    # Create a job and wait for completion
    job_response = await submit_job(bids_zip, participant_label, modalities, "", n_procs, mem_gb)
    job_id = job_response["job_id"]
    
    # Wait for completion (simple polling)
    for _ in range(360):  # Wait up to 1 hour
        await asyncio.sleep(10)
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Job disappeared")
        
        if job["status"] == "complete":
            result_zip = Path(job["job_dir"]) / "mriqc_results.zip"
            return FileResponse(
                result_zip,
                filename=f"mriqc_results_{participant_label}.zip",
                media_type="application/zip"
            )
        elif job["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=job.get("error", "Job failed")
            )
    
    raise HTTPException(status_code=408, detail="Processing timeout")

#########################################
# WebSocket Endpoint
#########################################

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/mriqc")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

#########################################
# Helper Functions
#########################################

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """Locate the BIDS root directory by searching for dataset_description.json"""
    queue: Deque[Path] = deque()
    queue.append(upload_dir)
    
    while queue:
        current = queue.popleft()
        if (current / "dataset_description.json").exists():
            return current
        
        for child in current.iterdir():
            if child.is_dir() and not child.name.startswith('.'):
                queue.append(child)
    
    return None

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        timeout_keep_alive=300
    )
