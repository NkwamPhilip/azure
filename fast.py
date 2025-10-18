import asyncio
import shutil
import subprocess
import zipfile
import uuid
import os
from pathlib import Path
from typing import Dict, Optional
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import WebSocket, WebSocketDisconnect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRIQC-Backend")

app = FastAPI()

# Job storage - keeps track of all jobs
jobs: Dict[str, dict] = {}

# Base directory for all jobs
JOBS_FOLDER = "/tmp/mriqc_jobs"
os.makedirs(JOBS_FOLDER, exist_ok=True)

#########################################
# Health Check
#########################################

@app.get("/health")
async def health_check():
    return {
        "status": "ready",
        "resources": {
            "memory_gb": 64,
            "cpus": 16,
            "disk_space": shutil.disk_usage("/").free // (2**30)
        }
    }

#########################################
# Job Submission
#########################################

@app.post("/submit-job")
async def submit_job(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    session_id: str = Form("baseline"),
    n_procs: str = Form("4"),
    mem_gb: str = Form("16")
):
    """Submit a new MRIQC job"""
    
    job_id = str(uuid.uuid4())
    job_dir = Path(JOBS_FOLDER) / job_id
    bids_data_dir = job_dir / "bids_data"
    mriqc_output_dir = job_dir / "mriqc_output"
    
    # Create directories
    bids_data_dir.mkdir(parents=True, exist_ok=True)
    mriqc_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "job_id": job_id,
        "participant_label": participant_label,
        "modalities": modalities,
        "session_id": session_id
    }
    
    try:
        # Save and extract BIDS zip
        zip_path = job_dir / "bids_input.zip"
        with open(zip_path, "wb") as f:
            content = await bids_zip.read()
            f.write(content)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(bids_data_dir)
        
        logger.info(f"BIDS data extracted for job {job_id}")
        
        # Find BIDS root
        bids_root = find_bids_root(bids_data_dir)
        if not bids_root:
            raise HTTPException(status_code=400, detail="No valid BIDS dataset found")
        
        # Parse modalities
        modality_list = modalities.split()
        
        # Build Docker command
        cmd = [
            "docker", "run", "--rm",
            f"--memory={mem_gb}g",
            f"--cpus={n_procs}",
            "-v", f"{bids_root}:/data:ro",
            "-v", f"{mriqc_output_dir}:/out",
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", participant_label,
            "-m", *modality_list,
            "--nprocs", n_procs,
            "--no-sub",
            "--verbose-reports"
        ]
        
        if session_id and session_id.lower() != "baseline":
            cmd += ["--session-id", session_id]
        
        logger.info(f"Running MRIQC for job {job_id}: {' '.join(cmd)}")
        
        # Run MRIQC in background
        asyncio.create_task(run_mriqc_async(job_id, cmd, mriqc_output_dir))
        
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "message": "MRIQC job started"
        })
        
    except Exception as e:
        logger.error(f"Unexpected error processing job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

#########################################
# Background Task Runner
#########################################

async def run_mriqc_async(job_id: str, cmd: list, output_dir: Path):
    """Run MRIQC command asynchronously"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Save logs
        log_path = output_dir / "mriqc_log.txt"
        with open(log_path, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout.decode())
            f.write("\n=== STDERR ===\n")
            f.write(stderr.decode())
        
        if process.returncode == 0:
            jobs[job_id]["status"] = "complete"
            logger.info(f"Job {job_id} completed successfully")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"MRIQC exited with code {process.returncode}"
            logger.error(f"Job {job_id} failed with return code {process.returncode}")
            
    except Exception as e:
        logger.exception(f"Error running MRIQC for job {job_id}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

#########################################
# Job Status
#########################################

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(jobs[job_id])

#########################################
# Download Results
#########################################

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download MRIQC results as a zip file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")
    
    job_dir = Path(JOBS_FOLDER) / job_id
    output_dir = job_dir / "mriqc_output"
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Create zip
    zip_path = job_dir / "results.zip"
    shutil.make_archive(
        str(zip_path.with_suffix("")),
        'zip',
        root_dir=output_dir
    )
    
    return FileResponse(
        zip_path,
        filename=f"mriqc_results_{job_id}.zip",
        media_type="application/zip"
    )

#########################################
# Cleanup Job
#########################################

@app.delete("/delete-job/{job_id}")
async def delete_job(job_id: str):
    """Delete job data from server"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = Path(JOBS_FOLDER) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    del jobs[job_id]
    
    return JSONResponse({"message": f"Job {job_id} deleted"})

#########################################
# Helper Functions
#########################################

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """Find BIDS root by looking for dataset_description.json"""
    from collections import deque
    
    queue = deque([upload_dir])
    
    while queue:
        current = queue.popleft()
        if (current / "dataset_description.json").exists():
            return current
        
        try:
            for child in current.iterdir():
                if child.is_dir() and not child.name.startswith('.'):
                    queue.append(child)
        except PermissionError:
            continue
    
    return None

#########################################
# WebSocket (Optional)
#########################################

@app.websocket("/ws/mriqc")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or handle real-time updates
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=16)
