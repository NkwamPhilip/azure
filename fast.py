import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import subprocess
import os
import shutil
import zipfile
import uvicorn
from typing import Optional, Deque
from collections import deque

app = FastAPI()

# Configure directories (using persistent storage)
UPLOAD_FOLDER = "/mnt/mriqc_upload"  # Changed from /tmp
OUTPUT_FOLDER = "/mnt/mriqc_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ready", "resources": {"memory_gb": 48, "cpus": 12}}

#########################################
# Enhanced MRIQC Endpoint with Dynamic Resource Allocation
#########################################

@app.post("/run-mriqc")
async def run_mriqc_endpoint(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w T2w bold"),  # Now accepts multiple modalities
    n_procs: str = Form("12"),  # Default matches your 16vCPU instance (leaving 4 for system)
    mem_gb: str = Form("48")    # 48GB for MRIQC, leaving 16GB for system
):
    """Enhanced endpoint with dynamic resource allocation"""
    
    # Step 1-5: Same file handling as before
    try:
        contents = await bids_zip.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    zip_path = Path(UPLOAD_FOLDER) / "bids_data.zip"
    with open(zip_path, "wb") as f:
        f.write(contents)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(UPLOAD_FOLDER)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to unzip file: {str(e)}")

    bids_root = find_bids_root(Path(UPLOAD_FOLDER))
    if not bids_root:
        return JSONResponse(
            content={"error": "No directory with dataset_description.json found."},
            status_code=400
        )

    # Enhanced Docker command with dynamic resources
    cmd = [
        "docker", "run", "--rm",
        f"--memory={mem_gb}g", f"--memory-swap={mem_gb}g",
        f"--cpus={n_procs}",
        "-v", f"{bids_root.absolute()}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",
        "/data", "/out",
        "participant",
        "--participant_label", participant_label,
        "-m", *modalities.split(),  # Unpack modalities
        "--nprocs", n_procs,
        "--omp-nthreads", str(min(4, int(n_procs)//3)),  # Auto-scale threads
        "--no-sub"
    ]

    # Run with enhanced error handling
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1hr timeout
    except subprocess.TimeoutExpired:
        return JSONResponse(
            content={"error": "MRIQC processing timed out after 1 hour"},
            status_code=500
        )

    # Log handling (same as before)
    log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
    with open(log_path, "w") as log_file:
        log_file.write(f"=== MRIQC Docker Command ===\n{' '.join(cmd)}\n\n")
        log_file.write("=== Output ===\n")
        log_file.write(result.stdout)
        log_file.write("\n=== Errors ===\n")
        log_file.write(result.stderr)

    if result.returncode != 0:
        return JSONResponse(
            content={
                "error": "MRIQC failed",
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            },
            status_code=500
        )

    # Zip results
    result_zip_path = "/mnt/mriqc_results.zip"  # Persistent storage
    shutil.make_archive(
        base_name=result_zip_path.replace(".zip", ""),
        format="zip",
        root_dir=OUTPUT_FOLDER
    )

    return FileResponse(
        result_zip_path,
        filename="mriqc_results.zip",
        headers={"X-MRIQC-Resources": f"CPU:{n_procs},RAM:{mem_gb}GB"}
    )

#########################################
# WebSocket Endpoint with Enhanced Resources
#########################################

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def run_mriqc_process_ws(bids_dir: str, participant_label: str = "01"):
    """Enhanced WebSocket processor with dynamic resources"""
    cmd = [
        "docker", "run", "--rm",
        "--memory=48g", "--memory-swap=48g",
        "--cpus=12",
        "-v", f"{bids_dir}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",
        "/data", "/out",
        "participant",
        "--participant_label", participant_label,
        "-m", "T1w", "T2w", "bold",
        "--nprocs", "12",
        "--omp-nthreads", "4",
        "--no-sub"
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.decode().strip())
    
    await process.wait()
    await manager.broadcast("MRIQC process completed.")

@app.websocket("/ws/mriqc")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        bids_dir = "/mnt/mriqc_upload/bids_data"  # Persistent location
        await run_mriqc_process_ws(bids_dir)
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.broadcast(f"Error: {str(e)}")
        manager.disconnect(websocket)

#########################################
# Helper Functions (unchanged)
#########################################

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    queue: Deque[Path] = deque()
    queue.append(upload_dir)
    while queue:
        current = queue.popleft()
        if (current / "dataset_description.json").exists():
            return current
        for child in current.iterdir():
            if child.is_dir():
                queue.append(child)
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
