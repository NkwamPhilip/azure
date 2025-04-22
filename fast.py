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

# Directories for processing
UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#########################################
# Helper function: find BIDS root by looking for dataset_description.json
#########################################

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """
    Performs a breadth-first search in upload_dir for a directory
    that contains dataset_description.json at its top level.
    Returns that directory if found, otherwise None.
    """
    queue: Deque[Path] = deque()
    queue.append(upload_dir)

    while queue:
        current = queue.popleft()
        # Check if current directory has dataset_description.json
        ds_file = current / "dataset_description.json"
        if ds_file.exists():
            return current

        # Otherwise, enqueue subdirectories
        for child in current.iterdir():
            if child.is_dir():
                queue.append(child)

    return None

#########################################
# POST Endpoint: Run MRIQC via Docker
#########################################

@app.post("/run-mriqc")
async def run_mriqc_endpoint(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01")
):
    """
    Accepts a BIDS ZIP file and a participant label.
    Unzips into /tmp/mriqc_upload, scans for the directory containing dataset_description.json,
    then runs MRIQC in Docker with memory=15g, concurrency flags, etc.
    """

    # Step 1: Read the uploaded file
    try:
        contents = await bids_zip.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    # Step 2: Clean old data
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 3: Save the ZIP to /tmp/mriqc_upload
    zip_path = Path(UPLOAD_FOLDER) / "bids_data.zip"
    with open(zip_path, "wb") as f:
        f.write(contents)

    # Step 4: Unzip
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(UPLOAD_FOLDER)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to unzip file: {str(e)}")

    # Debug: List everything in /tmp/mriqc_upload
    print("DEBUG: After unzipping, /tmp/mriqc_upload has:")
    for thing in Path(UPLOAD_FOLDER).iterdir():
        print("  ", thing, "(dir)" if thing.is_dir() else "(file)")

    # Step 5: Find the BIDS root by searching for dataset_description.json
    bids_root = find_bids_root(Path(UPLOAD_FOLDER))
    if not bids_root:
        return JSONResponse(
            content={"error": "No directory with dataset_description.json found."},
            status_code=400
        )

    print(f"DEBUG: Found BIDS root at: {bids_root}")

    # Step 6: Build the Docker command for MRIQC
    cmd = [
        "docker", "run", "--rm",
        "--memory=15g", "--memory-swap=15g",
        "-v", f"{bids_root.absolute()}:/data:ro",
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
    print("DEBUG: MRIQC Docker command:")
    print("  " + " ".join(cmd))

    # Step 7: Run
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Step 8: Write logs
    log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("=== MRIQC Docker Run Logs ===\n")
        log_file.write(result.stdout)
        log_file.write("\n=== Errors ===\n")
        log_file.write(result.stderr)

    if result.returncode != 0:
        return JSONResponse(
            content={"error": "MRIQC failed", "stderr": result.stderr},
            status_code=500
        )

    # Step 9: Zip the OUTPUT_FOLDER
    result_zip_path = "/tmp/mriqc_results.zip"
    shutil.make_archive(
        base_name=result_zip_path.replace(".zip", ""),
        format="zip",
        root_dir=OUTPUT_FOLDER
    )

    return FileResponse(result_zip_path, filename="mriqc_results.zip")


#########################################
# WebSocket Endpoint: Real-Time Logs
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

async def run_mriqc_process_ws():
    """
    Runs the MRIQC Docker command with memory=15g, concurrency, etc.
    Streams output line by line over the WebSocket.
    """
    # This is still hard-coded to /tmp/mriqc_upload/bids_data
    # If you want it to be dynamic, do something similar as above (search for dataset_description.json).
    bids_dir = "/tmp/mriqc_upload/bids_data"
    cmd = [
        "docker", "run", "--rm",
        "--memory=15g", "--memory-swap=15g",
        "-v", f"{bids_dir}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",
        "/data", "/out",
        "participant",
        "--participant_label", "01",
        "-m", "T1w", "T2w", "bold",
        "--nprocs", "8",
        "--omp-nthreads", "1",
        "--no-sub"
    ]
    print("DEBUG: WebSocket MRIQC Docker command:")
    print("  " + " ".join(cmd))

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
        await run_mriqc_process_ws()
        while True:
            # keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

#########################################
# Run the Uvicorn Server
#########################################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
