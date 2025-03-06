import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import subprocess
import os
import shutil
import zipfile
import uvicorn

app = FastAPI()

# Directories for processing
UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
    It saves and unzips the file, finds the BIDS root,
    runs MRIQC in a Docker container (nipreps/mriqc:22.0.6),
    writes logs to a file, zips the output folder, and returns the ZIP.
    """
    try:
        contents = await bids_zip.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    
    # Clean up previous data
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
        raise HTTPException(status_code=400, detail="Failed to unzip file")
    
    # Find first directory as BIDS root
    bids_root = None
    for item in Path(UPLOAD_FOLDER).iterdir():
        if item.is_dir():
            bids_root = item
            break
    if bids_root is None:
        return JSONResponse(content={"error": "No BIDS directory found after unzipping."}, status_code=400)
    
    # Build MRIQC Docker command
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{bids_root.absolute()}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",
        "/data", "/out",
        "participant",
        "--participant_label", participant_label,
        "-m", "T1w", "T2w", "bold"
    ]
    
    # Run MRIQC synchronously
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Write logs to file
    log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("=== MRIQC Docker Run Logs ===\n")
        log_file.write(result.stdout)
        log_file.write("\n=== Errors ===\n")
        log_file.write(result.stderr)
    
    if result.returncode != 0:
        return JSONResponse(content={"error": "MRIQC failed", "stderr": result.stderr}, status_code=500)
    
    # Zip the OUTPUT_FOLDER
    result_zip_path = "/tmp/mriqc_results.zip"
    shutil.make_archive(base_name=result_zip_path.replace(".zip", ""), format="zip", root_dir=OUTPUT_FOLDER)
    
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
    Runs the MRIQC Docker command and streams its output line by line over the WebSocket.
    Adjust the volumes and parameters as needed.
    """
    # Example: assume BIDS data is already in /tmp/mriqc_upload/bids_data
    bids_dir = "/tmp/mriqc_upload/bids_data"  # Update as necessary
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{bids_dir}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",
        "/data", "/out",
        "participant",
        "--participant_label", "01",  # Hard-coded for example; parameterize as needed.
        "-m", "T1w", "T2w", "bold"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        encoding="utf-8",
        bufsize=1
    )
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await manager.broadcast(line.strip())
    await process.wait()
    await manager.broadcast("MRIQC process completed.")

@app.websocket("/ws/mriqc")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await run_mriqc_process_ws()
        while True:
            # Keep the connection alive (receive messages if needed)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

#########################################
# Run the Uvicorn Server
#########################################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
