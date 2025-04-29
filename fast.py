import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import subprocess
import os
import shutil
import zipfile
import uvicorn
from typing import Optional, Deque, List
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRIQC-Backend")

app = FastAPI()

# Configure directories (using persistent storage)
UPLOAD_FOLDER = "/mnt/mriqc_upload"
OUTPUT_FOLDER = "/mnt/mriqc_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Health check endpoint
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
# Enhanced MRIQC Endpoint
#########################################

@app.post("/run-mriqc")
async def run_mriqc_endpoint(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    n_procs: str = Form("12"),
    mem_gb: str = Form("48")
):
    """Process BIDS data with MRIQC using frontend-provided parameters"""
    
    try:
        # Validate inputs
        participant_label = participant_label.strip()
        if not participant_label.isalnum():
            raise HTTPException(
                status_code=400,
                detail="Participant label must be alphanumeric"
            )

        try:
            n_procs_int = int(n_procs)
            mem_gb_int = int(mem_gb)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="n_procs and mem_gb must be integers"
            )

        # Validate modalities
        valid_modalities = {"T1w", "T2w", "bold", "dwi", "flair", "asl"}
        input_modalities = set(mod.strip() for mod in modalities.split())
        
        if not input_modalities:
            raise HTTPException(
                status_code=400,
                detail="At least one modality must be specified"
            )
            
        invalid_mods = input_modalities - valid_modalities
        if invalid_mods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid modalities: {', '.join(invalid_mods)}. Valid options: {', '.join(valid_modalities)}"
            )

        # Clean working directories
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save uploaded zip
        zip_path = Path(UPLOAD_FOLDER) / "bids_data.zip"
        try:
            with open(zip_path, "wb") as f:
                while contents := await bids_zip.read(1024 * 1024):  # 1MB chunks
                    f.write(contents)
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )

        # Extract BIDS data
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(UPLOAD_FOLDER)
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=400,
                detail="Invalid ZIP file format"
            )

        # Find BIDS root directory
        bids_root = find_bids_root(Path(UPLOAD_FOLDER))
        if not bids_root:
            raise HTTPException(
                status_code=400,
                detail="No valid BIDS dataset found (missing dataset_description.json)"
            )

        # Build Docker command with dynamic resources
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
            "-m", *input_modalities,
            "--nprocs", n_procs,
            "--omp-nthreads", str(max(1, int(n_procs)//4)),  # Auto-scale threads
            "--no-sub",
            "--verbose-reports"
        ]

        logger.info(f"Running MRIQC with command: {' '.join(cmd)}")

        # Execute MRIQC
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
        except subprocess.TimeoutExpired:
            logger.error("MRIQC processing timed out after 2 hours")
            raise HTTPException(
                status_code=500,
                detail="Processing timed out after 2 hours"
            )

        # Save logs
        log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
        with open(log_path, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.write("=== STDOUT ===\n")
            log_file.write(result.stdout)
            log_file.write("\n=== STDERR ===\n")
            log_file.write(result.stderr)

        if result.returncode != 0:
            logger.error(f"MRIQC failed with return code {result.returncode}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "MRIQC processing failed",
                    "return_code": result.returncode,
                    "stderr": result.stderr[-1000:]  # Last 1000 chars of stderr
                }
            )

        # Package results
        result_zip_path = "/mnt/mriqc_results.zip"
        shutil.make_archive(
            base_name=result_zip_path.replace(".zip", ""),
            format="zip",
            root_dir=OUTPUT_FOLDER
        )

        # Verify results were generated
        if not Path(result_zip_path).exists():
            logger.error("Result ZIP file was not created")
            raise HTTPException(
                status_code=500,
                detail="Result packaging failed"
            )

        return FileResponse(
            result_zip_path,
            filename=f"mriqc_results_{participant_label}.zip",
            media_type="application/zip",
            headers={
                "X-MRIQC-Status": "complete",
                "X-MRIQC-Modalities": ",".join(input_modalities)
            }
        )

    except HTTPException:
        raise  # Re-raise our known HTTP exceptions
    except Exception as e:
        logger.exception("Unexpected error in MRIQC processing")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

#########################################
# WebSocket Endpoint (Optional)
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
            # Can implement real-time updates here if needed
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
