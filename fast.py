import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import shutil
import zipfile
from typing import Optional
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRIQC-Backend")

app = FastAPI()

# Configure directories
UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
# Direct MRIQC Execution Endpoint
#########################################

@app.post("/run-mriqc")
async def run_mriqc(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    session_id: str = Form("baseline"),
    n_procs: str = Form("16"),
    mem_gb: str = Form("64")
):
    """Run MRIQC directly and return results when complete"""
    
    try:
        # Validate inputs
        participant_label = participant_label.strip()
        if not participant_label.isalnum():
            raise HTTPException(status_code=400, detail="Participant label must be alphanumeric")

        # Clean working directories
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        logger.info(f"[{participant_label}] Received file: {bids_zip.filename}")

        # Save uploaded zip
        zip_path = Path(UPLOAD_FOLDER) / "bids_data.zip"
        content = await bids_zip.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info(f"[{participant_label}] Saving {len(content)} bytes")
        
        with open(zip_path, "wb") as f:
            f.write(content)

        # Extract BIDS data
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                logger.info(f"[{participant_label}] ZIP contains {len(file_list)} files")
                zf.extractall(UPLOAD_FOLDER)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file format")

        # Find BIDS root directory
        bids_root = find_bids_root(Path(UPLOAD_FOLDER))
        if not bids_root:
            raise HTTPException(
                status_code=400,
                detail="No valid BIDS dataset found (missing dataset_description.json)"
            )

        logger.info(f"[{participant_label}] BIDS root: {bids_root}")

        # Parse modalities
        modality_list = modalities.split()
        logger.info(f"[{participant_label}] Modalities: {modality_list}")

        # Build Docker command
        cmd = [
            "docker", "run", "--rm",
            f"--memory={mem_gb}g",
            f"--cpus={n_procs}",
            "-v", f"{bids_root}:/data",
            "-v", f"{OUTPUT_FOLDER}:/out",
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", participant_label,
            "-m", *modality_list,
            "--nprocs", n_procs,
            "--no-sub",
            "--verbose-reports"
        ]
        
        if session_id:
            cmd += ["--session-id", session_id]

        logger.info(f"[{participant_label}] Running: {' '.join(cmd)}")

        # Execute MRIQC synchronously (blocks until complete)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[{participant_label}] Timed out after 2 hours")
            raise HTTPException(status_code=500, detail="Processing timed out after 2 hours")

        # Save logs
        log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
        with open(log_path, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.write("=== STDOUT ===\n")
            log_file.write(result.stdout)
            log_file.write("\n=== STDERR ===\n")
            log_file.write(result.stderr)

        if result.returncode != 0:
            logger.error(f"[{participant_label}] Failed with code {result.returncode}")
            logger.error(f"[{participant_label}] STDERR: {result.stderr[-1000:]}")
            raise HTTPException(
                status_code=500,
                detail=f"MRIQC failed with code {result.returncode}: {result.stderr[-500:]}"
            )

        logger.info(f"[{participant_label}] ✅ MRIQC completed successfully")

        # Package results
        result_zip_path = "/tmp/mriqc_results.zip"
        shutil.make_archive(
            base_name=result_zip_path.replace(".zip", ""),
            format="zip",
            root_dir=OUTPUT_FOLDER
        )

        if not Path(result_zip_path).exists():
            raise HTTPException(status_code=500, detail="Failed to package results")

        logger.info(f"[{participant_label}] Returning results ZIP")

        return FileResponse(
            result_zip_path,
            filename=f"mriqc_results_{participant_label}.zip",
            media_type="application/zip",
            headers={
                "X-MRIQC-Status": "complete",
                "X-MRIQC-Participant": participant_label
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{participant_label}] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

#########################################
# Cleanup Endpoint
#########################################

@app.post("/cleanup")
async def cleanup():
    """Clean up temporary files after processing"""
    try:
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        result_zip = Path("/tmp/mriqc_results.zip")
        if result_zip.exists():
            result_zip.unlink()
        
        logger.info("✅ Cleanup completed")
        return {"status": "cleaned"}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#########################################
# Helper Functions
#########################################

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """Find BIDS root by looking for dataset_description.json"""
    queue = deque([upload_dir])
    visited = set()
    
    while queue:
        current = queue.popleft()
        
        if current in visited:
            continue
        visited.add(current)
        
        if (current / "dataset_description.json").exists():
            return current
        
        try:
            for child in current.iterdir():
                if child.is_dir() and not child.name.startswith('.'):
                    queue.append(child)
        except:
            continue
    
    return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
