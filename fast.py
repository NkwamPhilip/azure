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
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRIQC-Backend")

app = FastAPI()

UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "ready", "message": "MRIQC backend with Docker-in-Docker"}

@app.post("/run-mriqc")
async def run_mriqc(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    session_id: str = Form(""),
    n_procs: str = Form("4"),
    mem_gb: str = Form("16")
):
    try:
        participant_label = participant_label.strip()
        if not participant_label.isalnum():
            raise HTTPException(status_code=400, detail="Participant label must be alphanumeric")

        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        logger.info(f"[{participant_label}] Received file: {bids_zip.filename}")

        zip_path = Path(UPLOAD_FOLDER) / "bids_data.zip"
        content = await bids_zip.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info(f"[{participant_label}] Saving {len(content)} bytes")
        
        with open(zip_path, "wb") as f:
            f.write(content)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = zf.namelist()
                logger.info(f"[{participant_label}] ZIP contains {len(file_list)} files")
                zf.extractall(UPLOAD_FOLDER)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file format")

        bids_root = find_bids_root(Path(UPLOAD_FOLDER))
        if not bids_root:
            debug_structure(UPLOAD_FOLDER)
            raise HTTPException(
                status_code=400,
                detail="No valid BIDS dataset found (missing dataset_description.json)"
            )

        logger.info(f"[{participant_label}] BIDS root: {bids_root}")
        debug_structure(str(bids_root))

        modality_list = [mod.strip() for mod in modalities.split()]
        logger.info(f"[{participant_label}] Modalities: {modality_list}")

        cmd = [
            "docker", "run", "--rm",
            f"--memory={mem_gb}g",
            f"--cpus={n_procs}",
            "-v", f"{bids_root}:/data:ro",
            "-v", f"{OUTPUT_FOLDER}:/out",
            "nipreps/mriqc:22.0.6",
            "/data", "/out", "participant",
            "--participant_label", participant_label,
            "-m", *modality_list,
            "--nprocs", n_procs,
            "--no-sub",
            "--verbose-reports"
        ]
        
        if session_id and session_id.strip():
            session_id_lower = session_id.strip().lower()
            cmd += ["--session-id", session_id_lower]
            logger.info(f"[{participant_label}] Using session ID: {session_id_lower}")

        logger.info(f"[{participant_label}] Running Docker command: {' '.join(cmd)}")

        pull_cmd = ["docker", "pull", "nipreps/mriqc:22.0.6"]
        logger.info(f"[{participant_label}] Pulling MRIQC image...")
        pull_result = subprocess.run(pull_cmd, capture_output=True, text=True)
        if pull_result.returncode != 0:
            logger.warning(f"[{participant_label}] Failed to pull image: {pull_result.stderr}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[{participant_label}] Timed out after 2 hours")
            raise HTTPException(status_code=500, detail="Processing timed out after 2 hours")

        log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
        with open(log_path, "w") as log_file:
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.write("=== STDOUT ===\n")
            log_file.write(result.stdout)
            log_file.write("\n=== STDERR ===\n")
            log_file.write(result.stderr)

        if result.returncode != 0:
            logger.error(f"[{participant_label}] Failed with code {result.returncode}")
            logger.error(f"[{participant_label}] STDERR: {result.stderr}")
            
            if "got an empty result" in result.stderr:
                debug_mriqc_view(bids_root, participant_label, session_id)
                error_msg = f"MRIQC cannot find files. Expected structure: /data/sub-{participant_label}/[ses-*/]anat/sub-{participant_label}[_ses-*]_T1w.nii.gz"
            else:
                error_msg = f"MRIQC failed: {result.stderr[-1000:]}"
                
            raise HTTPException(status_code=500, detail=error_msg)

        logger.info(f"[{participant_label}] ✅ MRIQC completed successfully")

        result_files = list(Path(OUTPUT_FOLDER).rglob("*"))
        if not result_files:
            raise HTTPException(status_code=500, detail="MRIQC completed but no output files were generated")

        logger.info(f"[{participant_label}] Found {len(result_files)} result files")

        result_zip_path = "/tmp/mriqc_results.zip"
        
        if Path(result_zip_path).exists():
            Path(result_zip_path).unlink()
        
        shutil.make_archive(
            base_name=result_zip_path.replace(".zip", ""),
            format="zip",
            root_dir=OUTPUT_FOLDER
        )

        if not Path(result_zip_path).exists():
            raise HTTPException(status_code=500, detail="Failed to package results")

        zip_size = Path(result_zip_path).stat().st_size
        logger.info(f"[{participant_label}] Result ZIP created: {zip_size / (1024*1024):.2f} MB")

        logger.info(f"[{participant_label}] Returning file: {result_zip_path}, exists: {Path(result_zip_path).exists()}")

        return FileResponse(
            result_zip_path,
            filename=f"mriqc_results_{participant_label}.zip",
            media_type="application/zip"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{participant_label}] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def find_bids_root(upload_dir: Path) -> Optional[Path]:
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

def debug_structure(root_path: str):
    logger.info("=== DEBUG: File Structure ===")
    try:
        for root, dirs, files in os.walk(root_path):
            level = root.replace(root_path, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                logger.info(f"{subindent}{file}")
    except Exception as e:
        logger.error(f"Debug structure failed: {e}")

def debug_mriqc_view(bids_root: Path, participant_label: str, session_id: str):
    logger.info("=== DEBUG: MRIQC Expected View ===")
    
    participant_dir = bids_root / f"sub-{participant_label}"
    if participant_dir.exists():
        logger.info(f"✅ Found participant directory: {participant_dir}")
        
        if session_id:
            session_dir = participant_dir / f"ses-{session_id.lower()}"
            if session_dir.exists():
                logger.info(f"✅ Found session directory: {session_dir}")
                for modality in ['anat', 'func', 'dwi']:
                    modality_dir = session_dir / modality
                    if modality_dir.exists():
                        nifti_files = list(modality_dir.glob("*.nii*"))
                        logger.info(f"✅ {modality_dir}: {len(nifti_files)} NIfTI files")
                    else:
                        logger.warning(f"❌ Missing modality directory: {modality_dir}")
            else:
                logger.warning(f"❌ Missing session directory: {session_dir}")
                actual_sessions = [d.name for d in participant_dir.iterdir() if d.is_dir() and d.name.startswith('ses-')]
                logger.info(f"Actual sessions: {actual_sessions}")
        else:
            for modality in ['anat', 'func', 'dwi']:
                modality_dir = participant_dir / modality
                if modality_dir.exists():
                    nifti_files = list(modality_dir.glob("*.nii*"))
                    logger.info(f"✅ {modality_dir}: {len(nifti_files)} NIfTI files")
                else:
                    logger.warning(f"❌ Missing modality directory: {modality_dir}")
    else:
        logger.error(f"❌ Missing participant directory: {participant_dir}")
        actual_participants = [d.name for d in bids_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        logger.info(f"Actual participants: {actual_participants}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
