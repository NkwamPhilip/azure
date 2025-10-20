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

# -----------------------------------
# Logging configuration
# -----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - MRIQC-Backend - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MRIQC-Backend")

# -----------------------------------
# FastAPI app
# -----------------------------------
app = FastAPI(title="MRIQC Backend", version="1.0")

UPLOAD_FOLDER = Path("/tmp/mriqc_upload")
OUTPUT_FOLDER = Path("/tmp/mriqc_output")
RESULT_ZIP = Path("/tmp/mriqc_results.zip")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health_check():
    return {"status": "ready", "message": "MRIQC backend with Docker-in-Docker"}


# -----------------------------------
# Main MRIQC endpoint
# -----------------------------------
@app.post("/run-mriqc")
async def run_mriqc(
    bids_zip: UploadFile = File(...),
    participant_label: str = Form("01"),
    modalities: str = Form("T1w"),
    session_id: str = Form("baseline"),
    n_procs: str = Form("4"),
    mem_gb: str = Form("16")
):
    """
    Run MRIQC inside Docker-in-Docker and return ZIP results.
    """
    try:
        participant_label = participant_label.strip()
        if not participant_label.isalnum():
            raise HTTPException(status_code=400, detail="Participant label must be alphanumeric")

        # Clean previous runs
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        UPLOAD_FOLDER.mkdir(exist_ok=True)
        OUTPUT_FOLDER.mkdir(exist_ok=True)

        # Save uploaded ZIP
        zip_path = UPLOAD_FOLDER / "bids_dataset.zip"
        content = await bids_zip.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(zip_path, "wb") as f:
            f.write(content)

        logger.info(f"[{participant_label}] Received file: {bids_zip.filename} ({len(content)} bytes)")

        # Extract uploaded BIDS dataset
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(UPLOAD_FOLDER)
                logger.info(f"[{participant_label}] ZIP extracted successfully.")
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file format")

        # Locate BIDS root
        bids_root = find_bids_root(UPLOAD_FOLDER)
        if not bids_root:
            debug_structure(str(UPLOAD_FOLDER))
            raise HTTPException(status_code=400, detail="No valid BIDS dataset found (missing dataset_description.json)")

        logger.info(f"[{participant_label}] BIDS root: {bids_root}")
        debug_structure(str(bids_root))

        # MRIQC command
        modality_list = [m.strip() for m in modalities.split()]
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

        if session_id.strip():
            session_id_clean = session_id.strip()
            cmd += ["--session-id", session_id_clean]
            logger.info(f"[{participant_label}] Using session ID: {session_id_clean}")

        # Ensure image exists
        pull_cmd = ["docker", "pull", "nipreps/mriqc:22.0.6"]
        subprocess.run(pull_cmd, capture_output=True, text=True)

        logger.info(f"[{participant_label}] Running Docker command: {' '.join(cmd)}")

        # Run MRIQC
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )
        except subprocess.TimeoutExpired:
            logger.error(f"[{participant_label}] ❌ MRIQC timed out after 2 hours.")
            raise HTTPException(status_code=500, detail="MRIQC processing timed out after 2 hours")

        # Log outputs
        log_path = OUTPUT_FOLDER / "mriqc_log.txt"
        with open(log_path, "w") as log_file:
            log_file.write("=== MRIQC Command ===\n")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.write("=== STDOUT ===\n" + result.stdout + "\n")
            log_file.write("=== STDERR ===\n" + result.stderr + "\n")

        if result.returncode != 0:
            logger.error(f"[{participant_label}] MRIQC failed with exit code {result.returncode}")
            if "got an empty result" in result.stderr:
                debug_mriqc_view(bids_root, participant_label, session_id)
                raise HTTPException(status_code=500, detail="MRIQC cannot find expected BIDS files.")
            else:
                raise HTTPException(status_code=500, detail=result.stderr[-1000:])

        logger.info(f"[{participant_label}] ✅ MRIQC completed successfully.")

        # Ensure output exists
        result_files = list(OUTPUT_FOLDER.rglob("*"))
        if not result_files:
            raise HTTPException(status_code=500, detail="MRIQC completed but no output files found.")

        # Zip results
        if RESULT_ZIP.exists():
            RESULT_ZIP.unlink()

        shutil.make_archive(
            base_name=str(RESULT_ZIP).replace(".zip", ""),
            format="zip",
            root_dir=OUTPUT_FOLDER
        )

        if not RESULT_ZIP.exists():
            raise HTTPException(status_code=500, detail="Failed to create results ZIP.")

        size_mb = RESULT_ZIP.stat().st_size / (1024 * 1024)
        logger.info(f"[{participant_label}] Result ZIP created ({size_mb:.2f} MB)")

        # Return result ZIP
        return FileResponse(
            RESULT_ZIP,
            filename=f"mriqc_results_{participant_label}.zip",
            media_type="application/zip"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{participant_label}] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# -----------------------------------
# Helper Functions
# -----------------------------------

def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """
    Traverse extracted folder to find dataset_description.json
    """
    queue = deque([upload_dir])
    while queue:
        current = queue.popleft()
        if (current / "dataset_description.json").exists():
            return current
        for sub in current.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                queue.append(sub)
    return None


def debug_structure(root_path: str):
    logger.info("=== DEBUG: File Structure ===")
    for root, dirs, files in os.walk(root_path):
        level = root.replace(root_path, "").count(os.sep)
        indent = " " * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")


def debug_mriqc_view(bids_root: Path, participant_label: str, session_id: str):
    logger.info("=== DEBUG: MRIQC Expected BIDS View ===")
    subj_dir = bids_root / f"sub-{participant_label}"
    if subj_dir.exists():
        logger.info(f"Found subject directory: {subj_dir}")
        ses_dir = subj_dir / f"ses-{session_id}" if session_id else subj_dir
        if ses_dir.exists():
            for modality in ["anat", "func", "dwi"]:
                mdir = ses_dir / modality
                if mdir.exists():
                    nfiles = len(list(mdir.glob("*.nii*")))
                    logger.info(f"{mdir}: {nfiles} NIfTI files")
                else:
                    logger.warning(f"Missing: {mdir}")
        else:
            logger.warning(f"Missing session directory: {ses_dir}")
    else:
        logger.warning(f"Missing subject directory: {subj_dir}")


# -----------------------------------
# Run with Uvicorn
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Fast:app", host="0.0.0.0", port=8000, workers=1)
