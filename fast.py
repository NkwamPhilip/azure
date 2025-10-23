import os
import json
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path
from typing import Optional
from collections import deque

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MRIQC-Backend")

# ----------------------------------
# App & Paths
# ----------------------------------
app = FastAPI()

UPLOAD_FOLDER = Path("/tmp/mriqc_upload")
OUTPUT_FOLDER = Path("/tmp/mriqc_output")
RESULT_ZIP = Path("/tmp/mriqc_results.zip")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# For deciding when to stream vs sendfile
LARGE_FILE_THRESHOLD_MB = 150  # tune if desired


# ----------------------------------
# Utilities
# ----------------------------------
def file_iterator(path: Path, chunk_size: int = 1024 * 1024):
    """Yield file in chunks for StreamingResponse."""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def find_bids_root(upload_dir: Path) -> Optional[Path]:
    """Breadth-first search for a directory with dataset_description.json."""
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
                if child.is_dir() and not child.name.startswith("."):
                    queue.append(child)
        except Exception:
            continue
    return None


def debug_structure(root_path: Path):
    logger.info("=== DEBUG: File Structure ===")
    try:
        for root, dirs, files in os.walk(root_path):
            rel = os.path.relpath(root, root_path)
            rel = "." if rel == "." else rel
            logger.info(f"{rel}/")
            for f in files:
                logger.info(f"  {f}")
    except Exception as e:
        logger.error(f"Debug structure failed: {e}")


def debug_mriqc_view(bids_root: Path, participant_label: str, session_id: str):
    logger.info("=== DEBUG: MRIQC Expected View ===")
    participant_dir = bids_root / f"sub-{participant_label}"

    if participant_dir.exists():
        logger.info(f"✅ Found participant directory: {participant_dir}")
        if session_id:
            session_dir = participant_dir / f"ses-{session_id}"
            if session_dir.exists():
                logger.info(f"✅ Found session directory: {session_dir}")
                for modality in ["anat", "func", "dwi"]:
                    p = session_dir / modality
                    if p.exists():
                        nifti = list(p.glob("*.nii*"))
                        logger.info(f"✅ {p}: {len(nifti)} NIfTI files")
                    else:
                        logger.warning(f"❌ Missing modality directory: {p}")
            else:
                logger.warning(f"❌ Missing session directory: {session_dir}")
                actual_sessions = [
                    d.name for d in participant_dir.iterdir()
                    if d.is_dir() and d.name.startswith("ses-")
                ]
                logger.info(f"Actual sessions: {actual_sessions}")
        else:
            for modality in ["anat", "func", "dwi"]:
                p = participant_dir / modality
                if p.exists():
                    nifti = list(p.glob("*.nii*"))
                    logger.info(f"✅ {p}: {len(nifti)} NIfTI files")
                else:
                    logger.warning(f"❌ Missing modality directory: {p}")
    else:
        logger.error(f"❌ Missing participant directory: {participant_dir}")
        actual_participants = [
            d.name for d in bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ]
        logger.info(f"Actual participants: {actual_participants}")


# ----------------------------------
# Endpoints
# ----------------------------------
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
    mem_gb: str = Form("16"),
):
    """
    Receive a BIDS ZIP, run MRIQC inside nipreps/mriqc container (Docker-in-Docker),
    pack results into /tmp/mriqc_results.zip and return them.
    Uses sendfile for small/medium zips and chunked streaming for large ones.
    """
    try:
        participant_label = participant_label.strip()
        if not participant_label or not participant_label.isalnum():
            raise HTTPException(status_code=400, detail="Participant label must be alphanumeric")

        # Fresh workspace
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        if OUTPUT_FOLDER.exists():
            shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)

        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{participant_label}] Received file: {bids_zip.filename}")
        content = await bids_zip.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        zip_path = UPLOAD_FOLDER / "bids_dataset.zip"
        logger.info(f"[{participant_label}] Saving {len(content)} bytes")
        with open(zip_path, "wb") as f:
            f.write(content)

        # Extract uploaded zip
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                logger.info(f"[{participant_label}] ZIP contains {len(zf.namelist())} files")
                zf.extractall(UPLOAD_FOLDER)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file format")

        # Identify BIDS root
        bids_root = find_bids_root(UPLOAD_FOLDER)
        if not bids_root:
            debug_structure(UPLOAD_FOLDER)
            raise HTTPException(status_code=400, detail="No valid BIDS dataset found (missing dataset_description.json)")

        logger.info(f"[{participant_label}] BIDS root: {bids_root}")
        debug_structure(bids_root)

        # Parse modalities
        modality_list = [m.strip() for m in modalities.split() if m.strip()]
        if not modality_list:
            modality_list = ["T1w"]
        logger.info(f"[{participant_label}] Modalities: {modality_list}")

        # Build MRIQC command
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
            "--verbose-reports",
        ]
        if session_id and session_id.strip():
            # Keep user's casing to match BIDS folder, which is case-sensitive
            sid = session_id.strip()
            cmd += ["--session-id", sid]
            logger.info(f"[{participant_label}] Using session ID: {sid}")

        logger.info(f"[{participant_label}] Running Docker command: {' '.join(cmd)}")

        # Optional: ensure image present (ignore errors if offline)
        try:
            pull_cmd = ["docker", "pull", "nipreps/mriqc:22.0.6"]
            subprocess.run(pull_cmd, capture_output=True, text=True, timeout=1800)
        except Exception:
            pass

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired:
            logger.error(f"[{participant_label}] Timed out after 2 hours")
            raise HTTPException(status_code=500, detail="Processing timed out after 2 hours")

        # Persist logs for debugging
        log_path = OUTPUT_FOLDER / "mriqc_log.txt"
        with open(log_path, "w") as lf:
            lf.write(f"Command: {' '.join(cmd)}\n\n")
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout or "")
            lf.write("\n=== STDERR ===\n")
            lf.write(result.stderr or "")

        if result.returncode != 0:
            logger.error(f"[{participant_label}] Failed with code {result.returncode}")
            tail = (result.stderr or "")[-2000:]
            if "got an empty result" in (result.stderr or ""):
                debug_mriqc_view(bids_root, participant_label, session_id)
                msg = f"MRIQC cannot find files. Expected structure: /data/sub-{participant_label}/[ses-*/]anat/sub-{participant_label}[_ses-*]_T1w.nii.gz"
            else:
                msg = f"MRIQC failed: {tail}"
            raise HTTPException(status_code=500, detail=msg)

        logger.info(f"[{participant_label}] ✅ MRIQC completed successfully.")

        # Verify output exists
        result_files = list(OUTPUT_FOLDER.rglob("*"))
        if not result_files:
            raise HTTPException(status_code=500, detail="MRIQC completed but no output files were generated")

        # Prepare ZIP
        if RESULT_ZIP.exists():
            RESULT_ZIP.unlink()
        shutil.make_archive(
            base_name=RESULT_ZIP.with_suffix("").as_posix(),
            format="zip",
            root_dir=OUTPUT_FOLDER.as_posix()
        )
        if not RESULT_ZIP.exists() or RESULT_ZIP.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to package results")

        size_mb = RESULT_ZIP.stat().st_size / (1024 * 1024)
        logger.info(f"[{participant_label}] Result ZIP created ({size_mb:.2f} MB)")

        # Decide how to return (sendfile for small, streaming for large)
        headers = {
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Content-Disposition": f'attachment; filename="mriqc_results_{participant_label}.zip"',
            "Connection": "keep-alive",
        }

        if size_mb <= LARGE_FILE_THRESHOLD_MB:
            return FileResponse(
                RESULT_ZIP,
                filename=f"mriqc_results_{participant_label}.zip",
                media_type="application/zip",
                headers=headers
            )
        else:
            headers["Content-Length"] = str(RESULT_ZIP.stat().st_size)
            return StreamingResponse(
                file_iterator(RESULT_ZIP),
                media_type="application/zip",
                headers=headers
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{participant_label}] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/results/latest")
def results_latest():
    """Direct link to the last produced ZIP (browser-managed download)."""
    if not RESULT_ZIP.exists() or RESULT_ZIP.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="No results available")
    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "Content-Disposition": f'attachment; filename="{RESULT_ZIP.name}"',
        "Connection": "keep-alive",
        "Content-Length": str(RESULT_ZIP.stat().st_size),
    }
    return FileResponse(RESULT_ZIP, media_type="application/zip", headers=headers)


@app.post("/cleanup")
def cleanup():
    """Optional: clear work dirs & last ZIP."""
    try:
        if UPLOAD_FOLDER.exists():
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        if OUTPUT_FOLDER.exists():
            shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        if RESULT_ZIP.exists():
            RESULT_ZIP.unlink()
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Keep connections alive for long downloads
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=300)
