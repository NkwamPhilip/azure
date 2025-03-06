#!/usr/bin/env python3
"""
azure_mriqc_server.py

This Flask API receives a POST request with:
  - 'bids_zip': A zip file containing a BIDS dataset.
  - 'participant_label': (optional) subject ID (default: '01').

It performs the following steps:
  1. Saves and unzips the uploaded BIDS ZIP to /tmp/mriqc_upload.
  2. Dynamically finds the first directory (assumed to be the BIDS root).
  3. Runs MRIQC inside a Docker container (using nipreps/mriqc:22.0.6).
  4. Captures Docker stdout and stderr and writes them to mriqc_log.txt in /tmp/mriqc_output.
  5. Zips the OUTPUT_FOLDER (which contains MRIQC outputs and the log) and returns it.
"""

from flask import Flask, request, send_file, jsonify
import subprocess
import os
import shutil
import zipfile
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/run-mriqc", methods=["POST"])
def run_mriqc():
    # Parse incoming form data
    bids_zip = request.files.get("bids_zip")
    subj_id = request.form.get("participant_label", "01")

    if not bids_zip:
        return jsonify({"error": "No BIDS zip provided"}), 400

    # Clean up old data
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save the uploaded zip
    zip_path = os.path.join(UPLOAD_FOLDER, "bids_data.zip")
    bids_zip.save(zip_path)

    # Unzip BIDS data
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(UPLOAD_FOLDER)

    # Dynamically find the first directory in UPLOAD_FOLDER as the BIDS root
    bids_root = None
    for item in Path(UPLOAD_FOLDER).iterdir():
        if item.is_dir():
            bids_root = item
            break

    if not bids_root:
        return jsonify({"error": "No BIDS directory found after unzipping."}), 400

    # Run MRIQC in Docker and capture logs
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{bids_root.absolute()}:/data:ro",  # mount BIDS as read-only
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",  # mount output folder
        "nipreps/mriqc:22.0.6",  # use MRIQC image; change version if needed
        "/data", "/out",
        "participant",
        "--participant_label", subj_id,
        "-m", "T1w", "T2w", "bold"
    ]

    run_result = subprocess.run(cmd, capture_output=True, text=True)

    # Write logs to a file
    log_path = Path(OUTPUT_FOLDER) / "mriqc_log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("=== MRIQC Docker Run Logs ===\n")
        log_file.write(run_result.stdout)
        log_file.write("\n=== Errors (if any) ===\n")
        log_file.write(run_result.stderr)

    if run_result.returncode != 0:
        return jsonify({
            "error": "MRIQC failed",
            "stderr": run_result.stderr
        }), 500

    # Zip the OUTPUT_FOLDER (which now includes MRIQC outputs and the log file)
    result_zip_path = "/tmp/mriqc_results.zip"
    shutil.make_archive(
        base_name=result_zip_path.replace(".zip", ""),
        format="zip",
        root_dir=OUTPUT_FOLDER
    )

    return send_file(result_zip_path, as_attachment=True)

if __name__ == "__main__":
    # Bind to 0.0.0.0 on port 8000 so it's externally accessible.
    app.run(host="0.0.0.0", port=8000)
