from flask import Flask, request, send_file, jsonify
import subprocess
import os
import shutil
import zipfile
from pathlib import Path

app = Flask(__name__)

# Temporary folders for processing
UPLOAD_FOLDER = "/tmp/mriqc_upload"
OUTPUT_FOLDER = "/tmp/mriqc_output"

# Ensure they're present
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/run-mriqc", methods=["POST"])
def run_mriqc():
    """
    Expects a POST with:
      - 'bids_zip': The BIDS dataset as a ZIP file
      - 'participant_label': The subject ID (optional, defaults to '01')

    Returns a ZIP of MRIQC derivatives.
    """
    # 1) Parse incoming form data
    bids_zip = request.files.get("bids_zip")
    subj_id = request.form.get("participant_label", "01")

    if not bids_zip:
        return jsonify({"error": "No BIDS zip provided"}), 400

    # Clean up old data
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2) Save the uploaded zip
    zip_path = os.path.join(UPLOAD_FOLDER, "bids_data.zip")
    bids_zip.save(zip_path)

    # 3) Unzip BIDS data
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(UPLOAD_FOLDER)

    # We'll assume the unzipped BIDS root is e.g. /tmp/mriqc_upload/bids_output
    bids_root = Path(UPLOAD_FOLDER) / "bids_output"
    if not bids_root.exists():
        return jsonify({"error": "BIDS root not found"}), 400

    # 4) Run MRIQC
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{bids_root.absolute()}:/data:ro",
        "-v", f"{Path(OUTPUT_FOLDER).absolute()}:/out",
        "nipreps/mriqc:22.0.6",    # The Docker image
        "/data", "/out",
        "participant",
        "--participant_label", subj_id,
        "-m", "T1w", "T2w", "bold"  # Types of data to process
    ]

    run_result = subprocess.run(cmd, capture_output=True, text=True)
    if run_result.returncode != 0:
        return jsonify({
            "error": "MRIQC failed",
            "stderr": run_result.stderr
        }), 500

    # 5) Zip the OUTPUT_FOLDER to return
    result_zip_path = "/tmp/mriqc_results.zip"
    shutil.make_archive(
        base_name=result_zip_path.replace(".zip", ""),
        format="zip",
        root_dir=OUTPUT_FOLDER
    )

    # 6) Send results as a file
    return send_file(result_zip_path, as_attachment=True)


if __name__ == "__main__":
    # Start the server on port 8000
    app.run(host="0.0.0.0", port=8000)
