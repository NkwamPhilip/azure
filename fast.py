from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import uvicorn
import subprocess

app = FastAPI()

# Simple connection manager to handle multiple websocket connections.
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

async def run_mriqc_process():
    # Replace the following command with your actual MRIQC Docker command.
    # Adjust paths and parameters as needed.
    cmd = [
        "docker", "run", "--rm",
        "-v", "/path/to/bids:/data:ro",      # <-- Update with your BIDS folder path on the VM
        "-v", "/path/to/output:/out",         # <-- Update with your desired output path
        "nipreps/mriqc:22.0.6",                # MRIQC image; change version if needed
        "/data", "/out",
        "participant",
        "--participant_label", "01",
        "-m", "T1w", "T2w", "bold"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    # Read process output line by line
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        message = line.decode('utf-8').strip()
        await manager.broadcast(message)
    await process.wait()
    await manager.broadcast("MRIQC process completed.")

@app.websocket("/ws/mriqc")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Start the MRIQC process when the first client connects.
        # (In a production system, you might trigger this separately.)
        await run_mriqc_process()
        while True:
            # Keep the connection open (this loop can be used to receive messages if needed)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
