from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import uuid
import subprocess

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"status": "server is running"}


@app.post("/api/v1/moments")
async def create_moment(
    video: UploadFile = File(...),
    captured_at: str = Form(...),
    duration: str = Form(...),
):
    moment_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{moment_id}.mp4"

    with open(file_path, "wb") as f:
        content = await video.read()
        f.write(content)
    
    cmd = [
        "python", "prepare_colmap_windows.py",
        file_path, "output",
        "--fps", "20",
        "--overwrite",
        "--export-ply",
    ]

    subprocess.run(cmd, check=True)

    return {
        "id": moment_id,
        "file_path": str(file_path),
        "captured_at": captured_at,
        "duration_seconds": float(duration),
        "size_bytes": file_path.stat().st_size,
    }