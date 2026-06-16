from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .rendering_pipeline import prepare_fastgs_input_and_train
    from .unity_splat_transfer import (
        DEFAULT_UNITY_PROJECT,
        find_fastgs_point_cloud,
        transfer_fastgs_model_to_unity,
    )
except ImportError:
    from rendering_pipeline import prepare_fastgs_input_and_train
    from unity_splat_transfer import (
        DEFAULT_UNITY_PROJECT,
        find_fastgs_point_cloud,
        transfer_fastgs_model_to_unity,
    )

app = FastAPI(title="Reminiscence API")

cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "REMINISCENCE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
COLMAP_OUTPUT_ROOT = BACKEND_DIR / "output"
PREPARE_COLMAP_SCRIPT = PROJECT_ROOT / "prepare_colmap_windows.py"
UNITY_PROJECT_DIR = DEFAULT_UNITY_PROJECT
ASYNC_JOBS = os.environ.get("REMINISCENCE_ASYNC_JOBS", "").lower() in {"1", "true", "yes"}
UNITY_IMPORT = os.environ.get("REMINISCENCE_UNITY_IMPORT", str(os.name == "nt")).lower() in {
    "1",
    "true",
    "yes",
}
API_KEY = os.environ.get("REMINISCENCE_API_KEY", "")
TRAINING_ITERATIONS = int(os.environ.get("REMINISCENCE_TRAINING_ITERATIONS", "5000"))
JOBS: dict[str, dict] = {}
PIPELINE_LOCK = threading.Lock()

UPLOAD_DIR.mkdir(exist_ok=True)
COLMAP_OUTPUT_ROOT.mkdir(exist_ok=True)


def require_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def process_moment(moment_id: str, file_path: Path, captured_at: str, duration: str) -> dict:
    JOBS[moment_id] = {"id": moment_id, "status": "processing"}
    colmap_output_dir = COLMAP_OUTPUT_ROOT / moment_id

    try:
        with PIPELINE_LOCK:
            subprocess.run(
                [
                    sys.executable,
                    str(PREPARE_COLMAP_SCRIPT),
                    str(file_path),
                    str(colmap_output_dir),
                    "--fps",
                    "5",
                    "--overwrite",
                    "--export-ply",
                ],
                check=True,
            )
            pipeline_result = prepare_fastgs_input_and_train(
                colmap_output_dir=colmap_output_dir,
                fastgs_root=PROJECT_ROOT / "fastgs",
                run_training=True,
                training_iterations=TRAINING_ITERATIONS,
            )

            result = {
                "id": moment_id,
                "status": "complete",
                "captured_at": captured_at,
                "duration_seconds": float(duration),
                "size_bytes": file_path.stat().st_size,
                "dataset_name": pipeline_result.dataset_name,
                "dataset_path": pipeline_result.dataset_path,
                "model_path": pipeline_result.model_path,
                "render_path": pipeline_result.render_path,
                "registered_image_count": pipeline_result.registered_image_count,
                "splat_download_url": f"/api/v1/moments/{moment_id}/splat",
            }

            if UNITY_IMPORT:
                unity_result = transfer_fastgs_model_to_unity(
                    model_dir=Path(pipeline_result.model_path),
                    unity_project=UNITY_PROJECT_DIR,
                    convert=True,
                )
                result.update(
                    {
                        "unity_ply_path": unity_result.copied_ply,
                        "unity_asset_path": unity_result.unity_asset_path,
                        "unity_asset_abs_path": unity_result.unity_asset_abs_path,
                        "unity_renderer_prefab_path": unity_result.unity_renderer_prefab_path,
                        "unity_latest_prefab_path": unity_result.unity_latest_prefab_path,
                        "unity_import_log_path": unity_result.unity_log_path,
                    }
                )

        JOBS[moment_id] = result
        return result
    except Exception as exc:
        JOBS[moment_id] = {"id": moment_id, "status": "failed", "error": str(exc)}
        raise


async def process_moment_background(moment_id: str, file_path: Path, captured_at: str, duration: str) -> None:
    try:
        await asyncio.to_thread(process_moment, moment_id, file_path, captured_at, duration)
    except Exception:
        # process_moment records the error for the status endpoint.
        pass


@app.get("/api/health")
def root():
    return {"status": "server is running", "async_jobs": ASYNC_JOBS, "unity_import": UNITY_IMPORT}


@app.post("/api/v1/moments", status_code=202 if ASYNC_JOBS else 200)
async def create_moment(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    captured_at: str = Form(...),
    duration: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)
    moment_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{moment_id}.mp4"

    with file_path.open("wb") as destination:
        while chunk := await video.read(1024 * 1024):
            destination.write(chunk)

    if ASYNC_JOBS:
        JOBS[moment_id] = {"id": moment_id, "status": "queued"}
        background_tasks.add_task(process_moment_background, moment_id, file_path, captured_at, duration)
        return JOBS[moment_id]

    try:
        return process_moment(moment_id, file_path, captured_at, duration)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed with exit code {exc.returncode}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/api/v1/moments/{moment_id}")
def get_moment(moment_id: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    if moment_id not in JOBS:
        raise HTTPException(status_code=404, detail="Moment not found")
    return JOBS[moment_id]


@app.get("/api/v1/moments/{moment_id}/splat")
def download_splat(moment_id: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    job = JOBS.get(moment_id)
    if not job:
        raise HTTPException(status_code=404, detail="Moment not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail=f"Moment is {job.get('status')}")

    ply_path = find_fastgs_point_cloud(Path(job["model_path"]), iteration=TRAINING_ITERATIONS)
    return FileResponse(ply_path, filename=f"{job['dataset_name']}.ply", media_type="application/octet-stream")


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    app.add_api_route("/", root, methods=["GET"], include_in_schema=False)
