import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "reminiscence"
GPU_IMAGE = "ghcr.io/andyjyzhang/reminiscence:gpu-latest"
VOLUME_NAME = "reminiscence-data"
USAGE_DICT_NAME = "reminiscence-usage"

DATA_ROOT = Path("/reminiscence-data")
JOB_ROOT = DATA_ROOT / "jobs"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MONTHLY_JOB_LIMIT = 30
TRAINING_ITERATIONS = 7000
JOB_TIMEOUT_SECONDS = 20 * 60
RETENTION_DAYS = 7
GPU_CPU_CORES = 4.0
GPU_MEMORY_MIB = 32 * 1024
WEB_CPU_CORES = 0.125
WEB_MEMORY_MIB = 512
SCALEDOWN_WINDOW_SECONDS = 30
FREE_MONTHLY_CREDIT_USD = 30.0
REQUIRED_WORKSPACE_BUDGET_USD = 29.0

# Current Modal list prices. The estimate deliberately assumes the public web
# API is busy every second of a 31-day month and every GPU job hits its timeout.
A10_USD_PER_SECOND = 0.000306
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
LONGEST_MONTH_SECONDS = 31 * 24 * 60 * 60
CLEANUP_MAX_SECONDS_PER_MONTH = 31 * 10 * 60


def estimated_max_monthly_compute_usd() -> float:
    gpu_seconds = MONTHLY_JOB_LIMIT * (JOB_TIMEOUT_SECONDS + SCALEDOWN_WINDOW_SECONDS)
    gpu_worker = gpu_seconds * (
        A10_USD_PER_SECOND
        + GPU_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + (GPU_MEMORY_MIB / 1024) * MEMORY_GIB_USD_PER_SECOND
    )
    public_api = LONGEST_MONTH_SECONDS * (
        WEB_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + (WEB_MEMORY_MIB / 1024) * MEMORY_GIB_USD_PER_SECOND
    )
    cleanup = CLEANUP_MAX_SECONDS_PER_MONTH * (
        WEB_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + (WEB_MEMORY_MIB / 1024) * MEMORY_GIB_USD_PER_SECOND
    )
    return round(gpu_worker + public_api + cleanup, 2)


ESTIMATED_MAX_MONTHLY_COMPUTE_USD = estimated_max_monthly_compute_usd()
if ESTIMATED_MAX_MONTHLY_COMPUTE_USD >= REQUIRED_WORKSPACE_BUDGET_USD:
    raise ValueError("Configured resource limits exceed the required Modal workspace budget")

app = modal.App(APP_NAME, tags={"project": "reminiscence"})
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
usage = modal.Dict.from_name(USAGE_DICT_NAME, create_if_missing=True)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.124.4",
    "python-multipart==0.0.20",
)
gpu_image = modal.Image.from_registry(GPU_IMAGE).add_local_file(
    "prepare_colmap_windows.py",
    "/app/prepare_colmap_windows.py",
    copy=True,
)


def _public_result(call_id: str, result: dict) -> dict:
    return {
        key: value
        for key, value in {**result, "id": call_id}.items()
        if key not in {
            "splat_path",
            "source_video_path",
            "render_preview_path",
            "source_content_type",
            "source_filename",
        }
    }


def _copy_middle_render(render_dir: Path, destination: Path) -> bool:
    if not render_dir.is_dir():
        return False

    renders = sorted(render_dir.glob("*.png"))
    if not renders:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(renders[len(renders) // 2], destination)
    return True


@app.function(
    image=gpu_image,
    gpu="A10",
    volumes={str(DATA_ROOT): data_volume},
    timeout=JOB_TIMEOUT_SECONDS,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    cpu=GPU_CPU_CORES,
    memory=GPU_MEMORY_MIB,
)
def reconstruct(
    upload_id: str,
    captured_at: str,
    duration: str,
    source_filename: str = "source.mp4",
    source_content_type: str = "video/mp4",
) -> dict:
    from backend.rendering_pipeline import prepare_fastgs_input_and_train
    from backend.unity_splat_transfer import find_fastgs_point_cloud

    data_volume.reload()
    project_root = Path("/app")
    job_dir = JOB_ROOT / upload_id
    video_path = job_dir / "input.mp4"
    work_dir = Path("/tmp") / f"reminiscence-{upload_id}"
    colmap_output_dir = work_dir / "colmap"
    pipeline_result = None

    if not video_path.is_file():
        raise FileNotFoundError(f"Uploaded video is missing: {video_path}")

    try:
        colmap_env = os.environ.copy()
        colmap_env["COLMAP_USE_GPU"] = "0"
        colmap_env["QT_QPA_PLATFORM"] = "offscreen"
        subprocess.run(
            [
                sys.executable,
                str(project_root / "prepare_colmap_windows.py"),
                str(video_path),
                str(colmap_output_dir),
                "--fps",
                "5",
                "--overwrite",
                "--export-ply",
            ],
            check=True,
            env=colmap_env,
        )
        pipeline_result = prepare_fastgs_input_and_train(
            colmap_output_dir=colmap_output_dir,
            fastgs_root=project_root / "fastgs",
            run_training=True,
            training_iterations=TRAINING_ITERATIONS,
        )

        source_ply = find_fastgs_point_cloud(
            Path(pipeline_result.model_path),
            iteration=TRAINING_ITERATIONS,
        )
        splat_path = job_dir / "memory.ply"
        render_preview_path = job_dir / "render_preview.png"
        shutil.copy2(source_ply, splat_path)
        has_render_preview = _copy_middle_render(
            Path(pipeline_result.render_path),
            render_preview_path,
        )
        video_size = video_path.stat().st_size
        data_volume.commit()

        return {
            "status": "complete",
            "captured_at": captured_at,
            "duration_seconds": float(duration),
            "size_bytes": video_size,
            "dataset_name": upload_id,
            "registered_image_count": pipeline_result.registered_image_count,
            "splat_download_url": "",
            "source_video_url": "",
            "render_preview_url": "" if has_render_preview else None,
            "source_video_path": str(video_path),
            "source_filename": source_filename,
            "source_content_type": source_content_type,
            "render_preview_path": str(render_preview_path) if has_render_preview else "",
            "splat_path": str(splat_path),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if pipeline_result is not None:
            shutil.rmtree(pipeline_result.dataset_path, ignore_errors=True)
            shutil.rmtree(pipeline_result.model_path, ignore_errors=True)


@app.function(
    image=web_image,
    volumes={str(DATA_ROOT): data_volume},
    timeout=15 * 60,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    cpu=WEB_CPU_CORES,
    memory=WEB_MEMORY_MIB,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    web_app = FastAPI(title="Reminiscence Modal API")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    async def poll_result(call_id: str) -> dict | None:
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            return await function_call.get.aio(timeout=0)
        except TimeoutError:
            return None
        except modal.exception.OutputExpiredError as exc:
            raise HTTPException(status_code=404, detail="Moment not found or expired") from exc
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    @web_app.get("/api/health")
    async def health():
        return {
            "status": "server is running",
            "gpu": "A10",
            "max_gpu_containers": 1,
            "monthly_job_limit": MONTHLY_JOB_LIMIT,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            "training_iterations": TRAINING_ITERATIONS,
            "estimated_max_monthly_compute_usd": ESTIMATED_MAX_MONTHLY_COMPUTE_USD,
            "modal_free_monthly_compute_credit_usd": FREE_MONTHLY_CREDIT_USD,
            "required_modal_workspace_budget_usd": REQUIRED_WORKSPACE_BUDGET_USD,
        }

    @web_app.post("/api/v1/moments", status_code=202)
    async def create_moment(
        video: UploadFile = File(...),
        captured_at: str = Form(...),
        duration: str = Form(...),
    ):
        try:
            duration_seconds = float(duration)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Duration must be a number") from exc
        if duration_seconds < 0:
            raise HTTPException(status_code=422, detail="Duration cannot be negative")

        now = datetime.now(timezone.utc)
        monthly_usage_key = f"jobs:month:{now.strftime('%Y-%m')}"
        jobs_this_month = await usage.get.aio(monthly_usage_key, 0)
        if jobs_this_month >= MONTHLY_JOB_LIMIT:
            raise HTTPException(status_code=429, detail="Monthly reconstruction limit reached")
        await usage.put.aio(monthly_usage_key, jobs_this_month + 1)

        upload_id = str(uuid.uuid4())
        job_dir = JOB_ROOT / upload_id
        video_path = job_dir / "input.mp4"
        job_dir.mkdir(parents=True, exist_ok=False)
        uploaded_bytes = 0

        try:
            with video_path.open("wb") as destination:
                while chunk := await video.read(1024 * 1024):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                        )
                    destination.write(chunk)

            await data_volume.commit.aio()
            call = await reconstruct.spawn.aio(
                upload_id,
                captured_at,
                str(duration_seconds),
                Path(video.filename or "source.mp4").name,
                video.content_type or "video/mp4",
            )
            return {"id": call.object_id, "status": "queued"}
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            await data_volume.commit.aio()
            raise

    @web_app.get("/api/v1/moments/{call_id}")
    async def get_moment(call_id: str):
        result = await poll_result(call_id)
        if result is None:
            return {"id": call_id, "status": "processing"}
        public_result = _public_result(call_id, result)
        if public_result.get("status") == "complete":
            public_result["splat_download_url"] = f"/api/v1/moments/{call_id}/splat"
            public_result["source_video_url"] = f"/api/v1/moments/{call_id}/source"
            if public_result.get("render_preview_url") is not None:
                public_result["render_preview_url"] = f"/api/v1/moments/{call_id}/preview"
        return public_result

    @web_app.get("/api/v1/moments/{call_id}/source")
    async def download_source(call_id: str):
        result = await poll_result(call_id)
        if result is None:
            raise HTTPException(status_code=409, detail="Moment is processing")
        if result.get("status") != "complete":
            raise HTTPException(status_code=409, detail=f"Moment is {result.get('status')}")

        data_volume.reload()
        source_video_path = Path(result.get("source_video_path") or Path(result["splat_path"]).parent / "input.mp4")
        if not source_video_path.is_file():
            raise HTTPException(status_code=404, detail="Source video has expired")
        return FileResponse(
            source_video_path,
            filename=result.get("source_filename") or f"{result['dataset_name']}.mp4",
            media_type=result.get("source_content_type") or "video/mp4",
            content_disposition_type="inline",
        )

    @web_app.get("/api/v1/moments/{call_id}/preview")
    async def download_render_preview(call_id: str):
        result = await poll_result(call_id)
        if result is None:
            raise HTTPException(status_code=409, detail="Moment is processing")
        if result.get("status") != "complete":
            raise HTTPException(status_code=409, detail=f"Moment is {result.get('status')}")

        data_volume.reload()
        render_preview_path = Path(result.get("render_preview_path") or "")
        if not render_preview_path.is_file():
            raise HTTPException(status_code=404, detail="Render preview is unavailable")
        return FileResponse(
            render_preview_path,
            filename=f"{result['dataset_name']}_preview.png",
            media_type="image/png",
            content_disposition_type="inline",
        )

    @web_app.get("/api/v1/moments/{call_id}/splat")
    async def download_splat(call_id: str):
        result = await poll_result(call_id)
        if result is None:
            raise HTTPException(status_code=409, detail="Moment is processing")
        if result.get("status") != "complete":
            raise HTTPException(status_code=409, detail=f"Moment is {result.get('status')}")

        data_volume.reload()
        splat_path = Path(result["splat_path"])
        if not splat_path.is_file():
            raise HTTPException(status_code=404, detail="Splat file has expired")
        return FileResponse(
            splat_path,
            filename=f"{result['dataset_name']}.ply",
            media_type="application/octet-stream",
        )

    return web_app


@app.function(
    image=web_image,
    volumes={str(DATA_ROOT): data_volume},
    schedule=modal.Cron("17 4 * * *"),
    timeout=10 * 60,
    max_containers=1,
    cpu=WEB_CPU_CORES,
    memory=WEB_MEMORY_MIB,
)
def cleanup_expired_jobs() -> int:
    data_volume.reload()
    cutoff = time.time() - RETENTION_DAYS * 24 * 60 * 60
    removed = 0
    monthly_usage_key = f"jobs:month:{datetime.now(timezone.utc).strftime('%Y-%m')}"
    monthly_usage = usage.get(monthly_usage_key)
    if monthly_usage is not None:
        usage.put(monthly_usage_key, monthly_usage)

    if JOB_ROOT.is_dir():
        for job_dir in JOB_ROOT.iterdir():
            if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                shutil.rmtree(job_dir)
                removed += 1

    if removed:
        data_volume.commit()
    return removed
