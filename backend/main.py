from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
import uuid
import subprocess
import sys

try:
    from .unity_splat_transfer import DEFAULT_UNITY_PROJECT
    from .rendering_pipeline import prepare_fastgs_input_and_train
    from .unity_splat_transfer import transfer_fastgs_model_to_unity
except ImportError:
    from unity_splat_transfer import DEFAULT_UNITY_PROJECT
    from rendering_pipeline import prepare_fastgs_input_and_train
    from unity_splat_transfer import transfer_fastgs_model_to_unity

app = FastAPI()

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
COLMAP_OUTPUT_DIR = BACKEND_DIR / "output"
PREPARE_COLMAP_SCRIPT = PROJECT_ROOT / "prepare_colmap_windows.py"
UNITY_PROJECT_DIR = DEFAULT_UNITY_PROJECT
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
        sys.executable,
        str(PREPARE_COLMAP_SCRIPT),
        str(file_path),
        str(COLMAP_OUTPUT_DIR),
        "--fps", "5",
        "--overwrite",
        "--export-ply",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"COLMAP preparation failed with exit code {exc.returncode}",
        ) from exc

    try:
        pipeline_result = prepare_fastgs_input_and_train(
            colmap_output_dir=COLMAP_OUTPUT_DIR,
            fastgs_root=PROJECT_ROOT / "fastgs",
            run_training=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FastGS training/rendering failed with exit code {exc.returncode}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FastGS input preparation failed: {exc}",
        ) from exc

    try:
        unity_result = transfer_fastgs_model_to_unity(
            model_dir=Path(pipeline_result.model_path),
            unity_project=UNITY_PROJECT_DIR,
            convert=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unity Gaussian splat import failed with exit code {exc.returncode}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unity Gaussian splat transfer failed: {exc}",
        ) from exc

    return {
        "id": moment_id,
        "file_path": str(file_path),
        "captured_at": captured_at,
        "duration_seconds": float(duration),
        "size_bytes": file_path.stat().st_size,
        "dataset_name": pipeline_result.dataset_name,
        "dataset_path": pipeline_result.dataset_path,
        "model_path": pipeline_result.model_path,
        "render_path": pipeline_result.render_path,
        "registered_image_count": pipeline_result.registered_image_count,
        "unity_ply_path": unity_result.copied_ply,
        "unity_asset_path": unity_result.unity_asset_path,
        "unity_asset_abs_path": unity_result.unity_asset_abs_path,
        "unity_renderer_prefab_path": unity_result.unity_renderer_prefab_path,
        "unity_latest_prefab_path": unity_result.unity_latest_prefab_path,
        "unity_import_log_path": unity_result.unity_log_path,
    }
