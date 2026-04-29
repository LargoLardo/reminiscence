import argparse
import shutil
import subprocess
import struct
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def run_command(command, cwd=None):
    """
    Runs a shell command safely and stops if it fails.
    """
    print("\nRunning:")
    print(" ".join(str(x) for x in command))

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in command)}")


def check_executable(name):
    """
    Makes sure ffmpeg/colmap exists on PATH.
    """
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Could not find '{name}' on PATH. Install it or add it to PATH."
        )


def clear_or_create_dir(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise RuntimeError(
                f"Folder already exists: {path}\n"
                f"Use --overwrite if you want to delete and recreate it."
            )

    path.mkdir(parents=True, exist_ok=True)


def copy_images_to_input(image_folder: Path, input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        raise RuntimeError(f"No images found in {image_folder}")

    for i, img in enumerate(images):
        new_name = f"image_{i:05d}{img.suffix.lower()}"
        shutil.copy2(img, input_dir / new_name)

    print(f"Copied {len(images)} images to {input_dir}")


def extract_frames_from_video(video_path: Path, input_dir: Path, fps: float, max_width: int):
    input_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = input_dir / "frame_%05d.jpg"

    vf = f"fps={fps},scale={max_width}:-1"

    run_command([
        "ffmpeg",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "2",
        str(output_pattern),
    ])

    frames = list(input_dir.glob("*.jpg"))

    if not frames:
        raise RuntimeError("FFmpeg finished but no frames were extracted.")

    print(f"Extracted {len(frames)} frames to {input_dir}")


def read_colmap_count(path: Path):
    try:
        with path.open("rb") as f:
            data = f.read(8)
    except OSError:
        return 0

    if len(data) != 8:
        return 0

    return struct.unpack("<Q", data)[0]


def find_best_sparse_model(sparse_dir: Path):
    model_dirs = sorted(p for p in sparse_dir.iterdir() if p.is_dir())

    if not model_dirs:
        raise RuntimeError(
            f"COLMAP did not create any sparse model in {sparse_dir}. "
            "Reconstruction probably failed."
        )

    ranked_models = []

    for model_dir in model_dirs:
        images_count = read_colmap_count(model_dir / "images.bin")
        points_count = read_colmap_count(model_dir / "points3D.bin")

        if (model_dir / "cameras.bin").exists() and images_count > 0:
            ranked_models.append((images_count, points_count, model_dir))

    if not ranked_models:
        raise RuntimeError(
            f"COLMAP created model folders in {sparse_dir}, but none contained "
            "a valid sparse reconstruction."
        )

    ranked_models.sort(key=lambda model: (model[0], model[1]), reverse=True)
    images_count, points_count, best_model = ranked_models[0]

    print(
        f"Using sparse model {best_model.name}: "
        f"{images_count} registered images, {points_count} points"
    )

    return best_model


def count_images(image_dir: Path):
    return sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def run_mapper_with_retries(database_path: Path, input_dir: Path, sparse_dir: Path, retries: int = 2):
    def mapper_cmd(output_path: Path):
        return [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(input_dir),
            "--output_path", str(output_path),
            "--Mapper.multiple_models", "1",
            "--Mapper.max_num_models", "50",
        ]

    run_command(mapper_cmd(sparse_dir))
    best_model = find_best_sparse_model(sparse_dir)
    best_count = read_colmap_count(best_model / "images.bin")
    target_count = min(100, max(10, count_images(input_dir) // 10))

    for attempt in range(1, retries + 1):
        if best_count >= target_count:
            break

        retry_dir = sparse_dir.parent / f"{sparse_dir.name}_retry_{attempt}"

        if retry_dir.exists():
            shutil.rmtree(retry_dir)

        retry_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Only {best_count} images registered; retrying mapper "
            f"({attempt}/{retries})..."
        )

        run_command(mapper_cmd(retry_dir))
        candidate_model = find_best_sparse_model(retry_dir)
        candidate_count = read_colmap_count(candidate_model / "images.bin")

        if candidate_count > best_count:
            best_model = candidate_model
            best_count = candidate_count

    print(f"Selected sparse model: {best_model}")
    return best_model


def run_colmap_sparse(scene_dir: Path, camera_model: str, single_camera: bool, matcher: str):
    """
    Runs:
    feature_extractor -> matcher -> mapper
    """
    input_dir = scene_dir / "input"
    distorted_dir = scene_dir / "distorted"
    sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--ImageReader.camera_model", camera_model,
    ]

    if single_camera:
        feature_cmd += ["--ImageReader.single_camera", "1"]

    run_command(feature_cmd)

    if matcher == "exhaustive":
        run_command([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
        ])
    elif matcher == "sequential":
        run_command([
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
        ])
    else:
        raise RuntimeError(f"Unknown matcher: {matcher}")

    sparse_model = run_mapper_with_retries(
        database_path=database_path,
        input_dir=input_dir,
        sparse_dir=sparse_dir,
    )

    print(f"Sparse reconstruction created at {sparse_model}")
    return sparse_model


def undistort_for_gaussian_splatting(scene_dir: Path, sparse_input: Path, max_image_size: int):
    """
    Creates:
    scene/images
    scene/sparse
    """
    input_dir = scene_dir / "input"
    run_command([
        "colmap", "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(sparse_input),
        "--output_path", str(scene_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    sparse_dir = scene_dir / "sparse"

    # Some COLMAP versions write sparse files directly into sparse/.
    # Gaussian Splatting usually expects sparse/0/.
    if (sparse_dir / "cameras.bin").exists():
        sparse_0 = sparse_dir / "0"
        sparse_0.mkdir(parents=True, exist_ok=True)

        for name in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sparse_dir / name
            dst = sparse_0 / name
            if src.exists():
                shutil.move(str(src), str(dst))

    expected_files = [
        scene_dir / "images",
        scene_dir / "sparse" / "0" / "cameras.bin",
        scene_dir / "sparse" / "0" / "images.bin",
        scene_dir / "sparse" / "0" / "points3D.bin",
    ]

    for p in expected_files:
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")

    print("\nGaussian Splatting-ready COLMAP scene created:")
    print(scene_dir)
    print("\nExpected structure:")
    print(scene_dir / "images")
    print(scene_dir / "sparse" / "0")


def export_sparse_ply(scene_dir: Path, sparse_model: Path):
    """
    Exports a regular COLMAP sparse point cloud PLY.
    This is NOT the final trained Gaussian Splat PLY.
    """
    output_ply = scene_dir / "sparse_points_colmap.ply"

    run_command([
        "colmap", "model_converter",
        "--input_path", str(sparse_model),
        "--output_path", str(output_ply),
        "--output_type", "PLY",
    ])

    print(f"Exported sparse COLMAP point cloud: {output_ply}")


def run_gaussian_training(scene_dir: Path, gs_repo: Path, output_dir: Path, iterations: int):
    """
    Optional: calls GraphDECO gaussian-splatting train.py.
    """
    train_py = gs_repo / "train.py"

    if not train_py.exists():
        raise RuntimeError(f"Could not find train.py at {train_py}")

    output_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        "python",
        str(train_py),
        "-s", str(scene_dir),
        "-m", str(output_dir),
        "--iterations", str(iterations),
    ])

    print("\nGaussian Splatting training done.")
    print("Look for final PLY around:")
    print(output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images/video with COLMAP for Gaussian Splatting."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to an image folder or video file.",
    )

    parser.add_argument(
        "scene",
        type=str,
        help="Output scene folder.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS to extract if input is a video. Default: 2",
    )

    parser.add_argument(
        "--frame-width",
        type=int,
        default=1600,
        help="Width for extracted video frames. Default: 1600",
    )

    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="Max undistorted image size for COLMAP. Default: 2000",
    )

    parser.add_argument(
        "--camera-model",
        type=str,
        default="OPENCV",
        help="COLMAP camera model. Good default: OPENCV",
    )

    parser.add_argument(
        "--multi-camera",
        action="store_true",
        help="Use this if images come from different cameras.",
    )

    parser.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="Use exhaustive for photos/small sets, sequential for video.",
    )

    parser.add_argument(
        "--export-ply",
        action="store_true",
        help="Export COLMAP sparse point cloud as PLY.",
    )

    parser.add_argument(
        "--train-gs",
        action="store_true",
        help="Also run Gaussian Splatting training after COLMAP.",
    )

    parser.add_argument(
        "--gs-repo",
        type=str,
        default=None,
        help="Path to GraphDECO gaussian-splatting repo if using --train-gs.",
    )

    parser.add_argument(
        "--gs-output",
        type=str,
        default=None,
        help="Output folder for Gaussian Splatting training.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
        help="Gaussian Splatting training iterations. Use 7000 quick, 30000 better.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing scene folder before running.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    scene_dir = Path(args.scene).resolve()

    check_executable("colmap")

    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        check_executable("ffmpeg")

    clear_or_create_dir(scene_dir, args.overwrite)

    input_dir = scene_dir / "input"

    if input_path.is_dir():
        copy_images_to_input(input_path, input_dir)
    elif input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        extract_frames_from_video(
            video_path=input_path,
            input_dir=input_dir,
            fps=args.fps,
            max_width=args.frame_width,
        )
    else:
        raise RuntimeError(
            "Input must be either an image folder or a video file.\n"
            f"Got: {input_path}"
        )

    sparse_model = run_colmap_sparse(
        scene_dir=scene_dir,
        camera_model=args.camera_model,
        single_camera=not args.multi_camera,
        matcher=args.matcher,
    )

    undistort_for_gaussian_splatting(
        scene_dir=scene_dir,
        sparse_input=sparse_model,
        max_image_size=args.max_image_size,
    )

    if args.export_ply:
        export_sparse_ply(scene_dir, sparse_model)

    if args.train_gs:
        if args.gs_repo is None:
            raise RuntimeError("--gs-repo is required when using --train-gs")

        gs_repo = Path(args.gs_repo).resolve()

        if args.gs_output is None:
            gs_output = scene_dir.parent / f"{scene_dir.name}_gs_output"
        else:
            gs_output = Path(args.gs_output).resolve()

        run_gaussian_training(
            scene_dir=scene_dir,
            gs_repo=gs_repo,
            output_dir=gs_output,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
