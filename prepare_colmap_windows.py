import argparse
import shutil
import subprocess
import struct
from time import time
from pathlib import Path
from typing import Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


DEFAULT_COLMAP_PATHS = [
    r"C:\Users\login\Downloads\colmap-x64-windows-cuda\COLMAP.bat",
    r"C:\Users\login\Downloads\colmap-x64-windows-cuda\bin\colmap.exe",
    r"C:\Program Files\COLMAP\COLMAP.bat",
    r"C:\Program Files\COLMAP\bin\colmap.exe",
]

START_TIME = time()


def find_colmap(user_path=None):
    if user_path:
        path = Path(user_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"COLMAP path does not exist: {path}")

    # If colmap is already on PATH
    found = shutil.which("colmap")
    if found:
        return found

    # Check common/default paths
    for p in DEFAULT_COLMAP_PATHS:
        path = Path(p)
        if path.exists():
            return str(path)

    raise RuntimeError(
        "Could not find COLMAP.\n"
        "Pass it manually with:\n"
        '--colmap "C:\\Users\\login\\Downloads\\colmap-x64-windows-cuda\\COLMAP.bat"'
    )


def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(
            "FFmpeg was not found on PATH.\n"
            "FFmpeg is only needed for video input.\n"
            "Install FFmpeg or use an image folder instead."
        )
    return found


def run(cmd):
    print("\nRunning:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")


def make_clean_dir(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to delete it first."
            )
    path.mkdir(parents=True, exist_ok=True)


def copy_images(data_dir: Path, input_dir: Path):
    images = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        raise RuntimeError(f"No images found in {data_dir}")

    input_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        new_name = f"frame_{i:05d}{img.suffix.lower()}"
        shutil.copy2(img, input_dir / new_name)

    print(f"Copied {len(images)} images into {input_dir}")


def extract_video_frames(video_path: Path, input_dir: Path, fps: float, width: Optional[int]):
    find_ffmpeg()

    input_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = input_dir / "frame_%05d.jpg"
    video_filter = f"fps={fps}"

    if width is not None:
        video_filter += f",scale={width}:-1"

    run([
        "ffmpeg",
        "-i", str(video_path),
        "-vf", video_filter,
        "-q:v", "2",
        str(output_pattern),
    ])

    frames = sorted(input_dir.glob("*.jpg"))

    if not frames:
        raise RuntimeError("No frames were extracted from the video.")

    print(f"Extracted {len(frames)} frames into {input_dir}")


def fix_sparse_folder(output_dir: Path):
    sparse_dir = output_dir / "sparse"
    sparse_0_dir = sparse_dir / "0"

    # Some COLMAP outputs place .bin files directly in sparse/
    # Gaussian Splatting usually expects sparse/0/
    if (sparse_dir / "cameras.bin").exists():
        sparse_0_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sparse_dir / filename
            dst = sparse_0_dir / filename

            if src.exists():
                shutil.move(str(src), str(dst))

    required = [
        output_dir / "images",
        sparse_0_dir / "cameras.bin",
        sparse_0_dir / "images.bin",
        sparse_0_dir / "points3D.bin",
    ]

    for path in required:
        if not path.exists():
            raise RuntimeError(f"Missing expected output: {path}")


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


def run_colmap(
    colmap_cmd: str,
    output_dir: Path,
    matcher: str,
    camera_model: str,
    max_image_size: int,
    export_ply: bool,
):
    input_dir = output_dir / "input"
    distorted_dir = output_dir / "distorted"
    distorted_sparse_dir = distorted_dir / "sparse"
    database_path = distorted_dir / "database.db"

    distorted_sparse_dir.mkdir(parents=True, exist_ok=True)

    run([
        colmap_cmd,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
    ])

    if matcher == "sequential":
        run([
            colmap_cmd,
            "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "30",
            "--SequentialMatching.quadratic_overlap", "1",
        ])
    else:
        run([
            colmap_cmd,
            "exhaustive_matcher",
            "--database_path", str(database_path),
        ])

    run([
        colmap_cmd,
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--output_path", str(distorted_sparse_dir),
        "--Mapper.multiple_models", "1",
        "--Mapper.max_num_models", "50",
    ])

    sparse_model = find_best_sparse_model(distorted_sparse_dir)

    run([
        colmap_cmd,
        "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(sparse_model),
        "--output_path", str(output_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ])

    fix_sparse_folder(output_dir)

    if export_ply:
        run([
            colmap_cmd,
            "model_converter",
            "--input_path", str(sparse_model),
            "--output_path", str(output_dir / "sparse_points.ply"),
            "--output_type", "PLY",
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images or video with COLMAP for Gaussian Splatting."
    )

    parser.add_argument(
        "input",
        help="Input image folder or video file."
    )

    parser.add_argument(
        "output_folder",
        help="Output folder."
    )

    parser.add_argument(
        "--colmap",
        default=None,
        help="Path to COLMAP.bat or colmap.exe."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for video input. Default: 2."
    )

    parser.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Optional width for extracted video frames. Default: keep original video size."
    )

    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="Use sequential for videos/ordered frames. Use exhaustive for unordered photos."
    )

    parser.add_argument(
        "--camera-model",
        default="OPENCV",
        help="COLMAP camera model. Default: OPENCV."
    )

    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="Max image size for undistorted output. Default: 2000."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output folder if it already exists."
    )

    parser.add_argument(
        "--export-ply",
        action="store_true",
        help="Also export sparse COLMAP point cloud as sparse_points.ply."
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_folder).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    colmap_cmd = find_colmap(args.colmap)
    print(f"Using COLMAP: {colmap_cmd}")

    make_clean_dir(output_dir, args.overwrite)

    input_dir = output_dir / "input"

    if input_path.is_dir():
        copy_images(input_path, input_dir)
    elif input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        extract_video_frames(
            video_path=input_path,
            input_dir=input_dir,
            fps=args.fps,
            width=args.frame_width,
        )
    else:
        raise RuntimeError(
            f"Input must be an image folder or video file. Got: {input_path}"
        )

    run_colmap(
        colmap_cmd=colmap_cmd,
        output_dir=output_dir,
        matcher=args.matcher,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        export_ply=args.export_ply,
    )

    print("\nDone. Took {:.1f} seconds.".format(time() - START_TIME))
    print("Gaussian Splatting-ready folder:")
    print(output_dir)

    print("\nExpected structure:")
    print(output_dir / "images")
    print(output_dir / "sparse" / "0" / "cameras.bin")
    print(output_dir / "sparse" / "0" / "images.bin")
    print(output_dir / "sparse" / "0" / "points3D.bin")

    if args.export_ply:
        print("\nSparse COLMAP point cloud:")
        print(output_dir / "sparse_points.ply")


if __name__ == "__main__":
    main()
