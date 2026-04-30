from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

#python cleanup_pipeline_artifacts.py --execute           

# Change these paths if your repo or Unity project moves.
REPO_ROOT = Path(__file__).resolve().parent
UNITY_PROJECTS = [
    Path(r"C:\Users\login\SplatTest"),
    REPO_ROOT / "unity_renderer",
]

BACKEND_UPLOADS = REPO_ROOT / "backend" / "uploads"
BACKEND_COLMAP_OUTPUT = REPO_ROOT / "backend" / "output"
FASTGS_DATASET_INPUTS = REPO_ROOT / "fastgs" / "datasets" / "input"
FASTGS_MODEL_OUTPUTS = REPO_ROOT / "fastgs" / "output"

INPUT_DIR_RE = re.compile(r"input[_-]?(\d+)$", re.IGNORECASE)
UNITY_IMPORT_LOG_RE = re.compile(r"gaussian_import_input[_-]?\d+\.log$", re.IGNORECASE)


@dataclass(frozen=True)
class CleanupTarget:
    path: Path
    category: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated Reminiscence pipeline artifacts without changing pipeline code. "
            "Dry-run is the default; pass --execute to delete."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Actually delete listed files/directories.")
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=1,
        help="Keep the newest N numbered input_N artifacts in each generated root. Default: 1.",
    )
    parser.add_argument(
        "--keep-uploads",
        type=int,
        default=0,
        help="Keep the newest N uploaded videos in backend/uploads. Default: 0.",
    )
    parser.add_argument(
        "--keep-logs",
        type=int,
        default=2,
        help="Keep the newest N Unity gaussian_import_input_N logs per Unity project. Default: 2.",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=None,
        help="Only delete targets older than this many days.",
    )
    parser.add_argument("--skip-uploads", action="store_true", help="Do not clean backend/uploads.")
    parser.add_argument("--skip-colmap", action="store_true", help="Do not clean backend/output.")
    parser.add_argument("--skip-fastgs-datasets", action="store_true", help="Do not clean fastgs/datasets/input/input_N.")
    parser.add_argument("--skip-fastgs-outputs", action="store_true", help="Do not clean fastgs/output/input_N.")
    parser.add_argument("--skip-unity-assets", action="store_true", help="Do not clean Unity Assets/GaussianAssets/input_N.")
    parser.add_argument("--skip-unity-logs", action="store_true", help="Do not clean Unity gaussian import logs.")
    parser.add_argument("--skip-cache", action="store_true", help="Do not clean Python __pycache__ folders.")
    return parser.parse_args()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_safe_target(target: CleanupTarget) -> None:
    path = target.path.resolve()
    allowed_roots = [
        BACKEND_UPLOADS,
        BACKEND_COLMAP_OUTPUT,
        FASTGS_DATASET_INPUTS,
        FASTGS_MODEL_OUTPUTS,
        REPO_ROOT / "backend",
    ]

    for unity_project in UNITY_PROJECTS:
        allowed_roots.extend(
            [
                unity_project / "Assets" / "GaussianAssets",
                unity_project / "Logs",
            ]
        )

    if not any(path == root.resolve() or is_within(path, root) for root in allowed_roots if root.exists() or root.parent.exists()):
        raise RuntimeError(f"Refusing to delete path outside generated artifact roots: {path}")

    if path in {REPO_ROOT.resolve(), (REPO_ROOT / "fastgs").resolve()}:
        raise RuntimeError(f"Refusing to delete broad project path: {path}")


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def numbered_children(root: Path) -> list[tuple[int, Path]]:
    if not root.exists():
        return []

    matches: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = INPUT_DIR_RE.fullmatch(child.name)
        if match:
            matches.append((int(match.group(1)), child))
    return sorted(matches, key=lambda item: item[0], reverse=True)


def old_enough(path: Path, older_than_days: float | None) -> bool:
    if older_than_days is None:
        return True
    if not path.exists():
        return False

    import time

    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds >= older_than_days * 24 * 60 * 60


def collect_numbered_dirs(root: Path, category: str, keep_latest: int) -> list[CleanupTarget]:
    targets: list[CleanupTarget] = []
    children = numbered_children(root)
    keep = {path.resolve() for _, path in children[: max(0, keep_latest)]}

    for index, path in children:
        if path.resolve() in keep:
            continue
        targets.append(CleanupTarget(path, category, f"old generated input_{index} directory"))
    return targets


def collect_unity_assets(unity_project: Path, keep_latest: int) -> list[CleanupTarget]:
    gaussian_root = unity_project / "Assets" / "GaussianAssets"
    targets = collect_numbered_dirs(gaussian_root, "unity-assets", keep_latest)

    for target in list(targets):
        meta_path = target.path.with_name(f"{target.path.name}.meta")
        if meta_path.exists():
            targets.append(CleanupTarget(meta_path, "unity-assets", "Unity folder .meta for generated splat directory"))

    return targets


def collect_uploads(keep_uploads: int) -> list[CleanupTarget]:
    if not BACKEND_UPLOADS.exists():
        return []

    files = sorted(
        [child for child in BACKEND_UPLOADS.iterdir() if child.is_file()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return [
        CleanupTarget(path, "uploads", "uploaded video already consumed by pipeline")
        for path in files[max(0, keep_uploads) :]
    ]


def collect_unity_logs(unity_project: Path, keep_logs: int) -> list[CleanupTarget]:
    logs_dir = unity_project / "Logs"
    if not logs_dir.exists():
        return []

    logs = sorted(
        [child for child in logs_dir.iterdir() if child.is_file() and UNITY_IMPORT_LOG_RE.fullmatch(child.name)],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return [
        CleanupTarget(path, "unity-logs", "old Gaussian splat import log")
        for path in logs[max(0, keep_logs) :]
    ]


def collect_pycache() -> list[CleanupTarget]:
    targets = []
    for root in (REPO_ROOT / "backend",):
        if not root.exists():
            continue
        for child in root.rglob("__pycache__"):
            if child.is_dir():
                targets.append(CleanupTarget(child, "cache", "Python bytecode cache"))
    return targets


def collect_targets(args: argparse.Namespace) -> list[CleanupTarget]:
    targets: list[CleanupTarget] = []

    if not args.skip_uploads:
        targets.extend(collect_uploads(args.keep_uploads))

    if not args.skip_colmap and BACKEND_COLMAP_OUTPUT.exists():
        targets.append(CleanupTarget(BACKEND_COLMAP_OUTPUT, "colmap", "intermediate COLMAP output"))

    if not args.skip_fastgs_datasets:
        targets.extend(collect_numbered_dirs(FASTGS_DATASET_INPUTS, "fastgs-datasets", args.keep_latest))
        stray_dataset_dir = FASTGS_DATASET_INPUTS / "distorted"
        if stray_dataset_dir.exists():
            targets.append(CleanupTarget(stray_dataset_dir, "fastgs-datasets", "stray COLMAP distorted folder"))

    if not args.skip_fastgs_outputs:
        targets.extend(collect_numbered_dirs(FASTGS_MODEL_OUTPUTS, "fastgs-outputs", args.keep_latest))

    for unity_project in UNITY_PROJECTS:
        if not unity_project.exists():
            continue
        if not args.skip_unity_assets:
            targets.extend(collect_unity_assets(unity_project, args.keep_latest))
        if not args.skip_unity_logs:
            targets.extend(collect_unity_logs(unity_project, args.keep_logs))

    if not args.skip_cache:
        targets.extend(collect_pycache())

    filtered = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if old_enough(target.path, args.older_than_days):
            filtered.append(target)
    return sorted(filtered, key=lambda item: (item.category, str(item.path).lower()))


def delete_target(target: CleanupTarget) -> None:
    require_safe_target(target)
    if target.path.is_dir():
        shutil.rmtree(target.path)
    elif target.path.exists():
        target.path.unlink()


def main() -> int:
    args = parse_args()
    targets = collect_targets(args)

    if not targets:
        print("No cleanup targets found.")
        return 0

    total_size = sum(path_size(target.path) for target in targets)
    action = "DELETE" if args.execute else "DRY-RUN"
    print(f"{action}: {len(targets)} target(s), {format_bytes(total_size)} total")
    print(f"Keeping newest {args.keep_latest} input_N artifact(s) per generated root.")
    print()

    for target in targets:
        size = format_bytes(path_size(target.path))
        print(f"[{target.category}] {size:>10}  {target.path}  ({target.reason})")

    if not args.execute:
        print()
        print("No files were deleted. Re-run with --execute to remove the listed targets.")
        return 0

    print()
    for target in targets:
        delete_target(target)
        print(f"deleted: {target.path}")

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
