from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Change this one path when you want the generated splat imported into a different Unity project.
TARGET_UNITY_PROJECT = Path(r"C:\Users\login\SplatTest")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FASTGS_OUTPUT_ROOT = REPO_ROOT / "fastgs" / "output"
DEFAULT_UNITY_PROJECT = TARGET_UNITY_PROJECT
DEFAULT_UNITY_ASSET_ROOT = "Assets/GaussianAssets"
DEFAULT_QUALITY = "Medium"

INPUT_DIR_RE = re.compile(r"input[_-]?(\d+)$")
ITERATION_DIR_RE = re.compile(r"iteration[_-]?(\d+)$")


@dataclass(frozen=True)
class UnitySplatTransferResult:
    dataset_name: str
    source_ply: str
    copied_ply: str
    unity_output_folder: str
    unity_asset_path: str
    unity_asset_abs_path: str
    unity_renderer_prefab_path: str
    unity_latest_prefab_path: str
    unity_log_path: str | None = None


def _numbered_child_dirs(root: Path, pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    matches: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match:
            matches.append((int(match.group(1)), child))
    return matches


def find_latest_fastgs_model(output_root: Path = DEFAULT_FASTGS_OUTPUT_ROOT) -> Path:
    """Return fastgs/output/input_N with the highest N."""
    candidates = _numbered_child_dirs(output_root.resolve(), INPUT_DIR_RE)
    if not candidates:
        raise FileNotFoundError(f"No input_N directories found in: {output_root}")
    return max(candidates, key=lambda item: item[0])[1]


def find_fastgs_point_cloud(model_dir: Path, iteration: int | None = None) -> Path:
    model_dir = model_dir.resolve()
    point_cloud_root = model_dir / "point_cloud"

    if iteration is None:
        candidates = _numbered_child_dirs(point_cloud_root, ITERATION_DIR_RE)
        if not candidates:
            raise FileNotFoundError(f"No iteration_N directories found in: {point_cloud_root}")
        iteration_dir = max(candidates, key=lambda item: item[0])[1]
    else:
        iteration_dir = point_cloud_root / f"iteration_{iteration}"

    ply_path = iteration_dir / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"FastGS point cloud PLY is missing: {ply_path}")
    return ply_path


def _unity_relative_asset_folder(unity_project: Path, asset_root: str | Path, dataset_name: str) -> tuple[str, Path]:
    unity_project = unity_project.resolve()
    asset_root_path = Path(asset_root)

    if asset_root_path.is_absolute():
        asset_root_abs = asset_root_path.resolve()
        try:
            asset_root_rel_path = asset_root_abs.relative_to(unity_project)
        except ValueError as exc:
            raise ValueError(f"Unity asset output must be inside the Unity project: {asset_root_abs}") from exc
        asset_root_rel = asset_root_rel_path.as_posix()
    else:
        asset_root_rel = str(asset_root).replace("\\", "/").strip("/")

    if asset_root_rel != "Assets" and not asset_root_rel.startswith("Assets/"):
        raise ValueError(f"Unity asset output must be inside Assets/: {asset_root_rel}")

    output_folder_rel = f"{asset_root_rel.rstrip('/')}/{dataset_name}"
    output_folder_abs = unity_project / Path(output_folder_rel.replace("/", os.sep))
    return output_folder_rel, output_folder_abs


def _project_unity_version(unity_project: Path) -> str | None:
    version_file = unity_project / "ProjectSettings" / "ProjectVersion.txt"
    if not version_file.is_file():
        return None

    for line in version_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("m_EditorVersion:"):
            return line.split(":", 1)[1].strip()
    return None


def find_unity_executable(unity_project: Path = DEFAULT_UNITY_PROJECT, unity_executable: Path | None = None) -> Path:
    if unity_executable is not None:
        unity_executable = unity_executable.resolve()
        if unity_executable.is_file():
            return unity_executable
        raise FileNotFoundError(f"Unity executable does not exist: {unity_executable}")

    for env_name in ("UNITY_EDITOR", "UNITY_EXE"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser().resolve()
            if candidate.is_file():
                return candidate

    editor_version = _project_unity_version(unity_project.resolve())
    if editor_version:
        candidate = Path("C:/Program Files/Unity/Hub/Editor") / editor_version / "Editor" / "Unity.exe"
        if candidate.is_file():
            return candidate

    hub_root = Path("C:/Program Files/Unity/Hub/Editor")
    if hub_root.is_dir():
        candidates = sorted(hub_root.glob("*/Editor/Unity.exe"), reverse=True)
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        "Could not find Unity.exe. Set UNITY_EDITOR to the full path, "
        "or pass --unity-exe."
    )


def transfer_fastgs_model_to_unity(
    model_dir: Path,
    unity_project: Path = DEFAULT_UNITY_PROJECT,
    asset_root: str | Path = DEFAULT_UNITY_ASSET_ROOT,
    *,
    iteration: int | None = None,
    convert: bool = True,
    unity_executable: Path | None = None,
    quality: str = DEFAULT_QUALITY,
) -> UnitySplatTransferResult:
    model_dir = model_dir.resolve()
    unity_project = unity_project.resolve()
    dataset_name = model_dir.name
    source_ply = find_fastgs_point_cloud(model_dir, iteration)
    output_folder_rel, output_folder_abs = _unity_relative_asset_folder(unity_project, asset_root, dataset_name)

    output_folder_abs.mkdir(parents=True, exist_ok=True)
    copied_ply = output_folder_abs / f"{dataset_name}.ply"
    shutil.copy2(source_ply, copied_ply)

    cameras_json = model_dir / "cameras.json"
    if cameras_json.is_file():
        shutil.copy2(cameras_json, output_folder_abs / "cameras.json")

    unity_asset_path = f"{output_folder_rel}/{dataset_name}.asset"
    unity_asset_abs_path = unity_project / Path(unity_asset_path.replace("/", os.sep))
    unity_renderer_prefab_path = f"{output_folder_rel}/{dataset_name}_Renderer.prefab"
    unity_latest_prefab_path = f"{output_folder_rel}/LatestGaussianSplat.prefab"
    unity_log_path: Path | None = None

    if convert:
        unity_exe = find_unity_executable(unity_project, unity_executable)
        log_dir = unity_project / "Logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        unity_log_path = log_dir / f"gaussian_import_{dataset_name}.log"

        cmd = [
            str(unity_exe),
            "-batchmode",
            "-quit",
            "-projectPath",
            str(unity_project),
            "-logFile",
            str(unity_log_path),
            "-executeMethod",
            "Reminiscence.Editor.BatchGaussianSplatImporter.ImportFromCommandLine",
            "-gsInput",
            str(copied_ply),
            "-gsOutputFolder",
            output_folder_rel,
            "-gsQuality",
            quality,
        ]
        subprocess.run(cmd, check=True)

        if not unity_asset_abs_path.is_file():
            raise FileNotFoundError(
                f"Unity finished, but the Gaussian splat asset was not created: {unity_asset_abs_path}"
            )

    return UnitySplatTransferResult(
        dataset_name=dataset_name,
        source_ply=str(source_ply),
        copied_ply=str(copied_ply),
        unity_output_folder=output_folder_rel,
        unity_asset_path=unity_asset_path,
        unity_asset_abs_path=str(unity_asset_abs_path),
        unity_renderer_prefab_path=unity_renderer_prefab_path,
        unity_latest_prefab_path=unity_latest_prefab_path,
        unity_log_path=str(unity_log_path) if unity_log_path else None,
    )


def transfer_latest_fastgs_to_unity(
    output_root: Path = DEFAULT_FASTGS_OUTPUT_ROOT,
    unity_project: Path = DEFAULT_UNITY_PROJECT,
    asset_root: str | Path = DEFAULT_UNITY_ASSET_ROOT,
    *,
    iteration: int | None = None,
    convert: bool = True,
    unity_executable: Path | None = None,
    quality: str = DEFAULT_QUALITY,
) -> UnitySplatTransferResult:
    model_dir = find_latest_fastgs_model(output_root)
    return transfer_fastgs_model_to_unity(
        model_dir=model_dir,
        unity_project=unity_project,
        asset_root=asset_root,
        iteration=iteration,
        convert=convert,
        unity_executable=unity_executable,
        quality=quality,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy the newest FastGS PLY into Unity and import it as a splat asset.")
    parser.add_argument("--fastgs-output", type=Path, default=DEFAULT_FASTGS_OUTPUT_ROOT)
    parser.add_argument("--unity-project", type=Path, default=DEFAULT_UNITY_PROJECT)
    parser.add_argument("--asset-root", default=DEFAULT_UNITY_ASSET_ROOT)
    parser.add_argument("--input-name", help="Use a specific fastgs/output/input_N directory instead of the newest one.")
    parser.add_argument("--iteration", type=int, help="Use a specific point_cloud/iteration_N directory.")
    parser.add_argument("--unity-exe", type=Path, help="Full path to Unity.exe.")
    parser.add_argument("--quality", default=DEFAULT_QUALITY, choices=["VeryHigh", "High", "Medium", "Low", "VeryLow"])
    parser.add_argument("--copy-only", action="store_true", help="Copy the PLY without running Unity's asset converter.")
    args = parser.parse_args()

    if args.input_name:
        model_dir = args.fastgs_output / args.input_name
        result = transfer_fastgs_model_to_unity(
            model_dir=model_dir,
            unity_project=args.unity_project,
            asset_root=args.asset_root,
            iteration=args.iteration,
            convert=not args.copy_only,
            unity_executable=args.unity_exe,
            quality=args.quality,
        )
    else:
        result = transfer_latest_fastgs_to_unity(
            output_root=args.fastgs_output,
            unity_project=args.unity_project,
            asset_root=args.asset_root,
            iteration=args.iteration,
            convert=not args.copy_only,
            unity_executable=args.unity_exe,
            quality=args.quality,
        )

    print(f"dataset_name={result.dataset_name}")
    print(f"source_ply={result.source_ply}")
    print(f"copied_ply={result.copied_ply}")
    print(f"unity_asset_path={result.unity_asset_path}")
    print(f"unity_asset_abs_path={result.unity_asset_abs_path}")
    print(f"unity_renderer_prefab_path={result.unity_renderer_prefab_path}")
    print(f"unity_latest_prefab_path={result.unity_latest_prefab_path}")
    if result.unity_log_path:
        print(f"unity_log_path={result.unity_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
