from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineResult:
	dataset_name: str
	dataset_path: str
	model_path: str
	wsl_command: str


def _next_input_index(dataset_root: Path) -> int:
	max_index = 0

	for child in dataset_root.iterdir():
		if not child.is_dir():
			continue

		match = re.fullmatch(r"input[_-]?(\d+)", child.name)
		if not match:
			continue

		max_index = max(max_index, int(match.group(1)))

	return max_index + 1


def _require_file(path: Path) -> None:
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"Required file is missing: {path}")


def _windows_to_wsl_path(path: Path) -> str:
	result = subprocess.run(
		["wsl", "wslpath", "-a", str(path)],
		check=True,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	return result.stdout.strip()


def _build_wsl_training_command(fastgs_root: Path, dataset_name: str) -> str:
	fastgs_wsl = _windows_to_wsl_path(fastgs_root)
	dataset_rel = f"./datasets/input/{dataset_name}"
	model_rel = f"./output/{dataset_name}"

	return (
		"set -e; "
		f"cd {fastgs_wsl}; "
		"CUDA_VISIBLE_DEVICES=0 "
		f"OAR_JOB_ID={dataset_name} "
		"python train.py "
		f"-s {dataset_rel} "
		"--eval --densification_interval 500 --optimizer_type default "
		"--test_iterations 30000 --highfeature_lr 0.0015 --dense 0.003 --mult 0.7; "
		"CUDA_VISIBLE_DEVICES=0 "
		"python render.py "
		f"-s {dataset_rel} -m {model_rel} --skip_train"
	)


def prepare_fastgs_input_and_train(
	colmap_output_dir: Path,
	fastgs_root: Path,
	run_training: bool = True,
) -> PipelineResult:
	colmap_output_dir = colmap_output_dir.resolve()
	fastgs_root = fastgs_root.resolve()

	dataset_root = fastgs_root / "datasets" / "input"
	dataset_root.mkdir(parents=True, exist_ok=True)

	input_idx = _next_input_index(dataset_root)
	dataset_name = f"input_{input_idx}"

	dataset_dir = dataset_root / dataset_name
	images_dir = dataset_dir / "images"
	sparse0_dir = dataset_dir / "sparse" / "0"

	images_src = colmap_output_dir / "images"
	cameras_src = colmap_output_dir / "sparse" / "0" / "cameras.bin"
	images_bin_src = colmap_output_dir / "sparse" / "0" / "images.bin"
	points3d_bin_src = colmap_output_dir / "sparse" / "0" / "points3D.bin"
	points3d_ply_src = colmap_output_dir / "sparse_points.ply"

	if not images_src.exists() or not images_src.is_dir():
		raise FileNotFoundError(f"Required image directory is missing: {images_src}")

	_require_file(cameras_src)
	_require_file(images_bin_src)
	_require_file(points3d_bin_src)

	if not points3d_ply_src.exists():
		alternate_ply = colmap_output_dir / "points3d.ply"
		if alternate_ply.exists():
			points3d_ply_src = alternate_ply
		else:
			raise FileNotFoundError(
				"Missing sparse PLY output. Expected one of: "
				f"{colmap_output_dir / 'sparse_points.ply'} or {alternate_ply}"
			)

	images_dir.mkdir(parents=True, exist_ok=False)
	sparse0_dir.mkdir(parents=True, exist_ok=False)

	shutil.copytree(images_src, images_dir, dirs_exist_ok=True)
	shutil.copy2(cameras_src, sparse0_dir / "cameras.bin")
	shutil.copy2(images_bin_src, sparse0_dir / "images.bin")
	# FastGS expects points3D.bin, also keep lowercase alias for downstream compatibility.
	shutil.copy2(points3d_bin_src, sparse0_dir / "points3D.bin")
	shutil.copy2(points3d_bin_src, sparse0_dir / "points3d.bin")
	shutil.copy2(points3d_ply_src, sparse0_dir / "points3d.ply")

	if colmap_output_dir.exists():
		shutil.rmtree(colmap_output_dir)

	wsl_command = _build_wsl_training_command(fastgs_root=fastgs_root, dataset_name=dataset_name)

	if run_training:
		subprocess.run(["wsl", "bash", "-lc", wsl_command], check=True)

	return PipelineResult(
		dataset_name=dataset_name,
		dataset_path=str(dataset_dir),
		model_path=str(fastgs_root / "output" / dataset_name),
		wsl_command=wsl_command,
	)
