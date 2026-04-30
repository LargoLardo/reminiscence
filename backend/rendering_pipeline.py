from __future__ import annotations

import re
import shutil
import shlex
import subprocess
import struct
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FASTGS_ITERATIONS = 5000


@dataclass(frozen=True)
class PipelineResult:
	dataset_name: str
	dataset_path: str
	model_path: str
	wsl_command: str
	render_path: str = ""
	registered_image_count: int = 0


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


def _read_colmap_registered_image_count(images_bin_path: Path) -> int:
	with open(images_bin_path, "rb") as image_file:
		data = image_file.read(8)
	if len(data) != 8:
		raise ValueError(f"COLMAP images.bin is too small to read image count: {images_bin_path}")
	return struct.unpack("<Q", data)[0]


def _windows_to_wsl_path(path: Path) -> str:
	path = path.resolve()

	if path.drive:
		drive = path.drive.rstrip(":").lower()
		rest = "/".join(path.parts[1:])
		return f"/mnt/{drive}/{rest}"

	return path.as_posix()


def _build_wsl_training_command(
	fastgs_root: Path,
	dataset_name: str,
	training_iterations: int = DEFAULT_FASTGS_ITERATIONS,
) -> str:
	fastgs_wsl = shlex.quote(_windows_to_wsl_path(fastgs_root))
	dataset_rel = shlex.quote(f"./datasets/input/{dataset_name}")
	model_rel = shlex.quote(f"./output/{dataset_name}")
	iterations = shlex.quote(str(training_iterations))

	return (
		"set -e; "
		f"cd {fastgs_wsl}; "
		"FASTGS_PYTHON=\\${FASTGS_WSL_PYTHON:-}; "
		'if [ -z "\\$FASTGS_PYTHON" ] && [ -x "\\$HOME/anaconda3/envs/fastgs/bin/python" ]; '
		'then FASTGS_PYTHON="\\$HOME/anaconda3/envs/fastgs/bin/python"; fi; '
		"if [ -z \"\\$FASTGS_PYTHON\" ] && [ -x /home/logan/anaconda3/envs/fastgs/bin/python ]; "
		"then FASTGS_PYTHON=/home/logan/anaconda3/envs/fastgs/bin/python; fi; "
		'if [ -z "\\$FASTGS_PYTHON" ]; then FASTGS_PYTHON=\\$(command -v python || command -v python3 || true); fi; '
		'test -n "\\$FASTGS_PYTHON"; '
		'"\\$FASTGS_PYTHON" -c "import torch, torchvision, plyfile, tqdm"; '
		"CUDA_VISIBLE_DEVICES=0 "
		f"OAR_JOB_ID={dataset_name} "
		'"\\$FASTGS_PYTHON" train.py '
		f"-s {dataset_rel} "
		f"-m {model_rel} "
		f"--iterations {iterations} "
		"--eval --densification_interval 500 --optimizer_type default "
		f"--test_iterations {iterations} "
		f"--save_iterations {iterations} "
		f"--checkpoint_iterations {iterations} "
		"--highfeature_lr 0.0015 --dense 0.003 --mult 0.7; "
		"CUDA_VISIBLE_DEVICES=0 "
		'"\\$FASTGS_PYTHON" render.py '
		f"-s {dataset_rel} -m {model_rel} --skip_train"
	)


def prepare_fastgs_input_and_train(
	colmap_output_dir: Path,
	fastgs_root: Path,
	run_training: bool = True,
	training_iterations: int = DEFAULT_FASTGS_ITERATIONS,
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
	registered_image_count = _read_colmap_registered_image_count(images_bin_src)

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
	# FastGS expects COLMAP's points3D casing; keep lowercase aliases for downstream compatibility.
	shutil.copy2(points3d_bin_src, sparse0_dir / "points3D.bin")
	shutil.copy2(points3d_bin_src, sparse0_dir / "points3d.bin")
	shutil.copy2(points3d_ply_src, sparse0_dir / "points3D.ply")
	shutil.copy2(points3d_ply_src, sparse0_dir / "points3d.ply")

	if colmap_output_dir.exists():
		shutil.rmtree(colmap_output_dir)

	wsl_command = _build_wsl_training_command(
		fastgs_root=fastgs_root,
		dataset_name=dataset_name,
		training_iterations=training_iterations,
	)

	if run_training:
		subprocess.run(["wsl", "bash", "-lc", wsl_command], check=True)

	return PipelineResult(
		dataset_name=dataset_name,
		dataset_path=str(dataset_dir),
		model_path=str(fastgs_root / "output" / dataset_name),
		render_path=str(fastgs_root / "output" / dataset_name / "test" / f"ours_{training_iterations}" / "renders"),
		wsl_command=wsl_command,
		registered_image_count=registered_image_count,
	)
