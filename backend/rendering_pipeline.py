from __future__ import annotations

import os
import re
import shutil
import shlex
import subprocess
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FASTGS_ITERATIONS = 30000
ProgressCallback = Callable[[str, int, int], None]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_FRACTION_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
FASTGS_STATIC_TRAIN_ARGS = [
	"--eval",
	"--densification_interval",
	"100",
	"--optimizer_type",
	"default",
	"--loss_thresh",
	"0.06",
	"--highfeature_lr",
	"0.0015",
	"--dense",
	"0.003",
	"--mult",
	"0.7",
	"--grad_abs_thresh",
	"0.0005",
]


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


def run_colmap_reconstruction(video_path: Path, output_dir: Path, script_path: Path, fps: int = 5) -> None:
	env = os.environ.copy()
	env["COLMAP_USE_GPU"] = "0"
	env["QT_QPA_PLATFORM"] = "offscreen"
	subprocess.run(
		[
			sys.executable,
			str(script_path),
			str(video_path),
			str(output_dir),
			"--fps",
			str(fps),
			"--overwrite",
			"--export-ply",
		],
		check=True,
		env=env,
	)


def _first_existing_file(paths: tuple[Path, ...], message: str) -> Path:
	for path in paths:
		if path.is_file():
			return path
	raise FileNotFoundError(message)


def _copy_colmap_scene(colmap_output_dir: Path, dataset_dir: Path) -> int:
	images_src = colmap_output_dir / "images"
	cameras_src = colmap_output_dir / "sparse" / "0" / "cameras.bin"
	images_bin_src = colmap_output_dir / "sparse" / "0" / "images.bin"
	points3d_bin_src = colmap_output_dir / "sparse" / "0" / "points3D.bin"
	points3d_ply_src = _first_existing_file(
		(colmap_output_dir / "sparse_points.ply", colmap_output_dir / "points3d.ply"),
		"Missing sparse PLY output. Expected sparse_points.ply or points3d.ply.",
	)

	if not images_src.is_dir():
		raise FileNotFoundError(f"Required image directory is missing: {images_src}")
	for path in (cameras_src, images_bin_src, points3d_bin_src):
		_require_file(path)

	sparse0_dir = dataset_dir / "sparse" / "0"
	sparse0_dir.mkdir(parents=True, exist_ok=False)
	shutil.copytree(images_src, dataset_dir / "images", dirs_exist_ok=True)
	for source, names in (
		(cameras_src, ("cameras.bin",)),
		(images_bin_src, ("images.bin",)),
		(points3d_bin_src, ("points3D.bin", "points3d.bin")),
		(points3d_ply_src, ("points3D.ply", "points3d.ply")),
	):
		for name in names:
			shutil.copy2(source, sparse0_dir / name)

	return _read_colmap_registered_image_count(images_bin_src)


def _windows_to_wsl_path(path: Path) -> str:
	path = path.resolve()

	if path.drive:
		drive = path.drive.rstrip(":").lower()
		rest = "/".join(path.parts[1:])
		return f"/mnt/{drive}/{rest}"

	return path.as_posix()


def _fastgs_paths(dataset_name: str) -> tuple[str, str]:
	return f"./datasets/input/{dataset_name}", f"./output/{dataset_name}"


def _fastgs_train_args(training_iterations: int) -> list[str]:
	iterations = str(training_iterations)
	return [
		"--iterations",
		iterations,
		*FASTGS_STATIC_TRAIN_ARGS,
		"--test_iterations",
		iterations,
		"--save_iterations",
		iterations,
		"--checkpoint_iterations",
		iterations,
		"--densify_until_iter",
		iterations,
	]


def _fastgs_native_commands(
	python: str,
	dataset_name: str,
	training_iterations: int,
) -> tuple[list[str], list[str]]:
	dataset_rel, model_rel = _fastgs_paths(dataset_name)
	common = ["-s", dataset_rel, "-m", model_rel]
	return (
		[python, "train.py", *common, *_fastgs_train_args(training_iterations)],
		[python, "render.py", *common, "--skip_test"],
	)


def _build_wsl_training_command(
	fastgs_root: Path,
	dataset_name: str,
	training_iterations: int = DEFAULT_FASTGS_ITERATIONS,
) -> str:
	fastgs_wsl = shlex.quote(_windows_to_wsl_path(fastgs_root))
	dataset_rel, model_rel = (shlex.quote(value) for value in _fastgs_paths(dataset_name))
	train_args = " ".join(shlex.quote(arg) for arg in _fastgs_train_args(training_iterations))

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
		"PYTHONUNBUFFERED=1 "
		f"OAR_JOB_ID={dataset_name} "
		'"\\$FASTGS_PYTHON" train.py '
		f"-s {dataset_rel} "
		f"-m {model_rel} "
		f"{train_args}; "
		"CUDA_VISIBLE_DEVICES=0 "
		"PYTHONUNBUFFERED=1 "
		'"\\$FASTGS_PYTHON" render.py '
		f"-s {dataset_rel} -m {model_rel} --skip_test"
	)


def _handle_process_output(
	fragment: str,
	progress_callback: ProgressCallback | None,
	logged_progress: dict[str, int],
) -> None:
	clean = ANSI_ESCAPE_RE.sub("", fragment).strip()
	if not clean:
		return

	stage = ""
	label = ""
	if "Training progress" in clean:
		stage = "training"
		label = "Training progress"
	elif "Rendering progress" in clean:
		stage = "rendering"
		label = "Rendering progress"

	if stage:
		match = PROGRESS_FRACTION_RE.search(clean)
		if match:
			current = int(match.group("current"))
			total = int(match.group("total"))
			if total > 0:
				if progress_callback:
					progress_callback(stage, current, total)
				percent = int(current * 100 / total)
				should_log = current == total or percent >= logged_progress.get(stage, -5) + 5
				if should_log:
					logged_progress[stage] = percent
					print(f"{label}: {percent}% ({current}/{total})", flush=True)
			return

	print(clean, flush=True)


def _run_streamed_subprocess(
	command: list[str],
	cwd: Path,
	env: dict[str, str] | None = None,
	progress_callback: ProgressCallback | None = None,
) -> None:
	with subprocess.Popen(
		command,
		cwd=cwd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=0,
	) as process:
		assert process.stdout is not None
		fragment = ""
		logged_progress: dict[str, int] = {}

		for char in iter(lambda: process.stdout.read(1), ""):
			if char in "\r\n":
				_handle_process_output(fragment, progress_callback, logged_progress)
				fragment = ""
			else:
				fragment += char

		if fragment:
			_handle_process_output(fragment, progress_callback, logged_progress)

		returncode = process.wait()
	if returncode:
		raise subprocess.CalledProcessError(returncode, command)


def _run_native_training(
	fastgs_root: Path,
	dataset_name: str,
	training_iterations: int,
	progress_callback: ProgressCallback | None = None,
) -> str:
	python = os.environ.get("FASTGS_PYTHON", "python")
	train_command, render_command = _fastgs_native_commands(python, dataset_name, training_iterations)
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
	env["OAR_JOB_ID"] = dataset_name
	env["PYTHONUNBUFFERED"] = "1"

	_run_streamed_subprocess(train_command, cwd=fastgs_root, env=env, progress_callback=progress_callback)
	_run_streamed_subprocess(render_command, cwd=fastgs_root, env=env, progress_callback=progress_callback)
	return shlex.join(train_command) + " && " + shlex.join(render_command)


def prepare_fastgs_input_and_train(
	colmap_output_dir: Path,
	fastgs_root: Path,
	run_training: bool = True,
	training_iterations: int = DEFAULT_FASTGS_ITERATIONS,
	progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
	colmap_output_dir = colmap_output_dir.resolve()
	fastgs_root = fastgs_root.resolve()

	dataset_root = fastgs_root / "datasets" / "input"
	dataset_root.mkdir(parents=True, exist_ok=True)

	input_idx = _next_input_index(dataset_root)
	dataset_name = f"input_{input_idx}"

	dataset_dir = dataset_root / dataset_name
	registered_image_count = _copy_colmap_scene(colmap_output_dir, dataset_dir)
	shutil.rmtree(colmap_output_dir, ignore_errors=True)

	if os.name == "nt" and os.environ.get("FASTGS_NATIVE", "").lower() not in {"1", "true", "yes"}:
		wsl_command = _build_wsl_training_command(
			fastgs_root=fastgs_root,
			dataset_name=dataset_name,
			training_iterations=training_iterations,
		)
		if run_training:
			_run_streamed_subprocess(
				["wsl", "bash", "-lc", wsl_command],
				cwd=fastgs_root,
				progress_callback=progress_callback,
			)
	else:
		wsl_command = "native FastGS execution"
		if run_training:
			wsl_command = _run_native_training(
				fastgs_root=fastgs_root,
				dataset_name=dataset_name,
				training_iterations=training_iterations,
				progress_callback=progress_callback,
			)

	return PipelineResult(
		dataset_name=dataset_name,
		dataset_path=str(dataset_dir),
		model_path=str(fastgs_root / "output" / dataset_name),
		render_path=str(fastgs_root / "output" / dataset_name / "train" / f"ours_{training_iterations}" / "renders"),
		wsl_command=wsl_command,
		registered_image_count=registered_image_count,
	)
