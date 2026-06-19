from __future__ import annotations

import os
import re
import shutil
import shlex
import subprocess
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FASTGS_ITERATIONS = 30000
ProgressCallback = Callable[[str, int, int], None]
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_FRACTION_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")


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
	densify_until = iterations

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
		f"--iterations {iterations} "
		"--eval --densification_interval 100 --optimizer_type default "
		f"--test_iterations {iterations} "
		f"--save_iterations {iterations} "
		f"--checkpoint_iterations {iterations} "
		f"--densify_until_iter {densify_until} "
		"--loss_thresh 0.06 "
		"--highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005; "
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
	process = subprocess.Popen(
		command,
		cwd=cwd,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=0,
	)
	assert process.stdout is not None
	fragment = ""
	logged_progress: dict[str, int] = {}
	last_output_at = time.monotonic()

	while True:
		char = process.stdout.read(1)
		if char == "" and process.poll() is not None:
			break
		if not char:
			if time.monotonic() - last_output_at > 30:
				print("Waiting for process output...", flush=True)
				last_output_at = time.monotonic()
			continue
		last_output_at = time.monotonic()
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
	dataset_rel = f"./datasets/input/{dataset_name}"
	model_rel = f"./output/{dataset_name}"
	common = ["-s", dataset_rel, "-m", model_rel]
	train_command = [
		python,
		"train.py",
		*common,
		"--iterations",
		str(training_iterations),
		"--eval",
		"--densification_interval",
		"100",
		"--optimizer_type",
		"default",
		"--test_iterations",
		str(training_iterations),
		"--save_iterations",
		str(training_iterations),
		"--checkpoint_iterations",
		str(training_iterations),
		"--densify_until_iter",
		str(training_iterations),
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
	render_command = [python, "render.py", *common, "--skip_test"]
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
