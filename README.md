# Reminiscence

Reminiscence turns an ordinary phone video into an explorable 3D memory: capture a scene on iPhone, reconstruct it into a Gaussian splat, import it into Unity, and walk through it in VR on a Meta Quest/Oculus headset. The cool part is that a flat clip becomes a spatial asset you can revisit, place in a scene, and experience at human scale.

Technically, the project connects a Swift capture app, a FastAPI backend, COLMAP, FastGS, and Unity/OpenXR into one local reconstruction pipeline.

## End-to-end pipeline

1. Build and run the Swift app from `swift-app/` on an iPhone.
2. The iPhone uploads a selected video to the backend through ngrok.
3. `backend/main.py` receives `POST /api/v1/moments`.
4. `prepare_colmap_windows.py` extracts frames, runs COLMAP, and creates sparse reconstruction output.
5. `backend/rendering_pipeline.py` copies the COLMAP result into `fastgs/datasets/input/input_N`.
6. FastGS trains in WSL and writes `fastgs/output/input_N/point_cloud/iteration_5000/point_cloud.ply`.
7. `backend/unity_splat_transfer.py` copies the newest splat into the configured Unity project.
8. Unity batch mode imports the PLY into `Assets/GaussianAssets/input_N`.
9. Open the Unity project, add the generated prefab to a VR scene, and build/run it on the Quest.

## Repository layout

```text
backend/
  main.py                       FastAPI upload endpoint and pipeline entrypoint
  rendering_pipeline.py         COLMAP-to-FastGS dataset prep and WSL FastGS launch
  unity_splat_transfer.py       FastGS PLY to Unity asset transfer/import

fastgs/                         FastGS training code and WSL/conda environment
prepare_colmap_windows.py       Windows COLMAP preparation script
swift-app/                      Xcode iOS app that uploads selected media
unity_renderer/                 Unity VR renderer project/template
cleanup_pipeline_artifacts.py   Safe cleanup script for generated artifacts
requirements.txt                Python backend dependencies
```

## Requirements

Windows reconstruction machine:

- Windows with an NVIDIA GPU.
- Python 3.10+ for the FastAPI backend.
- WSL with Ubuntu or another Linux distro.
- CUDA drivers working inside WSL.
- Anaconda or Miniconda inside WSL for the FastGS environment.
- COLMAP for Windows.
- FFmpeg on PATH for video input.
- Unity Hub with Unity `6000.4.4f1` or the version in the target project's `ProjectSettings/ProjectVersion.txt`.
- Unity Android Build Support, SDK/NDK tools, and OpenJDK.
- The Gaussian Splatting Unity package downloaded locally.
- ngrok.

Apple/iOS capture machine:

- macOS with Xcode.
- iPhone or iOS simulator capable of running the Swift app.
- An Apple developer team selected in Xcode signing settings.

Quest/Oculus device:

- Developer Mode enabled in the Meta/Oculus mobile app.
- USB debugging allowed on the headset.
- A USB cable or another working deployment path from Unity.

## Install checklist

Install these before running the full phone-to-headset pipeline:

- Windows Python 3.10+ for `backend/main.py`.
- Git for cloning the repo and submodules.
- WSL/Ubuntu for FastGS training.
- NVIDIA GPU drivers with WSL CUDA support.
- Anaconda or Miniconda inside WSL.
- COLMAP for Windows.
- FFmpeg for video frame extraction.
- ngrok for forwarding the local backend to the iPhone.
- Unity Hub and the Unity editor version used by the target project.
- Unity Android Build Support, SDK/NDK tools, and OpenJDK.
- The Unity Gaussian Splatting package used by the target Unity project.
- Xcode on macOS for the Swift app.
- Meta Quest/Oculus developer mode and USB debugging.

## Clone and dependencies

From the repo root on Windows:

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Install WSL and conda for FastGS

Install Ubuntu in WSL from an administrator PowerShell if it is not already installed:

```powershell
wsl --install -d Ubuntu
```

Open Ubuntu/WSL and install basic Linux tools:

```bash
sudo apt update
sudo apt install -y wget git build-essential
```

Install Miniconda into `~/anaconda3`. The folder name is intentional: the backend automatically checks `~/anaconda3/envs/fastgs/bin/python`.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -u -p ~/anaconda3
~/anaconda3/bin/conda init bash
exec bash
conda --version
```

Full Anaconda also works, as long as the FastGS environment ends up under `~/anaconda3` or you set `FASTGS_WSL_PYTHON`.

FastGS is not installed from `requirements.txt`. It uses its own WSL conda environment because it depends on CUDA-specific PyTorch builds and local CUDA extension submodules:

```bash
cd /mnt/c/Users/login/reminiscence/reminiscence/fastgs
conda env create -f environment.yml
conda activate fastgs
python -c "import torch, torchvision, plyfile, tqdm; print(torch.cuda.is_available())"
```

The final command should print `True`. If it prints `False`, FastGS will not train on the GPU until the NVIDIA/WSL CUDA setup is fixed.

If the backend cannot find the FastGS Python executable automatically, set this in Windows PowerShell before starting the backend:

```powershell
$env:FASTGS_WSL_PYTHON="/home/<your-wsl-user>/anaconda3/envs/fastgs/bin/python"
```

## Install external tools

Install COLMAP for Windows. `prepare_colmap_windows.py` checks these paths automatically:

```text
C:\Users\login\Downloads\colmap-x64-windows-cuda\COLMAP.bat
C:\Users\login\Downloads\colmap-x64-windows-cuda\bin\colmap.exe
C:\Program Files\COLMAP\COLMAP.bat
C:\Program Files\COLMAP\bin\colmap.exe
```

To verify COLMAP:

```powershell
"C:\Users\login\Downloads\colmap-x64-windows-cuda\COLMAP.bat" gui
```

Install FFmpeg for video frame extraction:

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
```

Install ngrok and make sure this works:

```powershell
ngrok version
```

Install Unity through Unity Hub. For the current renderer project, use Unity `6000.4.4f1` or the version listed in the target project's `ProjectSettings/ProjectVersion.txt`, and include Android Build Support, SDK/NDK tools, and OpenJDK from Unity Hub's module installer.

## Configure the Unity target project

The backend imports splats into the project configured at the top of `backend/unity_splat_transfer.py`:

```python
TARGET_UNITY_PROJECT = Path(r"C:\Users\login\SplatTest")
```

Change only that value when you want to use a different Unity project. The target project must have:

- The Gaussian Splatting Unity package in `Packages/manifest.json`.
- XR Interaction Toolkit, XR Management, OpenXR, Input System, and URP.
- `Assets/Editor/BatchGaussianSplatImporter.cs`.
- `Assets/AutoFitObjectToCamera.cs`.
- A scene with an XR Origin or another camera rig.

The current target package reference is local:

```json
"org.nesnausk.gaussian-splatting": "file:C:/Users/login/Downloads/UnityGaussianSplatting-main/UnityGaussianSplatting-main/package"
```

That folder must exist on the Windows machine running Unity. If Unity is installed somewhere nonstandard, set:

```powershell
$env:UNITY_EDITOR="C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Unity.exe"
```

Close the target Unity project before running the upload pipeline. The backend opens Unity in batch mode to import the asset, and Unity cannot reliably open the same project twice.

## Quest/Oculus rendering settings

For standalone Quest builds, Unity uses the Android quality/render pipeline settings. In the current project, `PC_Renderer.asset` has `GaussianSplatURPFeature`, while `Mobile_Renderer.asset` does not. Before building for Quest, make sure the renderer used by Android includes the Gaussian Splat URP feature.

In Unity:

1. Open `C:\Users\login\SplatTest` or your configured target project.
2. Go to `Edit > Project Settings > Quality`.
3. Check the default quality for Android.
4. Open the URP renderer used by that quality level.
5. Add `GaussianSplatURPFeature` to that renderer if it is missing.

If the splat imports successfully but is invisible in the headset, this renderer feature is the first thing to check.

## Start the backend

Use port `8000`. `README1` used `800`, but the Swift/ngrok setup and backend notes expect `8000`.

```powershell
cd C:\Users\login\reminiscence\reminiscence\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check that it is alive:

```powershell
curl.exe http://localhost:8000/
```

Expected response:

```json
{"status":"server is running"}
```

## Start ngrok

In a second terminal:

```powershell
ngrok http 8000
```

Copy the public HTTPS forwarding URL and update the upload URL in `swift-app/ReminiscienceV2/ContentView.swift`:

```swift
let url = URL(string: "https://<your-ngrok-domain>/api/v1/moments")!
```

Every time ngrok gives you a new domain, update this string and rebuild the app.

## Build and run the Swift app

On the Mac:

1. Open `swift-app/ReminiscienceV2.xcodeproj`.
2. Select the `ReminiscienceV2` scheme.
3. Select your iPhone as the run destination.
4. In signing settings, choose your development team.
5. Press `Command+R`.
6. In the app, tap `CHOOSE MOMENTS` and select a video.

The current backend saves incoming media as an `.mp4`, so use video for the end-to-end pipeline. The picker can show images, but image uploads are not a complete path yet.

The backend runs COLMAP, FastGS, and Unity import synchronously during the upload request. This can take several minutes.

## What the pipeline creates

For a successful upload, expect these generated paths:

```text
backend/uploads/<moment-id>.mp4
fastgs/datasets/input/input_N/
fastgs/output/input_N/point_cloud/iteration_5000/point_cloud.ply
C:\Users\login\SplatTest\Assets\GaussianAssets\input_N\input_N.ply
C:\Users\login\SplatTest\Assets\GaussianAssets\input_N\input_N.asset
C:\Users\login\SplatTest\Assets\GaussianAssets\input_N\input_N_Renderer.prefab
C:\Users\login\SplatTest\Assets\GaussianAssets\input_N\LatestGaussianSplat.prefab
C:\Users\login\SplatTest\Logs\gaussian_import_input_N.log
```

The backend response also includes the Unity asset, prefab, and log paths.

## Manually import the latest FastGS output

If FastGS already produced `fastgs/output/input_N`, you can rerun only the Unity transfer/import step:

```powershell
cd C:\Users\login\reminiscence\reminiscence
python backend\unity_splat_transfer.py
```

To import a specific output:

```powershell
python backend\unity_splat_transfer.py --input-name input_16
```

To copy the PLY without running the Unity importer:

```powershell
python backend\unity_splat_transfer.py --input-name input_16 --copy-only
```

## Put the splat in a VR scene

After the import finishes:

1. Open the target Unity project from Unity Hub.
2. Wait for Unity to compile and import assets.
3. Open your VR scene, or open the XR Interaction Toolkit demo scene.
4. Drag `Assets/GaussianAssets/input_N/LatestGaussianSplat.prefab` into the Hierarchy.
5. Make sure the scene has an XR Origin and a camera tagged `MainCamera`.
6. Press Play in Unity to test, or build to the headset.

The generated prefab is a reusable scene object. It does not appear in the scene by itself unless you drag it into the Hierarchy or have a scene script that instantiates it. `PlaceSplatInFrontOfCamera` positions it in front of the viewer camera at runtime.

## Build to Quest/Oculus

In Unity:

1. Connect the headset and allow USB debugging.
2. Go to `File > Build Profiles` or `File > Build Settings`.
3. Select `Android`.
4. Click `Switch Platform`.
5. Go to `Edit > Project Settings > XR Plug-in Management`.
6. Under Android, enable `OpenXR`.
7. In OpenXR settings, enable the relevant Meta/Oculus controller profile.
8. Confirm the Android URP renderer includes `GaussianSplatURPFeature`.
9. Add `LatestGaussianSplat.prefab` to the active scene.
10. Click `Build And Run`.

For PC VR through Quest Link, you can also use Play Mode with the Quest connected through Link, but standalone Quest builds require Android.

## Useful settings

Training iterations are set in `backend/rendering_pipeline.py`:

```python
DEFAULT_FASTGS_ITERATIONS = 5000
```

Video frame sampling is set in `backend/main.py`:

```python
"--fps", "5",
```

Unity import quality is set in `backend/unity_splat_transfer.py`:

```python
DEFAULT_QUALITY = "Medium"
```

Higher values can improve quality but increase import time, memory use, and headset load.

## Cleanup

Preview generated artifacts that can be removed:

```powershell
python cleanup_pipeline_artifacts.py
```

Actually delete old generated outputs while keeping the latest numbered output in each generated root:

```powershell
python cleanup_pipeline_artifacts.py --execute --keep-latest 1
```

The cleanup script targets generated uploads, COLMAP output, FastGS datasets, FastGS outputs, Unity imported Gaussian assets, Unity import logs, and Python caches. It does not change pipeline source code.

## Troubleshooting

If the Swift app cannot upload:

- Confirm the backend is running on port `8000`.
- Confirm ngrok is forwarding to `http://localhost:8000`.
- Confirm `ContentView.swift` uses the current ngrok HTTPS URL.

If COLMAP fails:

- Confirm COLMAP is installed in a path checked by `prepare_colmap_windows.py`.
- Confirm FFmpeg is installed and available with `ffmpeg -version`.
- Use sharper, slower video with more side-to-side camera motion and visible texture.

If FastGS fails:

- In WSL, run `nvidia-smi`.
- In the FastGS conda env, run `python -c "import torch; print(torch.cuda.is_available())"`.
- Set `FASTGS_WSL_PYTHON` if the backend is launching the wrong Python.

If Unity import fails:

- Close the target Unity project before processing.
- Check `C:\Users\login\SplatTest\Logs\gaussian_import_input_N.log`.
- Confirm the Gaussian Splatting package path in `Packages/manifest.json` exists.
- Confirm `BatchGaussianSplatImporter.cs` is in `Assets/Editor`.

If the asset exists but does not render in the Quest:

- Confirm the prefab is in the active scene Hierarchy.
- Confirm the XR camera exists and is tagged `MainCamera`.
- Confirm the Android URP renderer has `GaussianSplatURPFeature`.
- Try a smaller scale on `PlaceSplatInFrontOfCamera` if the splat appears too large or too close.
