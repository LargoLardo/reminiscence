FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FASTGS_NATIVE=1 \
    FASTGS_PYTHON=/opt/conda/bin/python \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" \
    REMINISCENCE_ASYNC_JOBS=1 \
    REMINISCENCE_UNITY_IMPORT=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 colmap ffmpeg git ninja-build wget \
    && rm -rf /var/lib/apt/lists/* && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir \
      torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
      --extra-index-url https://download.pytorch.org/whl/cu116 && \
    python -m pip install --no-cache-dir numpy==1.26.4 plyfile==0.9 tqdm websockets

COPY . .
RUN python -m pip install --no-build-isolation --no-cache-dir \
    ./fastgs/submodules/diff-gaussian-rasterization_fastgs \
    ./fastgs/submodules/simple-knn \
    ./fastgs/submodules/fused-ssim

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
