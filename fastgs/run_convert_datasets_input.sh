#!/bin/bash
set -e

# Run this from the repository root: bash run_convert_datasets_input.sh
DATASET_DIR="datasets/input"
COLMAP_BIN="${COLMAP_BIN:-colmap}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: Dataset directory '$DATASET_DIR' not found."
  exit 1
fi

if [ ! -d "$DATASET_DIR/input" ]; then
  if [ -d "$DATASET_DIR/images" ]; then
    echo "INFO: '$DATASET_DIR/input' not found, creating it from '$DATASET_DIR/images'."
    mkdir -p "$DATASET_DIR/input"
    cp -n "$DATASET_DIR/images"/* "$DATASET_DIR/input/"
  else
    echo "ERROR: Expected either '$DATASET_DIR/input' or '$DATASET_DIR/images' to exist."
    exit 1
  fi
fi

if ! command -v "$COLMAP_BIN" >/dev/null 2>&1; then
  echo "WARNING: COLMAP executable '$COLMAP_BIN' not found on PATH."
  echo "Set COLMAP_BIN to the correct path or install COLMAP."
fi

python convert.py -s "$DATASET_DIR" --colmap_executable "$COLMAP_BIN"

echo "Conversion complete. Output path: $DATASET_DIR/sparse/0"