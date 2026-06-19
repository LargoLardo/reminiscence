#!/bin/bash
set -e

MODEL_DIR="./output/input1"
DATASET_ROOT="./datasets/input"
DATASET_DIR="$DATASET_ROOT"
IMAGES_REL="images"

if [ ! -f "$MODEL_DIR/cfg_args" ]; then
  echo "ERROR: No trained model config found in $MODEL_DIR."
  echo "Train first with train_render_input.sh or use a model directory that contains cfg_args."
  exit 1
fi

if [ -d "$DATASET_ROOT/input1" ] && [ -d "$DATASET_ROOT/input1/sparse/0" ]; then
  DATASET_DIR="$DATASET_ROOT/input1"
  IMAGES_REL="images"
elif [ -d "$DATASET_ROOT/input1/images" ] && [ ! -d "$DATASET_ROOT/sparse/0" ]; then
  DATASET_DIR="$DATASET_ROOT/input1"
  IMAGES_REL="images"
elif [ -d "$DATASET_ROOT/sparse/0" ] && [ -d "$DATASET_ROOT/input1/images" ]; then
  IMAGES_REL="input1/images"
fi

CUDA_VISIBLE_DEVICES=0 python render.py -s "$DATASET_DIR" -i "$IMAGES_REL" -m "$MODEL_DIR" --skip_train
