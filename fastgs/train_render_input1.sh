#!/bin/bash
set -e

# Train on datasets/input/input1, just like playroom
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=input1 python train.py -s ./datasets/input/input1 --eval --densification_interval 500 --optimizer_type default --test_iterations 30000 --highfeature_lr 0.0015 --dense 0.003 --mult 0.7

# Render the trained model
CUDA_VISIBLE_DEVICES=0 python render.py -s ./datasets/input/input1 -m ./output/input1 --skip_train