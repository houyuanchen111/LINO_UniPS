#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

python eval_pbr.py \
  --task_name Real \
  --data_root data/LUCES/data/ \
  --num_images 1