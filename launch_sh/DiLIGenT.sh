#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

python eval_pbr.py \
  --task_name DiLiGenT \
  --data_root data/DiLiGenT/pmsData/ \
  --num_images 2