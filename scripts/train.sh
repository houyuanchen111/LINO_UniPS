#!/bin/bash

export CUDA_VISIBLE_DEVICES=0  


mkdir -p checkpoints
mkdir -p logs


python train.py \
    --data_root "/share/project/public_data/objects/PS_v3_render/level_1/" \ # change to your own data root
    --low_normal True \
    --num_images 6 \
    --pixel_samples 2048 \
    --depth 4 \
    --canonical_resolution 256 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --max_epochs 100 \
    --min_lr 1e-6 \
    --step_size 10 \
    --gamma 0.8 \
    --patience 15 \
    --devices 1 \
    --num_workers 4 \
    --seed 42 \
    --save_dir "checkpoints" \
    --val_check_interval 1.0 \
    --log_every_n_steps 50

