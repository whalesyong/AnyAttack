#!/bin/bash

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0


python finetune_robo.py \
    --acc_data_dir ~/datasets/roboflow_acc/Accident-and-Non-accident-label-Image-Dataset-13 \
    --bdd_data_dir ~/datasets/bdd100k_im/bdd100k/images/10k \
    --out_dir 'ckpts_robo' \
    --ckpt './release/checkpoints/coco_bi.pt' \
    --epoch 30 \
    --k_aug 6 \
    --batch_size 12 \
    --amp \
    --test


