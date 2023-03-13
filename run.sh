#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira \
--img_aug \
--batch_size 64 \
--MLM \
--loss_names 'sdm+mlm+id' \
--lrscheduler 'cosine' \
--target_lr 0 \
--num_epoch 60