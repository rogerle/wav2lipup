#!/bin/bash

gpus=0

CUDA_VISIBLE_DEVICES=${gpus} python ../trains/wl_train.py \
                                    --data_root=../data/processed_data \
                                    --checkpoint_dir=../data/checkpoint \
                                    --syncnet_checkpoint_path=../data/syncnet_checkpoint/sync_checkpoint_step000161000.pth
