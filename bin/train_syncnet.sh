#!/bin/bash

gpus=0

CUDA_VISIBLE_DEVICES=${gpus} python ../trains/syncnet_train.py \
                             --data_root=../data/processed_data \
                             --checkpoint_dir=../data/syncnet_checkpoint \
                             --config_file=../configs/train_config_288.yaml \
                             --gpunum=1