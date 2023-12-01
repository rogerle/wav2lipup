#!/bin/bash

export TEMP=/tmp
stage=0
stop_stage=100
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
export IMAGEIO_USE_GPU=True

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

source ./parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   python score_video.py --data_root=../data/processed_data \
                         --checkpoint_path=../data/syncnet_checkpoint/sync_checkpoint_step000370000.pth \
                         --num_worker=1
fi