#!/bin/bash

export TEMP=/tmp
stage=0
stop_stage=100
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
export IMAGEIO_USE_GPU=True

source ./parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   python score_video.py --data_root=../data/processed_data \
                         --checkpoint_path=../data/syncnet_checkpoint/sync_checkpoint_step000013000.pth \
                         --filter_score=0.69
fi