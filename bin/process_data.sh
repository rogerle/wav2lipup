#!/bin/bash

export TEMP=/tmp
stage=0
stop_stage=100
gpus=0
export IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
export IMAGEIO_USE_GPU=True

source ./parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  CUDA_VISIBLE_DEVICES=${gpus} python processData.py --data_root=../data \
                         --process_step=0  || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   CUDA_VISIBLE_DEVICES=${gpus} python processData.py --data_root=../data \
                         --preprocess_type=ASR \
                         --process_step=1 || exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   CUDA_VISIBLE_DEVICES=${gpus} python processData.py --data_root=../data \
                         --process_step=2 || exit -1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   CUDA_VISIBLE_DEVICES=${gpus} python processData.py --data_root=../data \
                         --process_step=3 || exit -1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   CUDA_VISIBLE_DEVICES=${gpus} python processData.py --data_root=../data \
                         --process_step=4 || exit -1
fi