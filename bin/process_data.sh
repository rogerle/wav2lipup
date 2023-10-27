#!/bin/bash

TEMP=/tmp

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   python processData.py --data_root=../data \
                         --process_step=0 \
                         --gpu_num=1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   python processData.py --data_root=../data \
                         --process_step=1 \
                         --gpu_num=1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   python processData.py --data_root=../data \
                         --process_step=2 \
                         --gpu_num=1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
   python processData.py --data_root=../data \
                         --process_step=3 \
                         --gpu_num=1
fi