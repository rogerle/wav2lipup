#!/bin/bash

gpus=0

CUDA_VISIBLE_DEVICES=${gpus} python inference.py \
                                    --esrgan_checkpoint=../data/pretrain_model/RealESRGAN_x2plus.pth \
                                    --gfpgan_checkpoint=../data/pretrain_model/GFPGANv1.3.pth \
                                    --segmentation_path=../data/pretrain_model/segments.pth \
                                    --checkpoint_path=../data/checkpoint/checkpoint_step000339000.pth \
                                    --face=../data/temp/self-1.mp4 \
                                    --bg_tile=0 \
                                    --audio=../data/temp/audio-1.wav \
                                    --outfile=../data/temp/output.mp4