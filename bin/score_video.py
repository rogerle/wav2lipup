"""
这个主要处理判断视频是否同步，用训练的syncnet打分，对于偏离比较大的视频抛弃


"""
import os
import argparse
from pathlib import Path, PurePath

from tqdm import tqdm

from process_util.DataProcessor import DataProcessor
from process_util.PreProcessor import PreProcessor
from sklearn.model_selection import train_test_split

from process_util.SyncnetScore import SyncnetScore


def main():
    args = parse_args()
    data_root = args.data_root
    checkpoint = args.checkpoint_path

    score_tools= SyncnetScore(data_root,8,checkpoint)
    score_tools.score_video()





def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="score for the video")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Load he pre-trained ', required=True,)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
