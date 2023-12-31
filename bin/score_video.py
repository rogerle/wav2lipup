"""
这个主要处理判断视频是否同步，用训练的syncnet打分，对于偏离比较大的视频抛弃


"""
import argparse
from functools import partial
from pathlib import Path

import torch
from torch import multiprocessing
from tqdm import tqdm

from process_util.SyncnetScore import SyncnetScore


def main():
    args = parse_args()
    data_root = args.data_root
    checkpoint = args.checkpoint_path
    num_worker = args.num_worker
    batch_size = args.batch_size

    train_txt = data_root + '/train.txt'
    test_txt = data_root + '/test.txt'
    eval_txt = data_root + '/eval.txt'
    dir_list = get_dirList(train_txt)
    test_list = get_dirList(test_txt)
    eval_list = get_dirList(eval_txt)
    dir_list += eval_list
    dir_list += test_list
    Path(data_root + '/score.txt').write_text('')
    Path(data_root + '/bad_off.txt').write_text('')
    score_tools = SyncnetScore()
    proc_f = partial(score_tools.score_video, checkpoint=checkpoint, data_root=data_root, batch_size=batch_size)
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=num_worker)
    # ctx = multiprocessing.get_context('spawn')
    # pool = ctx.Pool(num_worker)
    prog_bar = tqdm(pool.imap(proc_f, dir_list), total=len(dir_list))
    results = []
    bad_offset_f = []
    for v_file, offset, conf in prog_bar:
        results.append('video:{} offset:{} conf:{}'.format(v_file, offset, conf))
        with open(data_root + '/score.txt', 'a') as fw:
            fw.write("{},{},{}\n".format(v_file, offset, conf))
        if offset < -1 or offset > 1:
            bad_offset_f.append(v_file)
            with open(data_root + '/bad_off.txt', 'a', encoding='utf-8') as fw:
                fw.write("{}\n".format(v_file))
        prog_bar.set_description('score file:{} offset:{}'.format(v_file, offset))
    torch.cuda.empty_cache()
    pool.close()
    pool.join()


def get_dirList(path):
    dir_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            dir_list.append(line)
    return dir_list


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="score for the video")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True, type=str)
    parser.add_argument('--checkpoint_path', help='Load he pre-trained ', required=True, )
    parser.add_argument('--num_worker', help='multiprocessor number', default=6, type=int)
    parser.add_argument('--batch_size', help='produce img batch', default=20, type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
