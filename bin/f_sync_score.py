
import argparse
from pathlib import Path
def process_file_list(data_root):
    filelist=[]
    for file in Path(data_root).glob('*/*.mp4'):
        if file.is_file():
            filelist.append(str(file))
    with open(data_root+'/'+'process.txt','w') as f:
        f.write('\n'.join(filelist))
    return filelist


def continues_file_list(data_root):
    full_txt = data_root+'/'+'process.txt'
    full_list = get_list(full_txt)
    processed_txt = data_root+'/'+'processed.txt'
    processed_list = get_list(processed_txt)

    p_list = clear_pv(full_list,processed_list)

    return p_list


def get_list(inputText):
    list = []
    with open(inputText, 'r') as f:
        for line in f:
            line = line.strip()
            list.append(line)
    return list
def clear_pv(all_list, exclude_list):
    for item in exclude_list:
        if item in all_list:
            all_list.remove(item)

    return all_list
def main():
    args = parse_args()
    data_root = args.data_root
    break_p = args.break_point

    #处理的视频文件做成list文本
    if break_p == 0:
        process_list = process_file_list(data_root)
    else:
        process_list = continues_file_list(data_root)


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="score for the video")
    parser.add_argument("--video_root", help='Root folder of the dataset', required=True, type=str)
    parser.add_argument('--init_model', help='Load the pre-trained ', required=True, default='../data/pre-models/syncnet_v2.model' )
    parser.add_argument('--num_worker', help='multiprocessor number', default=6, type=int)
    parser.add_argument('--batch_size', help='produce img batch', default=20, type=int)
    parser.add_argument('--beark_point', help='score continus from beak point', default=0, type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()