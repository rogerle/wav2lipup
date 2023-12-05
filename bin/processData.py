"""
这个主要处理原始视频到相应的目录下
根目录的主要结构
./root
    ├── original_data       #这个是原始数据存放的目录
    ├── preProcessed_data   #切割成5s的存放目录
    ├── processed_data      #处理成训练所需的文件存放的目录


"""
import argparse
from functools import partial
from pathlib import Path, PurePath
from torch import multiprocessing
from tqdm import tqdm

from process_util.DataProcessor import DataProcessor
from process_util.PreProcessor import PreProcessor
from sklearn.model_selection import train_test_split


def orignal_process(inputdir):
    dirs = []
    root = Path(inputdir)
    # 把目录下的子目录都拿出来
    for dir in Path.rglob(root, '*'):
        if dir.is_dir():
            dirs.append(str(dir))
    # 重命名所有文件，把文件转换成6位数字的文件名，格式为000000.MP4
    temp_dir = inputdir + '/temp'
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    for i, dir in tqdm(enumerate(dirs), desc='process the original datasets:', total=len(dirs), unit='video'):
        j = 0
        files = []
        for f in Path.glob(Path(dir), '**/*.mp4'):
            j = j + 1
            newfilename = temp_dir + '/{0:04}.mp4'.format(j)
            files.append(newfilename)
            Path.rename(Path(str(f)), Path(newfilename))
        for nf in files:
            fname = dir + '/' + Path(nf).name
            Path.rename(Path(nf), Path(fname))

    Path(temp_dir).rmdir()


def preProcess(inputdir, outputdir, preprocess_type):
    processer = PreProcessor()
    if preprocess_type == 'Time':

        processer.videosPreProcessByTime(s_time=5,
                                         input_dir=inputdir,
                                         output_dir=outputdir,
                                         ext='mp4')
    else:
        processer.videosPreProcessByASR(input_dir=inputdir,
                                        output_dir=outputdir,
                                        ext='mp4')


def get_processed_files(inputdir, outputdir):
    files = []
    for f in Path.glob(Path(inputdir), '**/*.mp4'):
        if f.is_file():
            files.append(f.as_posix())
    files.sort()
    total_files = len(files)
    print('total files to processed:{}'.format(total_files))

    dones = get_processed_data(outputdir)
    done_files = []
    if dones is not None and len(dones) > 0:
        print('break point continue!')
        for done_file in dones:
            d_root = Path(done_file).parts[-2]
            d_name = Path(done_file).parts[-1]
            d_s = d_name.split('_')
            d_d = d_s[-3]
            d_f = d_s[-2] + '_' + d_s[-1] + ".mp4"
            d_full = inputdir + '/' + d_root + '/' + d_d + '/' + d_f
            done_files.append(d_full)
        done_bar = tqdm(enumerate(done_files), total=len(done_files), leave=False)
        for item in done_files:
            files.remove(item)
            done_bar.set_description('produce break point!{}'.format(item))
    return files


def process_data(inputdir, outputdir):
    dataProcessor = DataProcessor()
    results = []
    files = get_processed_files(inputdir, outputdir)
    proc_f = partial(dataProcessor.processVideoFile, processed_data_root=outputdir)

    num_p = int(multiprocessing.cpu_count()/2)
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_p)
    prog_bar = tqdm(pool.imap(proc_f,files), total=len(files))
    for result in prog_bar:
        results.append(result)
    pool.close()
    return results

def get_processed_data(processed_data_root):
    done_dir = []
    for done in Path.glob(Path(processed_data_root), '*/*'):
        if done.is_dir():
            done_dir.append(done)
    return done_dir


def train_file_write(inputdir):
    train_txt = inputdir + '/train.txt'
    eval_txt = inputdir + '/eval.txt'
    Path(train_txt).write_text('')
    Path(eval_txt).write_text('')
    result_list = []
    for line in Path.glob(Path(inputdir), '*/*'):
        if line.is_dir():
            dirs = line.parts
            input_line = str(dirs[-2] + '/' + dirs[-1])
            result_list.append(input_line)
    if len(result_list) < 14:
        test_result = eval_result = train_result = result_list
    else:
        train_result, test_result = train_test_split(result_list, test_size=0.15, random_state=42)
        test_result, eval_result = train_test_split(test_result, test_size=0.5, random_state=42)

    for file_name, data_set in zip(("train.txt", "test.txt", "eval.txt"), (train_result, test_result, eval_result)):
        with open(inputdir + '/' + file_name, 'w', encoding='utf-8') as fi:
            fi.write("\n".join(data_set))


def clear_data(inputdir):
    train_txt = inputdir + '/train.txt'
    test_txt = inputdir + '/test.txt'
    eval_txt = inputdir + '/eval.txt'
    train_list = get_list(train_txt)
    test_list = get_list(test_txt)
    eval_list = get_list(eval_txt)
    bad_list = []
    for line in tqdm(Path.glob(Path(inputdir), '*/*')):
        if line.is_dir():
            imgs = []
            for img in line.glob('**/*.jpg'):
                if img.is_file():
                    imgs.append(int(img.stem))
            if imgs is None or len(imgs) <=25 or len(imgs) < max(imgs):
                print('delete empty or bad video!{}'.format(line))
                dirs = line.parts
                bad_line = str(dirs[-2] + '/' + dirs[-1])
                bad_list.append(bad_line)

    train_list = clear_badv(train_list,bad_list)
    test_list = clear_badv(test_list,bad_list)
    eval_list = clear_badv(eval_list,bad_list)


    with open(inputdir + '/bad_v.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(bad_list))
    with open(inputdir + '/train.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(train_list))
    with open(inputdir + '/test.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(test_list))
    with open(inputdir + '/eval.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(eval_list))

def sync_data(inputdir):
    train_txt = inputdir + '/train.txt'
    test_txt = inputdir + '/test.txt'
    eval_txt = inputdir + '/eval.txt'
    exclude_txt = inputdir + '/bad_off.txt'
    train_list = get_list(train_txt)
    test_list = get_list(test_txt)
    eval_list = get_list(eval_txt)
    exclude_list = get_list(exclude_txt)
    train_list = clear_badv(train_list, exclude_list)
    test_list = clear_badv(test_list, exclude_list)
    eval_list = clear_badv(eval_list, exclude_list)

    with open(inputdir + '/train.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(train_list))

    with open(inputdir + '/test.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(test_list))

    with open(inputdir + '/eval.txt', 'w', encoding='utf-8') as fw:
        fw.write("\n".join(eval_list))


def get_list(inputText):
    list = []
    with open(inputText, 'r') as f:
        for line in f:
            line = line.strip()
            list.append(line)
    return list


def clear_badv(all_list, exclude_list):
    for item in exclude_list:
        if item in all_list:
            all_list.remove(item)

    return all_list


def main():
    args = parse_args()
    data_root = args.data_root
    p_step = args.process_step
    preprocess_type = args.preprocess_type
    original_dir = data_root + '/original_data'
    preProcess_dir = data_root + '/preProcessed_data'
    process_dir = data_root + '/processed_data'

    if p_step == 0:
        print("produce the step {}".format(p_step))
        orignal_process(original_dir)
    elif p_step == 1:
        print("produce the step {}".format(p_step))
        preProcess(original_dir, preProcess_dir, preprocess_type)
    elif p_step == 2:
        print("produce the step {}".format(p_step))
        process_data(preProcess_dir, process_dir)
    elif p_step == 3:
        print("produce the step {}".format(p_step))
        train_file_write(process_dir)
        clear_data(process_dir)
    elif p_step == 4:
        print("produce the step {}".format(p_step))
        sync_data(process_dir)
    else:
        print('wrong step number, finished!')


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="process the datasets for wav2lip")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True, type=str)
    parser.add_argument("--preprocess_type", help='ASR or time split', default='ASR', type=str)
    parser.add_argument("--process_step", help='process data\'s step 1 orig,2.pre 3.pro 4.write file', default=0,
                        type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
