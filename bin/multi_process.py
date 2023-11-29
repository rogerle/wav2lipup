import argparse
import shutil
from pathlib import Path, PurePath
import multiprocessing
from multiprocessing import Pool

from tqdm import tqdm
from functools import partial
from process_util.DataProcessor import DataProcessor
from process_util.PreProcessor import PreProcessor
from sklearn.model_selection import train_test_split

dataProcessor = DataProcessor()


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

def get_processed_data(processed_data_root):
    done_dir = []
    for done in Path.glob(Path(processed_data_root), '*/*'):
        if done.is_dir():
            done_dir.append(done)
    return done_dir

def process_data(inputdir, outputdir):
    files = get_processed_files(inputdir, outputdir)

    proc_f = partial(dataProcessor.processVideoFile,processed_data_root=outputdir)

    num_p = 4
    pool = multiprocessing.Pool(num_p)
    pool.map(proc_f,files)
    pool.close()


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="process the datasets for wav2lip")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True, type=str)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    data_root = args.data_root
    original_dir = data_root + '/original_data'
    preProcess_dir = data_root + '/preProcessed_data'
    process_dir = data_root + '/processed_data'

    process_data(preProcess_dir, process_dir)

if __name__ == '__main__':
    main()