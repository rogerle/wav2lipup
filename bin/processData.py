"""
这个主要处理原始视频到相应的目录下
根目录的主要结构
./root
    ├── original_data       #这个是原始数据存放的目录
    ├── preProcessed_data   #切割成5s的存放目录
    ├── processed_data      #处理成训练所需的文件存放的目录


"""
import argparse
from pathlib import Path

from tqdm import tqdm

from process_util.DataProcessor import DataProcessor
from process_util.PreProcessor import PreProcessor


def orignal_process(inputdir):
    dirs = []
    root = Path(inputdir)
    #把目录下的子目录都拿出来
    for dir in Path.rglob(root,'*'):
        if dir.is_dir():
            dirs.append(str(dir))
    #重命名所有文件，把文件转换成6位数字的文件名，格式为000000.MP4
    for i,dir in tqdm(enumerate(dirs),desc='process the original datasets:',total=len(dirs),unit='video'):
        j=0
        for f in Path.glob(Path(dir),'**/*.mp4'):
            j=j+1
            newfilename = str(f.parent)+'/{0:06}.mp4'.format(j)
            Path.rename(Path(str(f)),Path(newfilename))

def preProcess(inputdir,outputdir):
    processer= PreProcessor()
    processer.videosPreProcessByASR(input_dir=inputdir,
                                    output_dir=outputdir,
                                    ext='mp4')


def process_data(inputdir,outputdir,device):
    dataProcessor = DataProcessor()
    files=[]
    for f in Path.glob(Path(inputdir),'**/*.mp4'):
        if f.is_file():
            files.append(f)

    for i,fp in tqdm(enumerate(files)):
        print('process the video file: {}'.format(fp))
        dataProcessor.processVideoFile(str(fp),device=device,processed_data_root=outputdir)


def train_file_write(inputdir):
    train_txt = inputdir+'/train.txt'
    eval_txt = inputdir+'/eval.txt'
    for line in Path.glob(Path(inputdir), '**/*.mp4'):
        with open(train_txt,'a') as f:
            f.write(str(line))
        with open(eval_txt,'a') as f:
            f.write(str(line))



def main():
    args = parse_args()
    data_root = args.data_root
    p_step = args.process_step
    original_dir = data_root+'/original_data'
    preProcess_dir = data_root+'/preProcessed_data'
    process_dir = data_root+'/processed_data'

    if args.gpu_num == 0:
        device='cpu'
    else:
        device='gpu'

    if p_step==0:
        orignal_process(original_dir)
    elif p_step == 1:
        preProcess(original_dir,preProcess_dir)
    elif p_step == 2:
        process_data(preProcess_dir,process_dir,device)
    elif p_step == 3:
        orignal_process(original_dir)
        preProcess(original_dir, preProcess_dir)
        process_data(preProcess_dir, process_dir,device)
    elif p_step == 4:
        train_file_write(process_dir)
    else:
        print('wrong step number, finished!')


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="process the datasets for wav2lip")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True,type=str)
    parser.add_argument("--process_step", help='process data\'s step 0 orig,1.pre 2.pro 3.all', default=0,type=int)
    parser.add_argument("--gpu_num", help='gpu number', default=0, type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()