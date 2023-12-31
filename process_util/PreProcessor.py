from pathlib import Path
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy.editor import *

from tqdm import tqdm
import logging


'''
    预处理各类文件和数据，用于训练，包括初始视频的处理切割，还有训练前数据集的预处理等。
    videoProcess()处理视频，audioProcess()处理音频。
    音频处理以2s的静默作为分割，把大视频切小，保证每个视频为一句完整的语句。视频大小应该不大于5s，如果大于5s应该继续处理。
'''


class PreProcessor():
    logging.basicConfig(level=logging.ERROR)
    model_id = 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'

    '''
        利用ASR对音频文件进行时间戳分离，并写入相同文件名的json文件,然后根据时间戳对视频进行分割处理
    '''

    def videosPreProcessByASR(self,video,**kwargs):
        input_dir = kwargs.get('input_dir')
        output_dir = kwargs.get('output_dir')
        asr_func = pipeline(task=Tasks.auto_speech_recognition,
                            model=self.model_id)
        # 看是否已经有时间戳，没有的话就做时间戳文件
        parent_path = Path(video).parent
        file_name = Path(video).stem
        jsonfile = Path.joinpath(parent_path, file_name).with_suffix('.json')
        print('generate the viedo asr file:{}'.format(jsonfile))
        self.__genTimeStampByASR(video=video,
                                 asr=asr_func)
        with open(jsonfile, 'r') as f:
            dicts = json.load(f)
        timestamps = dicts['timestamps']
        start = 0
        videoC = VideoFileClip(str(video))
        movieEnd = int(videoC.duration)
        outputD = self.__genOutputDir(input_dir,
                                      output_dir,
                                      video)
        for time in timestamps:
            tmpEnd = time['end']
            if time['start'] != start:
                start = time['start']
            if (tmpEnd - start) < 1000:
                continue
            else:
                startTime = round(start / 1000)
                endTime = round(tmpEnd / 1000)
                if endTime > movieEnd:
                    endTime = movieEnd
                if endTime > startTime:
                    self.__genClipVideo(videoC, startTime, endTime, outputD)
                else:
                    continue
                start = tmpEnd
        return outputD

    '''
        把视频按时间戳文件进行切割
    '''

    def videosPreProcessByTime(self, video,**kwargs):
        S_TIME = kwargs.get('s_time')
        input_dir = kwargs.get('input_dir')
        output_dir = kwargs.get('output_dir')
        outputD = self.__genOutputDir(input_dir, output_dir, video)
        i = 0
        videoC = VideoFileClip(str(video))
        movieEnd = int(videoC.duration)

        # 按秒数来分割视频，最后一段到结束
        while i < movieEnd:
            startTime = i
            endTime = i + S_TIME
            if endTime > movieEnd:
                endTime = movieEnd
            self.__genClipVideo(videoC, startTime, endTime, outputD)
            i = i + S_TIME

        return outputD
    '''
        切割视频文件写入到指定目录
    '''

    def __genClipVideo(self, videoClip, startTime, endTime, outputD):
        outputName = '{0}/{1:05}_{2:05}.mp4'.format(outputD,
                                                    startTime,
                                                    endTime)
        clipVideo = videoClip.subclip(startTime, endTime)
        clipVideo.write_videofile(outputName,fps=25,logger=None)

    '''
        处理文件后的输出目录生成并返回目录名称
    '''

    def __genOutputDir(self, input_dir, output_dir, file):
        iparts = Path(input_dir).parts
        fparts = Path(file).parent.parts
        outparts = []
        # 分割文件路径，获取上层需要创建的路径名成
        for fp in fparts:
            if fp not in iparts:
                outparts.append(fp)
        # 构建输出路径，并创建不存在的输出目录
        op = output_dir
        if len(outparts) > 0:
            for o in outparts:
                op = op + '/{}'.format(o)
                Path(op).mkdir(exist_ok=True)
        lastPath = Path(file).name.split('.')[0]
        op = op + '/{}'.format(lastPath)
        Path(op).mkdir(exist_ok=True)
        dir = str(op)
        return dir

    '''
        获取所有处理文件，并返回文件列表，type是文件的扩展名，也是文件的类型，内部私有方法
    '''



    '''
        把视频文件的音频剥离出来
    '''

    def __genTimeStampByASR(self, **kwargs):
        video = kwargs.get('video')
        asr_func = kwargs.get('asr')

        wname = Path(video).name.replace('.mp4', '.wav')
        temp_dir = os.environ.get('TEMP')
        print('wavefile put in temp:{}'.format(temp_dir))
        wavfile = temp_dir + '/' + wname
        audio_clip = AudioFileClip(str(video))
        audio_clip.write_audiofile(wavfile, logger=None)

        rec_result = asr_func(audio_in=wavfile)

        # 获取语句时间戳
        sentences = rec_result.get('sentences')
        timest = []
        for items in sentences:
            if items['text'] is not None and items['text'].strip() != '':
                timest.append({'start': items['start'], 'text': items['text'], 'end': items['end']})
        fname = Path(video).name.replace('.mp4', '.json')
        path = Path(video).parent
        tf = str(path) + '/' + fname
        video_times = {'timestamps': timest}
        with open(tf, 'w') as f:
            f.write(json.dumps(video_times))
        return video_times
