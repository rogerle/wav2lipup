from pathlib import Path
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy.editor import *

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm import tqdm

'''
    预处理各类文件和数据，用于训练，包括初始视频的处理切割，还有训练前数据集的预处理等。
    videoProcess()处理视频，audioProcess()处理音频。
    音频处理以2s的静默作为分割，把大视频切小，保证每个视频为一句完整的语句。视频大小应该不大于5s，如果大于5s应该继续处理。
'''
class PreProcessor():

    model_id = 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch'

    '''
        根据静音的时间长短来对音频或mp4视频文件进行切割
    '''
    def videosPreProcessBySilent(self,**types):
        ext = types.get('ext')
        audioFiles = self.__getProcessFils(ext)
        '''
            对音频按静音分割记录时间戳
        '''
        for file in tqdm(audioFiles):
            aud = AudioSegment.from_file(file,format=ext)
            loudness = aud.dBFS
            segments = detect_nonsilent(aud,
                                      min_silence_len=500,
                                      silence_thresh=loudness - 50,
                                      seek_step=1
                                      )
            print(segments)
    '''
        利用ASR对音频文件进行时间戳分离，并写入相同文件名的json文件,然后根据时间戳对视频进行分割处理
    '''
    def videosPreProcessByASR(self,**kwargs):
        inputdir = kwargs.get('input_dir')
        outputdir = kwargs.get('output_dir')
        ext = kwargs.get('ext')


        videoFiles = self.__getProcessFils(input_dir=inputdir,
                                           type=ext)
        asr_func = pipeline(task=Tasks.auto_speech_recognition,
                            model=self.model_id)
        for video in tqdm(videoFiles):
            self.__genTimeStampByASR(video=video,
                                     asr=asr_func)

        #接下去要写根据时间戳来分割视频

    '''
        把视频按时间戳文件进行切割
    '''
    def videosPreProcessByTime(self, **kwargs):
        S_TIME= kwargs.get('s_time')
        input_dir = kwargs.get('input_dir')
        output_dir = kwargs.get('output_dir')
        ext = kwargs.get('ext')

        videos = self.__getProcessFils(input_dir,ext)
        for video in videos:
            outputD=self.__genOutputDir(input_dir,output_dir,video)
            i=0
            videoC = VideoFileClip(str(video))
            movieEnd = int(videoC.duration)
            while i < movieEnd:
                startTime = i
                endTime = i + S_TIME

                outputName = '{0}/{1:05}_{2:05}.mp4'.format(outputD,
                                                            startTime,
                                                            endTime)

                if endTime > movieEnd:
                    endTime = movieEnd
                clipVideo = videoC.subclip(startTime,endTime)
                clipVideo.write_videofile(outputName)
                i=i+S_TIME

    '''
        处理文件后的输出目录生成并返回目录名称
    '''
    def __genOutputDir(self,input_dir,output_dir,file):
        iparts = Path(input_dir).parts
        fparts = Path(file).parent.parts
        outparts = []
        #分割文件路径，获取上层需要创建的路径名成
        for fp in fparts:
            if fp not in iparts:
                outparts.append(fp)
        #构建输出路径，并创建不存在的输出目录
        op = output_dir
        if len(outparts) > 0:
            for o in outparts:
                op=op +'/{}'.format(o)
                Path(op).mkdir(exist_ok=True)
        lastPath = Path(file).name.split('.')[0]
        op = op +'/{}'.format(lastPath)
        Path(op).mkdir(exist_ok=True)
        dir = str(op)
        return dir


    '''
        获取所有处理文件，并返回文件列表，type是文件的扩展名，也是文件的类型，内部私有方法
    '''
    def __getProcessFils(self,input_dir,type):
        inputPath = input_dir
        fileType = type
        files = []
        for file in Path.glob(Path(inputPath),'**/*.{}'.format(fileType)):
            files.append(file)
        return files

    '''
        把视频文件的音频剥离出来
    '''
    def __genTimeStampByASR(self, **kwargs):
        video = kwargs.get('video')
        asr_func = kwargs.get('asr')

        wname = Path(video).name.replace('.mp4','.wav')
        temp_dir = os.environ.get('TEMP')
        wavfile = temp_dir+'/'+wname
        audio_clip = AudioFileClip(str(video))
        audio_clip.write_audiofile(wavfile)

        rec_result = asr_func(audio_in=wavfile)

        # 获取语句时间戳
        sentences = rec_result.get('sentences')
        timest = []
        for items in sentences:
            timest.append({'start': items['start'], 'end': items['end']})
        fname = Path(video).name.replace('.mp4', '.json')
        path = Path(video).parent
        tf = str(path) + '/' + fname
        video_times = {'timestamps': timest}
        with open(tf, 'w') as f:
            f.write(json.dumps(video_times))
        print(video_times)
