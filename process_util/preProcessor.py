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
    def __init__(self,*args):
        self.inputPath = args[0]



    '''
        根据静音的时间长短来对音频或mp4视频文件进行切割
    '''
    def audioProcessBySilent(self,**types):
        audioType = types.get('audioType')
        audioFiles = self.__getProcessFils(audioType)
        '''
            对音频按静音分割记录时间戳
        '''
        for file in tqdm(audioFiles):
            aud = AudioSegment.from_file(file,format=audioType)
            loudness = aud.dBFS
            segments = detect_nonsilent(aud,
                                      min_silence_len=500,
                                      silence_thresh=loudness - 50,
                                      seek_step=1
                                      )
            print(segments)
    '''
        利用ASR对音频文件进行时间戳分离，并写入相同文件名的json文件
    '''
    def audioProcessByASR(self,**types):
        audioType = types.get('audioType')
        audioFiles = self.__getProcessFils(audioType)
        if audioType == 'mp4':
            for file in tqdm(audioFiles):
                self.__splitAudioFromVideo(file)
        audioFiles = self.__getProcessFils('wav')

        asr_func = pipeline(task=Tasks.auto_speech_recognition, model=self.model_id)


        for file in tqdm(audioFiles):
            rec_result = asr_func(audio_in=file)
            sentences = rec_result.get('sentences')
            timest=[]
            for items in sentences:
                timest.append({'start':items['start'],'end':items['end']})
            #对时间戳进行记录到文件，下次再处理可以不用再用语音处理
            absfile = os.path.abspath(file)
            filename = absfile.split('.')[0]
            newfile = filename+'.json'
            video_times={'timestamps':timest}
            with open(newfile,'w') as wf:
                wf.write(json.dumps(video_times))

    '''
        把视频按时间戳文件进行切割
    '''
    def videosPreProcess(self,**kwargs):
        S_TIME=5
        input_dir = kwargs.get('input_dir')
        output_dir = kwargs.get('output_dir')

        videos = self.__getProcessFils(input_dir,'mp4')
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
        suffix = Path(file).suffix
        fparts = Path(file).parent.parts
        outparts = []
        for fp in fparts:
            if fp not in iparts:
                outparts.append(fp)
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
        for file in Path.glob(Path(inputPath),'*/*.{}'.format(fileType)):
            files.append(file)
        return files

    '''
        把视频文件的音频剥离出来
    '''
    def __splitAudioFromVideo(self,videoFile):
        video = videoFile
        audio_name = os.path.basename(video).split('.')[0]
        path = self.processPath
        audio_file = os.path.join(path,audio_name+'.wav')
        audio_clip = AudioFileClip(video)
        audio_clip.write_audiofile(audio_file)

