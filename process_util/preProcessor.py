import os

import json
import ffmpy

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
        self.processPath = args[0]



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
    def videosPreProcess(self):
        videos = self.__getProcessFils('mp4')
        for video in videos:
            splitName=os.path.splitext(video)[0]
            jsonFile = splitName+'.json'
            with open(jsonFile,'r') as f:
                dicts = json.load(f)
            timestamps=dicts['timestamps']
            os.makedirs(splitName,exist_ok=True)
            i=0
            videoC = VideoFileClip(video)
            movieEnd = int(videoC.duration)
            for timestamp in timestamps:
                i=i+1

                startTime = int(timestamp['start']/1000)
                endTime = int(timestamp['end'] / 1000)
                if endTime == startTime:
                    endTime= endTime+1
                outputName = '{0}/{1:06}.mp4'.format(splitName,i)

                if endTime > movieEnd:
                    endTime = movieEnd
                clipVideo = videoC.subclip(startTime,endTime)
                clipVideo.write_videofile(outputName)


    '''
        获取所有处理文件，并返回文件列表，type是文件的扩展名，也是文件的类型，内部私有方法
    '''
    def __getProcessFils(self, type):
        fileType = type
        path = self.processPath
        files = []
        for f in os.listdir(path):
            file = os.path.join(path, f)
            if os.path.isfile(file) and file.endswith(fileType):
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

    def __ms_to_hours(self,millis):
        seconds = (millis/1000) % 60
        seconds = int (seconds)
        minutes = (millis/(1000*60)) % 60
        minutes = int (minutes)
        hours = (millis/(1000*60*60)) % 24
        hours = int (hours)

        return ("%d:%d:%d" % (hours,minutes,seconds))
