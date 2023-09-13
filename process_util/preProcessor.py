import os

import librosa
import oss2.models
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm import tqdm

'''
    预处理各类文件和数据，用于训练，包括初始视频的处理切割，还有训练前数据集的预处理等。
    videoProcess()处理视频，audioProcess()处理音频。
    音频处理以2s的静默作为分割，把大视频切小，保证每个视频为一句完整的语句。视频大小应该不大于5s，如果大于5s应该继续处理。
'''
class PreProcessor():

    model_id = 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    def __init__(self,*args):
        self.processPath = args[0]
        self.asr_func = pipeline(task=Tasks.auto_speech_recognition,model=self.model_id)


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

    def audioProcessByASR(self,**types):
        audioType = types.get('audioType')
        audioFiles = self.__getProcessFils(audioType)

        for file in tqdm(audioFiles):
            rec_result = self.asr_func(audioin=file)
            print(recresult)

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