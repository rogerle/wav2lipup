import pickle

import cv2
from pathlib import Path

import numpy as np
import torchaudio
from moviepy.editor import *
from tqdm import tqdm

from process_util.FaceDetector import FaceDetector


class DataProcessor():
    face_detector = FaceDetector()
    '''
        视频文件处理，把视频文件分解成脸部图片。并分离出音频
    '''

    def processVideoFile(self, vfile, **kwargs):
        vCap = cv2.VideoCapture()

        ok = vCap.open(vfile)

        frames = []
        while ok:
            success, frame = vCap.read()
            if not success:
                vCap.release()
                break
            frames.append(frame)
        # 创建解开的目录
        split_dir = self.__get_split_path(vfile, kwargs['processed_data_root'])
        self.__extract_face_img(frames, split_dir)
        self.__extract_audio(vfile, split_dir)

    """
        提取音频文件到数据处理文件夹
    """

    def __extract_audio(self, vfile, split_dir):
        audio_clip = AudioFileClip(str(vfile))
        audiofile = split_dir + '/audio.wav'
        audio_clip.write_audiofile(audiofile,logger=None)

        audio_meta_f = split_dir + '/audio_meta.info'

        wavform, sr = torchaudio.load(audiofile)
        resample = torchaudio.transforms.Resample(sr, 16000)
        wavform = resample(wavform)
        torchaudio.save(audiofile, wavform, sample_rate=16000)

        audio_meta = torchaudio.info(audiofile)
        with open(audio_meta_f, 'w') as f:
            f.write(str(audio_meta))

    """
        提取人脸图片放入数据处理文件
    """

    def __extract_face_img(self, frames, split_dir):
        prog_bar = tqdm(enumerate(frames), total=len(frames), leave=False)
        faces={}
        face_file = split_dir+'/faces.pkl'
        for j, frame in prog_bar:
            j = j+1
            face_result = self.face_detector.faceDetec(frame)
            scores = face_result['scores']
            boxes = face_result['boxes']
            if scores is None or len(scores) == 0:
                print('bad face video,drop it!')
                continue
            else:
                idx = scores.index(max(scores))
                box = boxes[idx]
                x1, y1, x2, y2 = box
                file_name = split_dir + '/{}.jpg'.format(j)
                face = frame[int(y1):int(y2), int(x1):int(x2)]
                #faces['{}'.format(j)] = face
                cv2.imwrite(file_name, face)
                """max_score = scores[0]
                   for i, score in enumerate(scores):
                    if score >= max_score:
                        max_score = score
                        box = boxes[i]
                        x1, y1, x2, y2 = box
                        file_name = split_dir + '/{}.jpg'.format(j)
                        face = frame[int(y1):int(y2), int(x1):int(x2)]
                        faces['{}'.format(j)]=face
                        cv2.imwrite(file_name, face)"""
            prog_bar.set_description('Extract Face Image：{}.jpg'.format(j))

        #写入脸部文件“faces.bin",注意的是这个里面保存的是dict文件
        #with open(face_file,'wb') as f:
            #pickle.dump(faces,f)

    def __get_split_path(self, vfile, processed_data_root):
        vf = Path(vfile)
        fdir = vf.parts[-3]
        fbase = vf.parts[-2] + '_' + vf.stem
        fulldir = processed_data_root + '/' + str(fdir) + '/' + fbase
        Path(fulldir).mkdir(parents=True, exist_ok=True)

        return fulldir
