import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from process_util.FaceDetector import FaceDetector


class VideoExtractor():
    face_detector = FaceDetector()

    def __init__(self, tmp_dir, fps):
        self.tmp_dir = tmp_dir
        self.fps = fps

    def __extract_video(self, video_f):
        v_f = Path(video_f)
        v_path = v_f.parts[-2]
        v_name = v_f.stem
        tmp_video_name = v_path + '_' + v_name + '.avi'
        output_dir = self.tmp_dir + '/' + v_path + '_' + v_name
        extract_dir = self.tmp_dir + '/' + v_path + '_' + v_name + '/frames'
        Path(extract_dir).mkdir(exist_ok=True, parents=True)
        print('start extract video file {}'.format(v_f.as_posix()))
        s = time.time()
        # convert video to 25fps
        ffmpeg_cmd = "ffmpeg -loglevel error -y -i {0} -qscale:v 2 -async 1 -r {1} {2}/{3}".format(v_f, self.fps,
                                                                                                   output_dir,
                                                                                                   tmp_video_name)
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        # extract video to jpg image
        ffmpeg_cmd = 'ffmpeg -loglevel error -y -i {0} -qscale:v 2 -threads 6 -f image2 {1}/{2}.jpg'.format(
            output_dir + '/' + tmp_video_name, extract_dir, '%d')
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        # extract audio file
        ffmpeg_cmd = 'ffmpeg -loglevel error -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}/{}'.format(
            output_dir + '/' + tmp_video_name, extract_dir, 'audio.wav')
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        os.unlink(output_dir+'/'+tmp_video_name)
        st = time.time()
        print('process the video file {} cost {:.2f}s'.format(tmp_video_name, st - s))
        return output_dir

    def pipline_video(self, video_f, **kwargs):
        data_root = kwargs['data_root']
        video_file = data_root + '/' + video_f

        video_path = self.__extract_video(video_file)
        faces = self.__face_crop(video_path)
        shutil.rmtree(video_path + '/frames')

    def __face_crop(self, video_path):
        frames = []

        for frame in Path(video_path).glob('**/*.jpg'):
            if frame.is_file():
                frames.append(int(frame.stem))
        frames.sort(key=int)
        bad_f_l = len(frames) % 25
        frames = frames[:-bad_f_l]
        sc = 1
        j = 0
        face_files = []
        start_frame = frames[0]
        end_frame = 0
        face_flag = 0
        probar = tqdm(enumerate(frames), total=len(frames), leave=False)
        for idx, frame in probar:
            j += 1
            frame = video_path + '/frames/' + '{}.jpg'.format(idx + 1)
            img = cv2.imread(frame)
            y_max = img.shape[0]
            x_max = img.shape[1]
            face_result = self.face_detector.faceDetec(img)
            scores = face_result['scores']
            boxes = face_result['boxes']
            if scores is None or len(scores) == 0:
                if start_frame < end_frame:
                    aud_file = self.__write_aud_file(sc, video_path, start_frame, end_frame)
                    face_files.append('sc_{}'.format(sc))
                    start_frame = end_frame
                if face_flag == 0:
                    sc += 1
                    face_flag = 1
                j = 0
                continue
            else:
                face_flag=0
                end_frame = idx + 1
                idx_s = scores.index(max(scores))
                box = boxes[idx_s]
                x1, y1, x2, y2 = box
                face = img[max(int(y1)-110,0):min(int(y2)+110,y_max),max(int(x1)-110,0):min(int(x2)+110,x_max)]
                face_path = video_path + '/' + 'sc_{}'.format(sc)
                Path(face_path).mkdir(exist_ok=True,parents=True)
                cv2.imwrite(face_path+'/{}.jpg'.format(j),face)
        aud_file = self.__write_aud_file(sc, video_path, start_frame, end_frame)
        face_files.append('sc_{}'.format(sc))
        return face_files

    def __write_aud_file(self, sc, vid_path, start_frame, end_frame):

        aud_start = int(start_frame) / 25
        aud_end = int(end_frame) / 25
        Path(vid_path).mkdir(exist_ok=True, parents=True)
        aud_file = vid_path + '/' + 'sc_{}/audio_sc{}'.format(sc,sc) + '.wav'
        ffmpeg_cmd = 'ffmpeg -loglevel error -y -i {} -ss {:.3f} -to {:.3f} {}'.format(vid_path + '/frames/audio.wav',
                                                                                       aud_start, aud_end, aud_file)
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        return aud_file
