import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm

from process_util.FaceDetector import FaceDetector
from process_util.ParamsUtil import ParamsUtil

hp = ParamsUtil()


class InferenceUtil():
    def __init__(self, fps, model_path):
        self.tmp_dir = os.environ.get('TEMP')
        self.fps = fps
        self.model_path = model_path

    def get_smoothened_boxes(self,boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    def src_video_process(self, video_f):
        v_f = Path(video_f)
        v_name = v_f.stem
        tmp_video_name = v_name + '.avi'
        output_dir = self.tmp_dir + '/' + v_name
        extract_dir = self.tmp_dir + '/' + v_name + '/frames'
        Path(extract_dir).mkdir(exist_ok=True, parents=True)
        print('start extract source video file {}'.format(v_f.as_posix()))
        s = time.time()
        # change source video to 25fps
        ffmpeg_cmd = "ffmpeg -loglevel error -y -i {0} -qscale:v 2 -async 1 -r {1} {2}/{3}".format(v_f, self.fps,
                                                                                                   output_dir,
                                                                                                   tmp_video_name)
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        # extract video to jpg
        ffmpeg_cmd = 'ffmpeg -loglevel error -y -i {0} -qscale:v 2 -threads 6 -f image2 {1}/{2}.jpg'.format(
            output_dir + '/' + tmp_video_name, extract_dir, '%d')
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)
        st = time.time()
        print('process the video file {} cost {:.2f}s'.format(video_f, st - s))
        return extract_dir

    def faces_detect(self, f_frames, args):
        results = []
        head_exist = []

        face_detector = FaceDetector()
        batch_size = args.face_det_batch_size
        print('start detecting faces...')
        s = time.time()
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(f_frames), batch_size),leave=False):
                    predictions.extend(face_detector.faceBatchDetection(f_frames[i:i+batch_size]))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('Image too large to run face detection')
                batch_size //=2
                print('Recovering from OOM; New batchsize is {}'.format(batch_size))
                continue
            break
        pady1,pady2,padx1,padx2 =args.pads

        #获取第一帧头像的大小
        f_head_rec = None
        f_head_img = None
        for rect,img in zip(predictions,f_frames):
            if rect is not None:
                f_head_rec = rect
                f_head_img = img
                break

        for rect,img in zip(predictions,f_frames):
            if rect is None:
                head_exist.append(False)
                if len(results) == 0:
                    y1 = max(0,f_head_rec[1]-pady1)
                    y2 = min(f_head_img.shape[0],f_head_rec[3]+pady2)
                    x1 = max(0,f_head_rec[0]-padx1)
                    x2 = min(f_head_img.shape[1],f_head_rec[2]+padx2)
                    results.append([x1,y1,x2,y2])
                else:
                    results.append(results[-1])
            else:
                head_exist.append(True)
                y1 =max(0,rect[1] - pady1)
                y2 = min(f_head_img.shape[0],rect[3]+pady2)
                x1 = max(0,rect[0] - padx1)
                x2 = min(f_head_img.shape[1],rect[2]+padx2)
                results.append([x1,y1,x2,y2])
        boxes = np.array(results)
        if not args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[f_frames[y1:y2,x1:x2],(y1,y2,x1,x2)] for f_frames,(x1,y1,x2,y2) in zip(f_frames,boxes)]
        del face_detector
        return results,head_exist

    def get_frames(self, img_path, names):
        imgs = []
        for img in names:
            img_dir = img_path + '/' + str(img) + '.jpg'
            img = cv2.imread(img_dir)
            imgs.append(img)
        return imgs

    def load_img_names(self, frame_path):
        img_names = []
        for img in Path(frame_path).glob('**/*.jpg'):
            img = img.stem
            img_names.append(img)
        img_names.sort(key=int)
        return img_names

    def extract_audio(self, wavfile):
        print('Extracting raw audio...')
        tmp_audio = self.tmp_dir + '/temp.wav'
        ffmpeg_cmd = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(wavfile, tmp_audio)
        output = subprocess.call(ffmpeg_cmd, shell=True, stdout=None)

        return tmp_audio

    def get_mel(self, wavefile):
        try:
            waveform, sf = torchaudio.load(wavefile)

            resample = torchaudio.transforms.Resample(sf, 16000)
            waveform = resample(waveform)
            waveform = F.preemphasis(waveform, hp.preemphasis)

            specgram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate,
                                                            n_fft=hp.n_fft,
                                                            power=1.,
                                                            hop_length=hp.hop_size,
                                                            win_length=hp.win_size,
                                                            f_min=hp.fmin,
                                                            f_max=hp.fmax,
                                                            n_mels=hp.num_mels,
                                                            normalized=hp.signal_normalization)
            orig_mel = specgram(waveform)
            orig_mel = F.amplitude_to_DB(orig_mel, multiplier=10., amin=hp.min_level_db,
                                         db_multiplier=hp.ref_level_db, top_db=100)
            orig_mel = torch.mean(orig_mel, dim=0)
            orig_mel = orig_mel.numpy()
        except Exception as e:
            orig_mel = None

        return orig_mel

    def gen_data(self, frames, mels, args):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        #*******识别人脸位置坐标，未识别的对应为none *******************
        if args.box[0] == -1:
            if not args.static:
                face_det_result,head_exist = self.faces_detect(frames,args)
            else:
                face_det_result,head_exist = self.faces_detect([frames[0]],args)
        else:
                print('Using the specified bounding box instead of face detaction....')
                y1,y2,x1,x2 = args.box
                face_det_result = [[(f[y1:y2, x1:x2],(y1,y2,x1,x2))] for f in frames ]
                head_exit = [True]*len(frames)
        if face_det_result is None:
            raise ValueError('No faces in video!Face not detected! Ensure the video contains a face in all the frames.')

        for i, m in tqdm(enumerate(mels),total=len(mels),leave=False):
            idx = 0 if args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_result[idx].copy()
            face = cv2.resize(face, (hp.img_size, hp.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, hp.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, hp.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
