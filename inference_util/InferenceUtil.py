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
        self.face_detector = FaceDetector()
        self.tmp_dir = os.environ.get('TEMP')
        self.fps = fps
        self.model_path = model_path

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

    def faces_detect(self, f_frames, batch_size):
        results = []
        print('start detecting faces...')
        s = time.time()
        for i in tqdm(range(0, len(f_frames), batch_size),leave=False):
            frames = f_frames[i:i + batch_size]
            boxes = self.face_detector.faceBatchDetection(frames)
            if boxes is None:
                return None
            boxes = np.array(boxes)
            batch_faces = [
                [frame[max(0, int(y1)):min(int(y2), frame.shape[0]), max(0, int(x1)):min(int(x2), frame.shape[1])],
                 (y1, y2, x1, x2)]
                for frame, (y1, y2, x1, x2) in zip(frames, boxes)]
            results.extend(batch_faces)
        st = time.time()
        print('Detect faces cost {:.2f}s:'.format(st - s))
        return results

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

    def gen_data(self, frame_names, mels, args):
        frames = self.get_frames(args.img_path, frame_names)
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        faces = self.faces_detect(frames, args.face_det_batch_size)
        if faces is None:
            raise ValueError('No faces in video!Face not detected! Ensure the video contains a face in all the frames.')

        for i, m in enumerate(mels):
            idx = 0 if args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = faces[idx].copy()
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
