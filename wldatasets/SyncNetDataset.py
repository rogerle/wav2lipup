import random

import cv2
import numpy as np
import torch
from pathlib import Path

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset

from process_util.ParamsUtil import ParamsUtil


class SyncNetDataset(Dataset):
    hp = ParamsUtil()
    def __init__(self, data_dir,
                 run_type: str = 'train',
                 **kwargs):
        self.data_dir = data_dir
        self.run_type = run_type
        self.dirlist = self.__get_split_video_list()
        self.img_size = kwargs['img_size']

    def __getitem__(self, idx):
        img_dir = self.dirlist[idx]
        while 1:
            image_names = self.__get_imgs(img_dir)
            if image_names is None or len(image_names)==0:
                print('dir is {} {} is empty'.format(idx,img_dir))
                continue
        #取图片进行训练

            img_name,choosen,y = self.__get_choosen(image_names)
            window = self.__get_window(choosen,img_dir)
            if window is None or len(window) < 5:
                continue

            if window is None:
                continue

            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]

            mel = self.__get_segment_mel(img_dir,img_name)
            if mel.shape[0] != int(self.hp.syncnet_mel_step_size):
                #print("mel's shape is {} ,dir is {} {}，rechoose！！！".format(mel.shape[0],img_dir,img_name))
                continue

            x = torch.tensor(x, dtype=torch.float)
            mel = torch.tensor(np.transpose(mel, (1, 0)), dtype=torch.float).unsqueeze(0)
            return x, mel, y


    def __len__(self):
        return len(self.dirlist)

    def __get_split_video_list(self):
        load_file = self.data_dir + '/{}.txt'.format(self.run_type)
        dirlist = []
        with open(load_file, 'r') as f:
            for line in f:
                line = line.strip()
                dirlist.append(line)
        return dirlist

    def __get_imgs(self, img_dir):
        img_names = []
        for img in Path(self.data_dir+'/'+img_dir).glob('**/*.jpg'):
            img = img.stem
            img_names.append(img)
        img_names.sort(key=int)
        return img_names

    def __get_window(self, img_name,img_dir):
        start_id = int(img_name)
        seek_id = start_id + int(self.hp.syncnet_T)
        vidPath = self.data_dir+'/'+img_dir
        window_frames = []
        for frame_id in range(start_id, seek_id):
            frame = vidPath + '/{}.jpg'.format(frame_id)
            if Path(frame).exists() is False:
                return None
            try:
                img = cv2.imread(frame)
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception as e:
                return None
            window_frames.append(img)
        return window_frames

    def __crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = start_frame
        start_idx = int(80. * (start_frame_num / float(self.hp.fps))) #80.乘出来刚好是frame的长度

        end_idx = start_idx + int(self.hp.syncnet_mel_step_size)

        spec = spec[start_idx:end_idx, :]
        return spec

    def __get_choosen(self, image_names):
        img_name = random.choice(image_names)
        wrong_img_name = random.choice(image_names)
        while wrong_img_name == img_name:
            wrong_img_name = random.choice(image_names)

        if random.choice([True, False]):
            y = torch.ones(1).float()
            choosen = img_name
        else:
            y = torch.zeros(1).float()
            choosen = wrong_img_name
        return img_name,choosen,y

    def __get_segment_mel(self, img_dir, choosen):
        wavfile = self.data_dir + '/' + img_dir + '/audio.wav'
        try:
            wavform, sf = torchaudio.load(wavfile)

            wavform = F.preemphasis(wavform, self.hp.preemphasis)
            specgram = torchaudio.transforms.MelSpectrogram(sample_rate=self.hp.sample_rate,
                                                            n_fft=self.hp.n_fft,
                                                            power=1.,
                                                            hop_length=self.hp.hop_size,
                                                            win_length=self.hp.win_size,
                                                            f_min=self.hp.fmin,
                                                            f_max=self.hp.fmax,
                                                            n_mels=self.hp.num_mels,
                                                            normalized=self.hp.signal_normalization)
            orig_mel = specgram(wavform)
            orig_mel = F.amplitude_to_DB(orig_mel, multiplier=10., amin=self.hp.min_level_db,
                                         db_multiplier=self.hp.ref_level_db, top_db=80)
            orig_mel = torch.mean(orig_mel, dim=0)
            orig_mel = orig_mel.t().numpy()
            spec = self.__crop_audio_window(orig_mel.copy(), int(choosen))
            #spec = self.__normalization(spec)
        except Exception as e:
            print("Mel trasfer execption:{}".format(e))
            spec = None

        return spec
    def __normalization(self,S):
        if self.hp.allow_clipping_in_normalization:
            if self.hp.symmetric_mels:
                return np.clip((2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value,
                               -self.hp.max_abs_value, self.hp.max_abs_value)
            else:
                return np.clip(self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)), 0, self.hp.max_abs_value)
