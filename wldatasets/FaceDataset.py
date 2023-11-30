import random

import cv2
import numpy as np
import torch
from pathlib import Path

import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset

from process_util.ParamsUtil import ParamsUtil


class FaceDataset(Dataset):
    hp = ParamsUtil()

    def __init__(self, data_dir,
                 run_type: str = 'train',
                 **kwargs):
        """
            :param data_dir: 数据文件的根目录
            :param video_info: 所有视频文件的目录信息，一般放在train.txt文件中。
        """
        self.data_dir = data_dir
        self.type = run_type
        self.img_size = kwargs['img_size']
        self.dirlist = self.__get_split_video_list()

    def __getitem__(self, idx):
        """
        循环去一段视频和一段错误视频进行网络对抗，这里就是取两个不同视频的方法
        :param index: the index of item
        :return: image
        """
        img_dir = self.dirlist[idx]
        while 1:
            # 随机抽取一个帧作为起始帧进行处理
            image_names = self.__get_imgs(img_dir)
            if image_names is None or len(image_names) <= 3 * self.hp.syncnet_T:
                continue

            # 获取连续5张脸，正确和错误的
            img_name, wrong_img_name = self.__get_choosen(image_names)
            window = self.__get_window(img_name,img_dir)
            wrong_window = self.__get_window(wrong_img_name,img_dir)
            if window is None or wrong_window is None:
                continue

            # 对音频进行mel图谱化，并进行对应。
            orginal_mel = self.__get_orginal_mel(img_dir)

            mel = self.__crop_audio_window(orginal_mel.copy(),int(img_name))
            if mel is None or mel.shape[0] != self.hp.syncnet_mel_step_size:
                continue

            indiv_mels = self.__getsegmented_mels(orginal_mel.copy(), img_name)

            if indiv_mels is None:
                continue
            # 对window进行范围缩小到0-1之间的array的处理
            window = self.__prepare_window(window)
            y = window.copy()
            # 把图片的上半部分抹去
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.__prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            mel = torch.tensor(np.transpose(mel, (1, 0)), dtype=torch.float).unsqueeze(0)
            indiv_mels = torch.tensor(indiv_mels, dtype=torch.float).unsqueeze(1)

            return x, indiv_mels, mel, y

    def __len__(self):
        return len(self.dirlist)

    """
        下面的方法都是内部的方法，用于数据装载时，对数据的处理
    """

    def __get_choosen(self, image_names):
        img_name = random.choice(image_names)
        wrong_img_name = random.choice(image_names)
        while wrong_img_name == img_name:
            wrong_img_name = random.choice(image_names)

        return img_name, wrong_img_name

    def __get_split_video_list(self):
        load_file = self.data_dir + '/{}.txt'.format(self.type)
        dirlist = []
        with open(load_file, 'r') as f:
            for line in f:
                line = line.strip()
                dirlist.append(line)

        return dirlist

    def __get_imgs(self, img_dir):
        img_names = []
        for img in Path(self.data_dir + '/' + img_dir).glob('**/*.jpg'):
            img = img.stem
            img_names.append(img)
        img_names.sort(key=int)
        return img_names

    def __get_window(self, img_name, img_dir):
        start_id = int(img_name)
        seek_id = start_id + int(self.hp.syncnet_T)
        vidPath = self.data_dir + '/' + img_dir
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

    def __read_window(self, window_fnames):
        window = []
        for f_name in window_fnames:
            img = cv2.imread(f_name)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception as e:
                print('Resize the face image error: {}'.format(e))
                return None
            window.append(img)
        return window

    def __prepare_window(self, window):
        # 数组转换3xTxHxW
        wa = np.asarray(window) / 255.
        wa = np.transpose(wa, (3, 0, 1, 2))

        return wa

    def __crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = int(Path(start_frame).stem)

        start_idx = int(80. * (start_frame_num / float(self.hp.fps)))

        end_idx = start_idx + self.hp.syncnet_mel_step_size

        spec = spec[start_idx:end_idx, :]

        return spec

    def __getsegmented_mels(self, spec, image_name):
        mels = []
        start_frame_num = int(Path(image_name).stem) + 1
        if start_frame_num - 2 < 0:
            return None
        for i in range(start_frame_num, start_frame_num + self.hp.syncnet_T):
            m = self.__crop_audio_window(spec, i - 2)
            if m.shape[0] != self.hp.syncnet_mel_step_size:
                return None
            mels.append(np.transpose(m, (1, 0)))
        mels = np.asarray(mels)

        return mels

    def __get_orginal_mel(self, img_dir):
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
                                         db_multiplier=self.hp.ref_level_db)
            orig_mel = torch.mean(orig_mel, dim=0)
            orig_mel = orig_mel.t().numpy()
        except Exception as e:
            orig_mel = None

        return orig_mel
