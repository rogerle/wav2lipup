import random

import cv2
import numpy as np
import torch
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

from process_util.ParamsUtil import ParamsUtil


class FaceDataset(Dataset):

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
        self.hp = ParamsUtil()

    def __getitem__(self, index):
        """
        循环去一段视频和一段错误视频进行网络对抗，这里就是取两个不同视频的方法
        :param index: the index of item
        :return: image
        """
        while 1:
            # 随机抽取一个视频的文件进行处理
            idx = random.randint(0, len(self.dirlist) - 1)
            audio_index = idx
            image_names = self.__get_imgs(idx)
            hp = self.hp
            if len(image_names) <=3 * hp.syncnet_T:
                continue

            window, g_window, image_name, g_image_name = self.__get_gan_window(image_names)
            if window is None or g_window is None:
                continue


            # 对音频进行mel图谱化，并进行对应。
            vid = self.dirlist[audio_index]

            wavfile = self.data_dir + '/' + vid + '/audio.wav'
            try:
                wavform, sf = torchaudio.load(wavfile,channels_first=True)
                specgram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate,
                                                                n_fft=hp.n_fft,
                                                                hop_length=hp.hop_size,
                                                                win_length=hp.win_size,
                                                                power=1,
                                                                f_min=hp.fmin,
                                                                f_max=hp.fmax,
                                                                n_mels=hp.num_mels,
                                                                normalized=hp.signal_normalization)
                orig_mel = specgram(wavform)[0]
                orig_mel = orig_mel.t().numpy()
            except Exception as e:
                continue

            mel = self.__crop_audio_window(orig_mel.copy(),image_name)

            if mel.shape[0] != hp.syncnet_mel_step_size:
                continue

            indiv_mels = self.__getsegmented_mels(orig_mel.copy(), image_name)

            if indiv_mels is None:
                continue
            # 对window进行范围缩小到0-1之间的array的处理
            window = self.__narray_window(window)
            y = window.copy()
            # 把图片的下半部分抹去
            window[:, :, window.shape[2] // 2:] = 0.

            g_window = self.__narray_window(g_window)
            x = np.concatenate([window, g_window], axis=0)


            x = torch.tensor(x,dtype=torch.float)
            y = torch.tensor(y,dtype=torch.float)
            mel = torch.tensor(np.transpose(mel,(1,0)),dtype=torch.float).unsqueeze(0)
            indiv_mels = torch.tensor(indiv_mels,dtype=torch.float).unsqueeze(1)

            return x, indiv_mels, mel, y

    def __len__(self):
        return len(self.dirlist)

    """
        下面的方法都是内部的方法，用于数据装载时，对数据的处理
    """

    def __get_split_video_list(self):
        load_file = self.data_dir + '/{}.txt'.format(self.type)
        dirlist = []
        with open(load_file, 'r') as f:
            for line in f:
                line = line.strip()
                dirlist.append(line)

        return dirlist

    def __get_imgs(self, index):
        vid = self.dirlist[index]
        img_names = []
        for img in Path().joinpath(self.data_dir, vid).glob('**/*.jpg'):
            img_names.append(img)
        return img_names

    # 随机找出对抗的两张脸的图，一个是正确的，一个是错误的
    def __get_gan_window(self, image_names):
        img_name = random.choice(image_names)
        g_img_name = random.choice(image_names)
        while img_name == g_img_name:
            g_img_name = random.choice(image_names)

        window_fnames = self.__get_window(img_name)
        g_window_fnames = self.__get_window(g_img_name)
        if window_fnames is None or g_window_fnames is None:
            return None, None,img_name,g_img_name
        window = self.__read_window(window_fnames)
        g_window = self.__read_window(g_window_fnames)

        return window, g_window,img_name,g_img_name

    def __get_window(self, img_name):
        start_id = int(Path(img_name).stem)
        seek_id = start_id + self.hp.syncnet_T
        vidPath = Path(img_name).parent
        window_frames = []
        for frame_id in range(start_id, seek_id):
            frame = str(vidPath) + '/{}.jpg'.format(frame_id)
            if not Path(frame).is_file():
                return None
            window_frames.append(frame)
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

    def __narray_window (self, window):
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

        spec = spec[start_idx:end_idx,:]

        return spec

    def __getsegmented_mels(self, spec, image_name):
        mels = []
        start_frame_num = int(Path(image_name).stem) + 1
        if start_frame_num -2 < 0:
            return None
        for i in range(start_frame_num,start_frame_num + self.hp.syncnet_T):
            m = self.__crop_audio_window(spec,i - 2)
            if m.shape[0] != self.hp.syncnet_mel_step_size:
                return None
            mels.append(np.transpose(m,(1,0)))
        mels = np.asarray(mels)

        return mels
