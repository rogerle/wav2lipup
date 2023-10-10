import random

import cv2
import numpy
import numpy as np
import torch
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

from process_util import audio
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
        while 1:
            idx = random.randint(0, len(self.dirlist) - 1)
            audio_index = idx
            image_names = self.__get_imgs(idx)
            hp = self.hp
            if len(image_names) <= 3 * self.hp.syncnet_T:
                continue

            img_name = random.choice(image_names)
            wrong_img_name = random.choice(image_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(image_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.__get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True

            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read:
                continue

            vid = self.dirlist[audio_index]

            wavfile = self.data_dir + '/' + vid + '/audio.wav'
            """try:
                wav = audio.load_wav(wavfile, hp.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue"""

            try:
                wavform, sf = torchaudio.load(wavfile)
                specgram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate,
                                                                n_fft=hp.n_fft,
                                                                hop_length=hp.hop_size,
                                                                win_length=hp.win_size,
                                                                f_min=hp.fmin,
                                                                f_max=hp.fmax,
                                                                n_mels=hp.num_mels)
                orig_mel = specgram(wavform)[0]
                orig_mel = orig_mel.t().numpy()
            except Exception as e:
                continue

            mel = self.__crop_audio_window(orig_mel.copy(), img_name)

            if mel.shape[0] != self.hp.syncnet_mel_step_size:
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]
            if torch.cuda.is_available() is True:
                x = torch.cuda.FloatTensor(x)
                mel = torch.cuda.FloatTensor(np.transpose(mel,(1,0))).unsqueeze(0)
            else:
                x = torch.FloatTensor(x)
                mel = torch.FloatTensor(np.transpose(mel,(1,0))).unsqueeze(0)

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

    def __get_imgs(self, index):
        vid = self.dirlist[index]
        img_names = []
        for img in Path().joinpath(self.data_dir, vid).glob('**/*.jpg'):
            img_names.append(img)
        return img_names

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

    def __crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = int(Path(start_frame).stem)

        start_idx = int(80. * (start_frame_num / float(self.hp.fps)))

        end_idx = start_idx + self.hp.syncnet_mel_step_size

        spec = spec[start_idx:end_idx, :]

        return spec
