import random
from pathlib import Path

import cv2
import numpy as np
import torchaudio
import torch
from torch import nn
from torch.nn import functional as F2
import torchaudio.functional as F
from tqdm import tqdm

from models.SyncNetModel import SyncNetModel

"""
    利用已经训练过的syncnet的来判断数据的阈值，高于阈值的数据丢弃
"""


class SyncnetScore():
    def __init__(self, data_root, default_threshold, checkpoint_pth):
        self.data_root = data_root
        self.dt = default_threshold
        self.checkpoint_pth = checkpoint_pth

    def __load_checkpoint(self, model):
        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_pth)
        else:
            checkpoint = torch.load(self.checkpoint_pth, map_location=lambda storage, loc: storage)

        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        return model

    def score_video(self):
        root = self.data_root

        dir_list = []
        for dir in Path.rglob(Path(root), '*/*'):
            if dir.is_dir():
                dir_list.append(str(dir))

        # 开始对每个目录评分
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        syncnet = SyncNetModel().to(device)
        for p in syncnet.parameters():
            p.requires_grad = False
        syncnet = self.__load_checkpoint(syncnet)
        Path(root+'/score.txt').write_text('')
        prog_bar = tqdm(enumerate(dir_list), total=len(dir_list), leave=False)
        for i,dir in prog_bar:
            score = self.__score(dir, syncnet)
            prog_bar.set_description('score the sync video:{}/{}'.format(dir,score))
            if score > 0.693:
                with open(root + '/score.txt', 'a') as f:
                    f.write("{}:{}\n".format(dir, score))

    def __score(self, dir, syncnet):
        files = []
        wavfile = dir + '/audio.wav'
        for file in Path.glob(Path(dir), '**/*.jpg'):
            if file.is_file():
                files.append(file)
        files.sort()
        syncnet.eval()
        logloss = nn.BCELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        losses = []
        for i in range(1, len(files) - 6):
            window = []
            for idx in range(i, i + 5):
                img_name = dir + '/' + '{}.jpg'.format(idx)
                img = cv2.imread(img_name)
                try:
                    img = cv2.resize(img, (288, 288))
                except Exception as e:
                    print('image resize error:{}'.format(e))
                window.append(img)

            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]
            x = torch.tensor(x, dtype=torch.float)

            try:
                wavform, sf = torchaudio.load(wavfile)

                wavform = F.preemphasis(wavform, 0.97)
                specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                                n_fft=800,
                                                                power=1.,
                                                                hop_length=200,
                                                                win_length=800,
                                                                f_min=55,
                                                                f_max=7600,
                                                                n_mels=80,
                                                                normalized=True)
                orig_mel = specgram(wavform)
                orig_mel = F.amplitude_to_DB(orig_mel, multiplier=10., amin=-100,
                                             db_multiplier=20, top_db=100)
                orig_mel = torch.mean(orig_mel, dim=0)
                orig_mel = orig_mel.t().numpy()
            except Exception as e:
                continue
            mel = self.__crop_audio_window(orig_mel.copy(), i)

            mel = torch.tensor(np.transpose(mel, (1, 0)), dtype=torch.float).unsqueeze(0)
            x = x.unsqueeze(0)
            mel = mel.unsqueeze(0)
            # 计算分数
            x = x.to(device)
            mel = mel.to(device)

            a, v = syncnet(mel, x)

            d = F2.cosine_similarity(a, v)
            y = torch.ones(1).float()
            y=y.to(device)
            loss = logloss(d, y)
            losses.append(loss)
        return sum(losses) / len(losses)

    def __crop_audio_window(self, spec, start_frame):
        start_frame_num = start_frame

        start_idx = int(80. * (start_frame_num / 25.))

        end_idx = start_idx + 16

        spec = spec[start_idx:end_idx, :]

        return spec
