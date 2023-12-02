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
    def __load_checkpoint(self,checkpoint_pth, model):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_pth)
        else:
            checkpoint = torch.load(checkpoint_pth, map_location=lambda storage, loc: storage)

        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        return model
    def score_video(self, v_file, **kwargs):
        v_dir = kwargs['data_root'] + '/' + v_file
        checkpoint = kwargs['checkpoint']
        batch_size = kwargs['batch_size']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SyncNetModel().to(device)
        for p in model.parameters():
            p.requires_grad = False
        model = self.__load_checkpoint(checkpoint, model)
        score, conf = self.__score(v_dir, model, batch_size)
        return v_file, score, conf

    def __score(self, v_dir, model, batch_size):
        files = []
        wavfile = v_dir + '/audio.wav'
        for file in Path.glob(Path(v_dir), '**/*.jpg'):
            if file.is_file():
                img = file.stem
                files.append(int(img))
        files.sort(key=int)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #把图形文件名按batchsize做成batch
        last_fname = len(files) - 5
        original_mel = self.__get_mel(wavfile)
        lip_feats = []
        aud_feats = []

        for i in range(0, last_fname, batch_size):
            lip_batch = []
            aud_batch = []
            for fname in range(i + 1, min(last_fname, i + batch_size)):
                lip_win = self.__get_lipwin(v_dir, fname)
                aud_win = self.__get_aud_windows(original_mel, fname)
                lip_batch.append(lip_win)
                aud_batch.append(aud_win)
            lip_wins=torch.cat(lip_batch,0)
            aud_wins=torch.cat(aud_batch,0)
            x = lip_wins.to(device)
            mel = aud_wins.to(device)
            a, v = model(mel, x)
            lip_feats.append(v.cpu())
            aud_feats.append(a.cpu())

        if len(lip_feats) != len(aud_feats):
            return 15,15.
        lip_feat = torch.cat(lip_feats,0)
        aud_feat = torch.cat(aud_feats,0)
        a_pad = F2.pad(aud_feat, (0, 0, 15, 15))
        dists = []
        for i in range(0, len(lip_feat)):
            s_l = lip_feat[[i], :].repeat(31, 1)
            s_a = a_pad[i:i + 31, :]
            d = F2.cosine_similarity(s_a, s_l)
            dists.append(d)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        maxval, maxidx = torch.max(mdist, 0)

        offset = 15 - maxidx.item()
        conf = maxval - torch.median(mdist).item()

        return offset, conf

    def __crop_audio_window(self, spec, start_frame):
        start_frame_num = start_frame

        start_idx = int(80. * (start_frame_num / 25.))

        end_idx = start_idx + 16

        spec = spec[start_idx:end_idx, :]

        return spec

    def __get_mel(self, wavfile):
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
            print("mel error:".format(e))
            return None
        return orig_mel

    def __get_lipwin(self, path, fname):
        start_id = fname
        seek_id = fname+5
        window =[]
        for fidx in range(start_id, seek_id):
            img_name = path + '/' + '{}.jpg'.format(fidx)

            try:
                img_f = cv2.imread(img_name)
                img_f = cv2.resize(img_f, (288, 288))
            except Exception as e:
                print('image resize error:{}'.format(e))
                img_f = np.random.randn(288, 288, 3)
            window.append(img_f)

        x = np.concatenate(window, axis=2) / 255.
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1] // 2:]
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)


        return x

    def __get_aud_windows(self, original_mel, fname):
        mel = self.__crop_audio_window(original_mel.copy(), fname)
        if mel.shape[0] != 16:
            return torch.randn(1,1,80,16)
        mel = torch.tensor(np.transpose(mel, (1, 0)), dtype=torch.float).unsqueeze(0).unsqueeze(0)

        return mel
