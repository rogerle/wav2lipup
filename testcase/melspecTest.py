import random
import unittest

import librosa
import torchaudio.functional as F
import torchaudio
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class MelSpecTest(unittest.TestCase):
    def testMelShow(self):
        wavfile = "../data/test_data/pr_data/000001/000001_00058_00063/audio.wav"
        wavform, sf = torchaudio.load(wavfile)
        spec = T.Spectrogram(n_fft=800,hop_length=200,win_length=800,power=1.0)(wavform)
        spec = T.AmplitudeToDB(stype='magnitude',
                               top_db=80.)(spec)
        true=0
        false1=0
        for i in range(1,1000):
            if random.choice([True, False]):
                true +=1
            else:
                false1+=1

        print('chosie true:{} flas:{}'.format(true,false1))


        asis = 0.97  # filter coefficient.
        wavform = F.preemphasis(wavform,float(asis))
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                        n_fft=800,
                                                        power=1.,
                                                        hop_length=200,
                                                        win_length=800,
                                                        f_max=7600,
                                                        f_min=55,
                                                        norm='slaney',
                                                        normalized=True,
                                                        n_mels=80,
                                                        pad_mode='reflect',
                                                        mel_scale='htk'
                                                        )
        orig_mel = specgram(wavform)
        orig_mel = F.amplitude_to_DB(orig_mel, multiplier=10., amin=-100,
                                     db_multiplier=20, top_db=80)
        orig_mel = torch.mean(orig_mel, dim=0)
        mel = orig_mel.t().numpy().copy()

        #num_frames = (T x hop_size * fps) / sample_rate
        #np.clip((2 * 4) * (
        #            (mel - (-100)) / (-(-100))) - 4,
       #         -4, 4)
        start_frame_num = 48

        start_idx = int(80. * (start_frame_num / float(25)))  # 80.乘出来刚好是frame的长度

        end_idx = start_idx + 16
        mel = mel[start_idx:end_idx,:]


        fig, axs = plt.subplots(3, 1)
        self.plot_wavform(wavform,sf,title='Original wavform',ax=axs[0])
        self.plot_spectrogram(spec[0],title="spectrogram",ax=axs[1])
        self.plot_spectrogram(np.transpose(mel, (1, 0)), title="Mel-spectrogram",ax=axs[2])
        fig.tight_layout()


        plt.show()


    def plot_wavform(self,wavform,sr,title="wavform",ax=None):
        waveform=wavform.numpy()

        num_channels,num_frames = waveform.shape
        time_axis = torch.arange(0,num_frames)/sr

        if ax is None:
            _,ax = plt.subplots(num_channels,1)
        ax.plot(time_axis,waveform[0],linewidth=1)
        ax.grid(True)
        ax.set_xlim([0,time_axis[-1]])
        ax.set_title(title)

    def plot_spectrogram(self,specgram,title=None,ylabel='freq_bin',ax=None):
        if ax is None:
            _,ax =plt.subplots(1,1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(specgram,origin='lower',aspect='auto',interpolation='nearest')

