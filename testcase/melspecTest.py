import unittest

import librosa
import torchaudio.functional as F
import torchaudio
import torch
import matplotlib.pyplot as plt
from scipy import signal


class MelSpecTest(unittest.TestCase):
    def testMelShow(self):
        wavfile = "../data/test_data/pr_data/000001/000001_00054_00060/audio.wav"
        wavform, sf = torchaudio.load(wavfile)
        asis = 0.97  # filter coefficient.
        wavform = F.preemphasis(wavform,float(asis))
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                        n_fft=800,
                                                        f_min=55,
                                                        f_max=7600,
                                                        power=2,
                                                        hop_length=200,
                                                        win_length=800,
                                                        center=True,
                                                        normalized=True,
                                                        n_mels=80)
        orig_mel = specgram(wavform)
        mel = F.amplitude_to_DB(orig_mel,multiplier=10.,amin=-100,db_multiplier=20,top_db=100)
        mel = torch.mean(mel, dim=0)
        print(orig_mel.size())
        t, ax = plt.subplots(1, 1)
        ax.set_title("Mel- Frequece")
        ax.set_ylabel('Frequence')
        ax.imshow(mel, origin="lower", aspect="auto")
        plt.show()


