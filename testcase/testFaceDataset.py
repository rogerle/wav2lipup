import unittest
from collections import Counter
from pathlib import Path

import librosa.display
import torch
import torchaudio
from torch.utils.data import DataLoader
from wldatasets.FaceDataset import FaceDataset
import matplotlib.pyplot as plt
from process_util.ParamsUtil import ParamsUtil


class TestFaceDataset(unittest.TestCase):

    def test_getItem(self):
        hp = ParamsUtil()
        sData = FaceDataset('../data/test_data/pr_data', img_size=288)
        test_loader = DataLoader(sData)
        for x,y,mel1,invid_mels in test_loader:
            print("matrix x's size:{}".format(x.size()))
            print("matrix y size:{}".format(y.size()))
            print(mel1)
            print(invid_mels)


        """wavform, sf = torchaudio.load('../data/test_data/pr_data/000001/000001_00000_00006/audio.wav')
        print('wav shape is {}'.format(wavform.size()))
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sample_rate,
                                                        n_fft=hp.n_fft,
                                                        hop_length=hp.hop_size,
                                                        win_length=hp.win_size,
                                                        power=2,
                                                        f_min=hp.fmin,
                                                        f_max=hp.fmax,
                                                        n_mels=hp.num_mels,
                                                        normalized=hp.signal_normalization)(wavform)
        specgram = specgram.mT
        print('specgram shape is {}'.format(specgram.size()))
        plt.figure()
        p = plt.imsave('test.png',specgram.log2()[0,:,:].detach().numpy(),cmap="gray",bbox_inches=None,pad_inches=0)"""



