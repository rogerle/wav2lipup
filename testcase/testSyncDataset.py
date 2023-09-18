import unittest
from collections import Counter
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import DataLoader


from wavdatas.SyncDataset import SyncDataset


class TestSyncDataset(unittest.TestCase):

    def test_getItem(self):
        sData = SyncDataset('../data/test_data/pr_data',img_size=288)
        test_loader = DataLoader(sData)
        x,y = next(iter(test_loader))
        print("martix is {0} \n {1}".format(x,y))

        print(torch.__version__)
        print(torchaudio.__version__)
        wavform,sf = torchaudio.load('../data/test_data/pr_data/000001/000001/00000_00006/audio.wav',16000)
        print(wavform)
