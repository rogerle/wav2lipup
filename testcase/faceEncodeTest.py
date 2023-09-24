import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from models.BaseConv2D import BaseConv2D
import matplotlib.pyplot as plt

from models.FaceCreator import FaceCreator
from wavdatas.FaceDataset import FaceDataset


class FaceEncode(unittest.TestCase):

    def testfaceEncode(self):
        fe = FaceCreator()

        # andom_data = torch.randn([1,3,96,96])
        # output = fe.forward(andom_data)
        sData = FaceDataset('../data/test_data/pr_data', img_size=288)
        test_loader = DataLoader(sData)
        for x,y,mel1,invid_mels in test_loader:
            summary(fe,audio_sequences=mel1,face_sequences=y)

        print(torch.cuda.is_available())
        print(torch.cuda.memory_summary(0,abbreviated=True))

        #img = output.detach().numpy()
        #plt.imshow(img,interpolation='none',cmap='Blues')

