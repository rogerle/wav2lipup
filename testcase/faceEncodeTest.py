import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from models.BaseConv2D import BaseConv2D
import matplotlib.pyplot as plt

from models.Discriminator import Discriminator
from models.FaceCreator import FaceCreator
from wldatasets.FaceDataset import FaceDataset


class FaceEncode(unittest.TestCase):

    def testfaceEncode(self):
        fe = FaceCreator()
        # random_data = torch.randn([1,3,96,96])
        # output = fe.forward(random_data)
        audios=torch.randn(5,5,1,80,16)
        faces =torch.randn(5,6,5,288,288)
        summary(fe,input_data=(audios,faces))




