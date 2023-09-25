import unittest

import torch
from torch import nn
from torchinfo import summary

from models.BaseConv2D import BaseConv2D
import matplotlib.pyplot as plt

from models.FaceCreator import FaceCreator


class FaceEncode(unittest.TestCase):

    def testfaceEncode(self):
        fe = FaceCreator()

        # andom_data = torch.randn([1,3,96,96])
        # output = fe.forward(andom_data)
        summary(fe,(1,6,288,288))

        #img = output.detach().numpy()
        #plt.imshow(img,interpolation='none',cmap='Blues')

