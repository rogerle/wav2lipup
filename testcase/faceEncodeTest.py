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
        summary(fe,(1,3,288,288))
        print(torch.cuda.is_available())
        print(torch.cuda.memory_summary(0,abbreviated=True))

        #img = output.detach().numpy()
        #plt.imshow(img,interpolation='none',cmap='Blues')

