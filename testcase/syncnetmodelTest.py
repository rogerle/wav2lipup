import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from models.BaseConv2D import BaseConv2D

from models.SyncNetModel import SyncNetModel
from wldatasets.SyncNetDataset import SyncNetDataset

class TestSyncnetModel(unittest.TestCase):

    def testModelSummary(self):
        model = SyncNetModel()
        faces = torch.randn(5,15,144,288)
        audios = torch.randn(5,1,80,16)
        summary(model,input_data=(audios,faces),)