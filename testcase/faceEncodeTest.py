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
        audios=torch.randn(1,5,1,80,16)
        faces =torch.randn(1,6,5,288,288)
        summary(fe,input_data=(audios,faces))
        sData = FaceDataset('../data/test_data/pr_data', img_size=288)
        test_loader = DataLoader(sData)
        for i,(x, indiv_mels, mel, gt) in enumerate(test_loader):
            gen_img = fe(indiv_mels,x)
            print('Face generator data shape {} , {}'.format(gen_img.shape,type(gen_img)))
            plt.imshow(gen_img.detach()[0][0][0])
            plt.show()

            disc_tmp = Discriminator()
            pred_real = disc_tmp(gt)
            pred_fake = disc_tmp(gen_img)
            print("disc real and fake image shape:{} {}".format(pred_real.shape,pred_fake.shape))
        #img = output.detach().numpy()
        #plt.imshow(img,interpolation='none',cmap='Blues')


