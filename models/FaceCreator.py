import torch
from torch import nn

from models.BaseTranspose import BaseTranspose
from models.BaseConv2D import BaseConv2D


class FaceCreator(nn.Module):

    def __init__(self):
        super(FaceCreator,self).__init__()

        self.face_encoder_block = nn.ModuleList([
                nn.Sequential(BaseConv2D(6, 16, 7, 1, 3)), #输入大小 288*288
                # 转成 144*144
                nn.Sequential(BaseConv2D(16, 64, 3, 2, 1),
                              BaseConv2D(64, 64, 3, 1, 1, residual=True)),
                # 转成 24*24
                nn.Sequential(BaseConv2D(64, 128, 3, 2, 1),
                              BaseConv2D(128, 128, 3, 1, 1, residual=True)),
                # 转成 12*12
                nn.Sequential(BaseConv2D(128, 256, 3, 2, 1),
                              BaseConv2D(256, 256, 3, 1, 1, residual=True)),
                # 转成 6*6
                nn.Sequential(BaseConv2D(256, 512, 3, 2, 1),
                              BaseConv2D(512, 512, 3, 1, 1, residual=True)),
                # 转成 3*3
                nn.Sequential(BaseConv2D(512, 1024, 3, 2, 1),
                              BaseConv2D(1024, 1024, 3, 1, 1, residual=True)),
                # 转成 1*1
                nn.Sequential(BaseConv2D(1024, 2048, 3, 2, 1),
                              BaseConv2D(2048, 2048, 3, 1, 1, residual=True)),

                # 转成 2*2
                nn.Sequential(BaseConv2D(2048, 4096, 3, 2, 1),
                              BaseConv2D(4096, 4096, 3, 1, 1, residual=True)),

                nn.Sequential(BaseConv2D(4096, 8192, 2, 1, 0),
                              BaseConv2D(8192, 8192, 1, 1, 0))
                ])

        self.face_decoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(8192, 8192, 1, 1, 0)), #3*3

            nn.Sequential(BaseTranspose(8192, 4096, 3, 1, 0),
                          BaseConv2D(4096, 4096, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(4096, 2048, 3, 2, 1, 1),
                          BaseConv2D(2048, 2048, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(2048, 1024, 3, 2, 1,1),
                          BaseConv2D(1024, 1024, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(1024, 512, 3, 2, 1, 1),
                          BaseConv2D(512, 512, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(512, 256, 3, 2, 1, 1),
                          BaseConv2D(256, 256, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(256, 128, 3, 2, 1, 1),
                          BaseConv2D(128, 128, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(128, 64, 3, 2, 1, 1),
                          BaseConv2D(64, 64, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(64, 32, 3, 2, 1, 1),
                          BaseConv2D(32, 32, 3, 1, 1, residual=True)),

        ])

        self.output_block = nn.Sequential(BaseConv2D(32, 16, 3, 1, 1),
                                          BaseConv2D(16, 3, 1, 1, 0),
                                          nn.Sigmoid())

    def forward(self,x):
        y=x
        for f in self.face_encoder_block:
            y = f(y)
        z=y
        for f in self.face_decoder_block:
            z = f(z)
            print(z.shape)
        x = self.output_block(z)
        return x