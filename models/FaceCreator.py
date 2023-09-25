import torch
from torch import nn

from models.BaseTranspose import BaseTranspose
from models.BaseConv2D import BaseConv2D


class FaceCreator(nn.Module):

    def __init__(self):
        super(FaceCreator,self).__init__()

        self.face_encoder_block = nn.ModuleList([
                nn.Sequential(BaseConv2D(6, 16, 7, 1, 3)), #输入大小 [5 6 288 288]

                # 转成 48 48
                nn.Sequential(BaseConv2D(16, 32, 3, 2, 1),
                              BaseConv2D(32, 32, 3, 1, 1, residual=True)),
                # 转成 24 24
                nn.Sequential(BaseConv2D(32, 64, 3, 2, 1),
                              BaseConv2D(64, 64, 3, 1, 1, residual=True),
                              BaseConv2D(64, 64, 3, 1, 1, residual=True),
                              BaseConv2D(64, 64, 3, 1, 1, residual=True)),
                # 转成 12 12
                nn.Sequential(BaseConv2D(64, 128, 3, 2, 1),
                              BaseConv2D(128, 128, 3, 1, 1, residual=True),
                              BaseConv2D(128, 128, 3, 1, 1, residual=True)),

            nn.Sequential(BaseConv2D(128, 256, 3, 2, 1),
                          BaseConv2D(256, 256, 3, 1, 1, residual=True),
                          BaseConv2D(256, 256, 3, 1, 1, residual=True)),

            nn.Sequential(BaseConv2D(256, 512, 3, 2, 1),
                          BaseConv2D(512, 512, 3, 1, 1, residual=True)),

            nn.Sequential(BaseConv2D(512, 512, 3, 1, 0),
                              BaseConv2D(512, 512, 1, 1, 0))
                ])

        self.audio_encoder = nn.Sequential(
            BaseConv2D(1, 32, 3, 1, 1),
            BaseConv2D(32, 32, 3, 1, 1,residual=True),
            BaseConv2D(32, 32, 3, 1, 1,residual=True),

            BaseConv2D(32, 64, 3, (3, 1), 1),
            BaseConv2D(64, 64, 3, 1, 1, residual=True),
            BaseConv2D(64, 64, 3, 1, 1, residual=True),

            BaseConv2D(64, 128, 3, 3, 1),
            BaseConv2D(128, 128, 3, 1, 1, residual=True),
            BaseConv2D(128, 128, 3, 1, 1, residual=True),

            BaseConv2D(128, 256, 3, (3,2), 1),
            BaseConv2D(256, 256, 3, 1, 1, residual=True),

            BaseConv2D(256, 512, 3, 1, 0),
            BaseConv2D(512, 512, 1, 1, 0)
        )

        self.face_decoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(512, 512, 1, 1, 0)), #3*3

            nn.Sequential(BaseTranspose(1024, 512, 3, 2, 1, 1),
                          BaseConv2D(512, 512, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(1024, 512, 3, 2, 1, 1),
                          BaseConv2D(512, 512, 3, 1, 1, residual=True),
                          BaseConv2D(512, 512, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(768, 384, 3, 2, 1, 1),
                          BaseConv2D(384, 384, 3, 1, 1, residual=True),
                          BaseConv2D(384, 384, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(512, 256, 3, 2, 1, 1),
                          BaseConv2D(256, 256, 3, 1, 1, residual=True),
                          BaseConv2D(256, 256, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(320, 128, 3, 2, 1, 1),
                          BaseConv2D(128, 128, 3, 1, 1, residual=True),
                          BaseConv2D(128, 128, 3, 1, 1, residual=True)),

            nn.Sequential(BaseTranspose(160, 64, 3, 2, 1, 1),
                          BaseConv2D(64, 64, 3, 1, 1, residual=True),
                          BaseConv2D(64, 64, 3, 1, 1, residual=True)),

        ])

        self.output_block = nn.Sequential(BaseConv2D(80, 32, 3, 1, 1),
                                          BaseConv2D(32, 3, 1, 1, 0),
                                          nn.Sigmoid())

    def forward(self,audio_sequences,face_sequences):

        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)

        feats = []
        x = face_sequences
        for f in self.face_encoder_block:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_block:
            x = f(x)
            try:
                x = torch.cat((x,feats[-1]),dim=1)
            except Exception as e:
                print('exception got: {}'.format(e))
                print('audio size: {}'.format(x.size()))
                print('face size {}'.format(feats[-1].size()))
                raise e
            feats.pop()

        x = self.output_block(x)

        if input_dim_size >4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x,dim=2)
        else:
            outputs = x

        return outputs