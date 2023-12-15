import torch
from torch import nn

from models.BaseTranspose import BaseTranspose
from models.BaseConv2D import BaseConv2D


class FaceCreator(nn.Module):

    def __init__(self):
        super(FaceCreator, self).__init__()

        self.face_encoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(6, 32, 7, 1, 3)),  # 输入形状 [5,6,288 288]


            nn.Sequential(BaseConv2D(32, 64, kernel_size=5, stride=2, padding=2), # 144 144
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(64, 64, kernel_size=3, stride=2, padding=1),  # 72 72
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(64, 128, kernel_size=3, stride=2, padding=1),  # 转成 35 36
                          BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(128, 256, kernel_size=3, stride=2, padding=1),  # 转成 18 18
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(256, 256, kernel_size=3, stride=2, padding=1),  # 9 9
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(256, 512, kernel_size=3, stride=2, padding=1),  # 5 5
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(BaseConv2D(512, 512, 3, 1, 0),),# 3
            # 1 1
            nn.Sequential(BaseConv2D(512, 512, 3, 1, 0),  # 1  1
                          BaseConv2D(512, 512, 1, 1, 0))
        ])

        self.audio_encoder = nn.Sequential(
            # [5,1,80,16]
            BaseConv2D(1, 32, kernel_size=3, stride=1, padding=1),
            BaseConv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(64, 128, kernel_size=3, stride=3, padding=1),
            BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(256, 512, kernel_size=3, stride=1, padding=0),
            BaseConv2D(512, 512, kernel_size=1, stride=1, padding=0)
        )

        self.face_decoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(512, 512, 1, 1, 0)),# 1 1


            nn.Sequential(BaseTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),# 3 3

            nn.Sequential(BaseTranspose(1024, 512, kernel_size=3, stride=2, padding=1),
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True)), # 5 5

            nn.Sequential(BaseTranspose(1024, 512, kernel_size=3, stride=2, padding=1),
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True)), # 9 9

            nn.Sequential(BaseTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(384, 384, kernel_size=3, stride=1, padding=1, residual=True)), # 18 18

            nn.Sequential(BaseTranspose(640, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(320, 320, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(320, 320, kernel_size=3, stride=1, padding=1, residual=True)), #36 36

            nn.Sequential(BaseTranspose(448, 224, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(224, 224, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(224, 224, kernel_size=3, stride=1, padding=1, residual=True)), #72 72

            nn.Sequential(BaseTranspose(288, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),  # 144 144

            nn.Sequential(BaseTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),# 288 288

        ])

        self.output_block = nn.Sequential(BaseConv2D(96, 32, kernel_size=3, stride=1, padding=1),
                                          BaseConv2D(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):

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
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print('exception got: {}'.format(e))
                print('audio size: {}'.format(x.size()))
                print('face size {}'.format(feats[-1].size()))
                raise e
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs
