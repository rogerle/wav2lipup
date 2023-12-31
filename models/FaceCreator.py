import torch
from torch import nn

from models.BaseTranspose import BaseTranspose
from models.BaseConv2D import BaseConv2D


class FaceCreator(nn.Module):

    def __init__(self):
        super(FaceCreator, self).__init__()

        self.face_encoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(6, 16, kernel_size=7, stride=1, padding=3, act='relu')),  # 输入形状 [5,6,288 288]

            nn.Sequential(BaseConv2D(16, 32, kernel_size=5, stride=2, padding=2, act='relu'), # 144 144
                          BaseConv2D(32, 32, kernel_size=5, stride=1, padding=2, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=5, stride=1, padding=2)),

            nn.Sequential(BaseConv2D(32, 64, kernel_size=3, stride=2, padding=1, act='relu'),  # 72 72
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

            nn.Sequential(BaseConv2D(64, 128, kernel_size=3, stride=2, padding=1, act='relu'),  # 转成 36 36
                          BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

            nn.Sequential(BaseConv2D(128, 256, kernel_size=3, stride=2, padding=1, act='relu'),  # 转成 18 18
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),

            nn.Sequential(BaseConv2D(256, 512, kernel_size=3, stride=2, padding=1, act='relu'),
                          BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),  # 9 9

            nn.Sequential(BaseConv2D(512, 512, kernel_size=3, stride=2, padding=0, act='relu'),
                          nn.MaxPool2d(kernel_size=1, stride=1, padding=0)),  # 4 4
            nn.Sequential(BaseConv2D(512, 512, kernel_size=3, stride=1, padding=0, act='relu'),
                          nn.MaxPool2d(kernel_size=1, stride=1, padding=0)),  # 2 2
            nn.Sequential(BaseConv2D(512, 512, kernel_size=2, stride=1, padding=0, act='relu'),
                          nn.MaxPool2d(kernel_size=1, stride=1, padding=0))  # 1 1
        ])

        self.audio_encoder = nn.Sequential(
            # [5,1,80,16]
            BaseConv2D(1, 32, kernel_size=3, stride=1, padding=1, act='relu'),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            BaseConv2D(32, 64, kernel_size=3, stride=(3, 1), padding=1, act='relu'),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            BaseConv2D(64, 128, kernel_size=3, stride=3, padding=1, act='relu'),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            BaseConv2D(128, 256, kernel_size=3, stride=(3, 2), padding=1, act='relu'),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            BaseConv2D(256, 512, kernel_size=3, stride=1, padding=0, act='relu'),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            BaseConv2D(512, 512, kernel_size=1, stride=1, padding=0, act='relu'),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

        self.face_decoder_block = nn.ModuleList([
            nn.Sequential(BaseConv2D(512, 512, kernel_size=1, stride=1, padding=0, act='relu')),  # 1 1

            nn.Sequential(BaseTranspose(1024, 512, kernel_size=2, stride=1, padding=0), ),  # 2 2
            nn.Sequential(BaseTranspose(1024, 512, kernel_size=3, stride=1, padding=0), ),  # 4 4
            nn.Sequential(BaseTranspose(1024, 256, kernel_size=3, stride=2, padding=0), ),  # 9 9

            nn.Sequential(BaseTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(384, 384, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ),  # 18 18

            nn.Sequential(BaseTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ),  # 36 36

            nn.Sequential(BaseTranspose(384, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ),  # 72 72

            nn.Sequential(BaseTranspose(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ),  # 144 144

            nn.Sequential(BaseTranspose(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, act='relu'),
                          nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ),  # 288 288

        ])

        self.output_block = nn.Sequential(BaseConv2D(80, 32, kernel_size=3, stride=1, padding=1, act='relu'),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid(),
                                          nn.MaxPool2d(kernel_size=1, stride=1, padding=0))

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
