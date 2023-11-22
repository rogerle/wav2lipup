import torch
from torch import nn
from torch.nn import functional as F
from models.BaseConv2D import BaseConv2D


class SyncNetModel(nn.Module):

    def __init__(self):
        super(SyncNetModel, self).__init__()

        self.face_encoder = nn.Sequential(
            BaseConv2D(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            BaseConv2D(32, 32, kernel_size=5, stride=1, padding=1),
            BaseConv2D(32, 32, kernel_size=3, stride=1, padding=1),

            BaseConv2D(32, 64, kernel_size=5, stride=(1, 2), padding=1), #140 142
            BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(64, 128, kernel_size=3, stride=2, padding=1), #7072
            BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(128, 256, kernel_size=3, stride=2, padding=1), #35,36
            BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(256, 512, kernel_size=3, stride=2, padding=1),  #18 18
            BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            BaseConv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            BaseConv2D(512, 512, kernel_size=3, stride=2, padding=1),#9
            BaseConv2D(512, 512, kernel_size=3, stride=2, padding=0),#4
            BaseConv2D(512, 512, kernel_size=3, stride=1, padding=0),#2
            BaseConv2D(512, 512, kernel_size=2, stride=1, padding=0),#1
            BaseConv2D(512, 512, kernel_size=1, stride=1, padding=0),

        )

        self.audio_encoder = nn.Sequential(
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

    def forward(self,audio_sequences,face_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0),-1)
        face_embedding = face_embedding.view(face_embedding.size(0),-1)

        audio_embedding = F.normalize(audio_embedding,p=2,dim=1)
        face_embedding = F.normalize(face_embedding,p=2,dim=1)

        return audio_embedding,face_embedding
