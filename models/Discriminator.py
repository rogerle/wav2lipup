from torch import nn
import torch
from torch.nn import functional as F

from models.BaseConv2D import BaseConv2D
from models.NoNormConv import NoNormConv


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(NoNormConv(3, 32, kernel_size=7, stride=1, padding=3)),  # 144,288

            nn.Sequential(NoNormConv(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 144,144
                          nn.MaxPool2d(kernel_size=5,stride=1,padding=2)),

            nn.Sequential(NoNormConv(64, 128, kernel_size=3, stride=2, padding=1),  # 72,72
                          nn.MaxPool2d(kernel_size=3,stride=1,padding=1)),

            nn.Sequential(NoNormConv(128, 256, kernel_size=3, stride=2, padding=1),  # 36,36
                          nn.MaxPool2d(kernel_size=3,stride=1,padding=1)),

            nn.Sequential(NoNormConv(256, 512, kernel_size=3, stride=2, padding=1),  # 18,18
                          nn.MaxPool2d(kernel_size=3,stride=1,padding=1)),

            nn.Sequential(NoNormConv(512, 512, kernel_size=3, stride=2, padding=1),  # 9,9
                          nn.MaxPool2d(kernel_size=5,stride=1,padding=2)),

            nn.Sequential(NoNormConv(512, 512, kernel_size=3, stride=2, padding=0),  # 4,4
                          nn.MaxPool2d(kernel_size=3,stride=1,padding=1)),

            nn.Sequential(NoNormConv(512, 512, kernel_size=3, stride=1, padding=0),   # 3,3
                          nn.MaxPool2d(kernel_size=1,stride=1,padding=0)),

            nn.Sequential(NoNormConv(512, 512, kernel_size=2, stride=1, padding=0),  # 1, 1
                          nn.MaxPool2d(kernel_size=1,stride=1,padding=0))])

        #self.binary_pred = nn.Sequential(BaseConv2D(512, 1, kernel_size=1, stride=1, padding=0,act='sigmoid'))
        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid(),
                                         nn.MaxPool2d(kernel_size=1,stride=1,padding=0))
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def to_2d(self, face_sequences):

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = torch.ones(len(false_feats), 1,dtype=torch.float).to(device)
        x = self.binary_pred(false_feats).view(false_feats.size(0),-1)
        try:
            false_pred_loss = F.binary_cross_entropy_with_logits(x,y)
        except Exception as e:
            print('x value:{}'.format(x))
            raise e

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
        x = self.binary_pred(x).view(x.size(0), -1)
        return x
