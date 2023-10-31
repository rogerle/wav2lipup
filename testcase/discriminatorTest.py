import unittest
import torch
from torchinfo import summary
from models.Discriminator import Discriminator


class TestDiscriminator(unittest.TestCase):
    def test_modelNet(self):
        disc = Discriminator()
        faces =torch.randn(1,3,5,288,288)
        summary(disc, input_data=(faces))
