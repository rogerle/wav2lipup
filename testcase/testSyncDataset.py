import unittest
from torch.utils.data import DataLoader
from process_util.ParamsUtil import ParamsUtil
from wldatasets.SyncNetDataset import SyncNetDataset


class TestSyncDataset(unittest.TestCase):

    def test_getItem(self):
        hp = ParamsUtil()
        sData = SyncNetDataset('../data/test_data/pr_data', img_size=288)
        test_loader = DataLoader(sData)
        for x,mel,y in test_loader:
            print("matrix x's size:{}".format(x.size()))
            print("matrix y size:{}".format(y.size()))
            print("matrix mel1's size:{}".format(mel.size()))
