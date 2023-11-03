import unittest
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader

from models.FaceCreator import FaceCreator
from models.Discriminator import Discriminator
from models.SyncNetModel import SyncNetModel
from process_util.ParamsUtil import ParamsUtil
from trains import wl_train
from wldatasets.SyncNetDataset import SyncNetDataset
from wldatasets.FaceDataset import FaceDataset

class W2LtrainTest(unittest.TestCase):
    def test_train(self):
        param = ParamsUtil()
        data_root='../data/test_data/pr_data'
        checkpoint_dir = '../data/test_data/checkpoint'
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        disc_checkpoint_path = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        syncnet_checkpoint_path='../data/syncnet_checkpoint/sync_checkpoint_step000033000.pth'

        train_dataset = FaceDataset(data_root, run_type='train', img_size=param.img_size)
        test_dataset = FaceDataset(data_root, run_type='eval', img_size=param.img_size)

        train_data_loader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True,
                                       num_workers=param.num_works)

        test_data_loader = DataLoader(test_dataset, batch_size=param.batch_size,
                                      num_workers=param.num_works)


        model=FaceCreator()
        disc = Discriminator()
        syncnet = SyncNetModel()
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=param.init_learning_rate, betas=(0.5, 0.999))

        disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                                    lr=param.disc_initial_learning_rate, betas=(0.5, 0.999))



        start_step = 0
        start_epoch = 0

        # 装在sync_net
        syncnet, step, epoch = wl_train.load_checkpoint(syncnet_checkpoint_path, syncnet, None, reset_optimizer=True)

        wl_train.train(device, model, disc, syncnet, train_data_loader, test_data_loader, optimizer, disc_optimizer,
                        checkpoint_dir=checkpoint_dir, start_step=start_step, start_epoch=start_epoch)