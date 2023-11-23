import unittest
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.SyncNetModel import SyncNetModel
from process_util.ParamsUtil import ParamsUtil
from trains import syncnet_train
from wldatasets.SyncNetDataset import SyncNetDataset


class TestSyncnetTrain(unittest.TestCase):

    def testGPUTrain(self):
        device=torch.device('cuda'if torch.cuda.is_available() else'cpu')
        data_root='../data/test_data/pr_data'
        train_txt = data_root + '/train.txt'
        eval_txt = data_root + '/eval.txt'
        Path(train_txt).write_text('')
        Path(eval_txt).write_text('')
        for line in Path.glob(Path(data_root), '*/*'):
            if line.is_dir():
                dirs = line.parts
                input_line = str(dirs[-2] + '/' + dirs[-1])
                with open(train_txt, 'a') as f:
                    f.write(input_line + '\n')
                with open(eval_txt, 'a') as f:
                    f.write(input_line + '\n')

        param = ParamsUtil()
        train_dataset = SyncNetDataset(data_root, run_type='train', img_size=288)
        val_dataset = SyncNetDataset(data_root, run_type='eval', img_size=288)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                                      num_workers=8,drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=2,num_workers=8,drop_last=True)

        model = SyncNetModel().to(device)
        print("SyncNet Model's Total trainable params {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(param.syncnet_learning_rate))
        #optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(param.syncnet_learning_rate))
        start_step = 0
        start_epoch = 0
        syncnet_train.train(device, model, train_dataloader, val_dataloader, optimizer, '../data/test_data/checkpoint', start_step, start_epoch)

