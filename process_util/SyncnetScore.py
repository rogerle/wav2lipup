from pathlib import Path

import torch

from models.SyncNetModel import SyncNetModel

"""
    利用已经训练过的syncnet的来判断数据的阈值，高于阈值的数据丢弃
"""


class PreProcessor():
    def __init__(self, data_root, default_threshold, checkpoint_pth):
        self.data_root = data_root
        self.dt = default_threshold
        self.checkpoint_pth = checkpoint_pth
    def __load_checkpoint(self,model):
        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_pth)
        else:
            checkpoint = torch.load(self.checkpoint_pth, map_location=lambda storage, loc: storage)

        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        return model

    def score_video(self):
        root = self.data_root + '/processed'

        dir_list = []
        for dir in Path.rglob(Path(root), '*'):
            if dir.is_dir():
                dir_list.append(str(dir))

        # 开始对每个目录评分
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        syncnet = SyncNetModel().to(device)
        for p in syncnet.parameters():
            p.requires_grad = False
        syncnet = self.__load_checkpoint(syncnet)

        for dir in dir_list:
            w = self.__getwindows(root+'/'+dir)

    def __getwindows(self, dir):
        pass



