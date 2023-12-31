import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualdl import LogWriter

from models.Discriminator import Discriminator
from models.FaceCreator import FaceCreator
from models.SyncNetModel import SyncNetModel
from process_util.ParamsUtil import ParamsUtil
from wldatasets.FaceDataset import FaceDataset


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
# 判断是否使用gpu
import os

param = ParamsUtil()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

syncnet = SyncNetModel().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

logloss = nn.BCEWithLogitsLoss()
# logloss = nn.BCELoss()
recon_loss = nn.L1Loss()


def load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False):
    print("Load checkpoint from: {}".format(checkpoint_path))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(checkpoint_path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["global_step"]
    epoch = checkpoint["global_epoch"]

    return model, step, epoch


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3) // 2:]
    g = torch.cat([g[:, :, i] for i in range(param.syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1, dtype=torch.float).to(device)
    return cosine_loss(a, v, y)


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = checkpoint_dir + "/samples_step_{:09d}".format(global_step)
    Path(folder).mkdir(parents=True, exist_ok=True)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])
    return collage


def save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = checkpoint_dir + "/{}checkpoint_step{:09d}.pth".format(prefix, global_step)
    optimizer_state = optimizer.state_dict() if param.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def eval_model(test_data_loader, model, disc):
    eval_steps = 300
    print('Evaluating for {} steps:'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    while 1:
        for step, (x, indiv_mels, mel, gt) in enumerate(test_data_loader):
            model.eval()
            disc.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((pred.size(0), 1)).to(device))

            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((pred.size(0), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())

            sync_loss = get_sync_loss(mel=mel, g=g)

            if param.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)
            loss = float(param.syncnet_wt) * sync_loss + float(param.disc_wt) * perceptual_loss + \
                   (1. - float(param.syncnet_wt) - float(param.disc_wt)) * l1loss
            running_l1_loss.append(loss.item())
            running_sync_loss.append(sync_loss.item())

            if param.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if step > eval_steps: break

        print('L1: {}, \n Sync: {}, \n Percep: {} | Fake: {}, Real: {}'.format(
            sum(running_l1_loss) / len(running_l1_loss),
            sum(running_sync_loss) / len(
                running_sync_loss),
            sum(running_perceptual_loss) / len(
                running_perceptual_loss),
            sum(running_disc_fake_loss) / len(
                running_disc_fake_loss),
            sum(running_disc_real_loss) / len(
                running_disc_real_loss)))
        eval_loss = sum(running_sync_loss) / len(running_sync_loss)

        return eval_loss


def train(model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,checkpoint_dir,
          start_step, start_epoch):
    global_step = start_step
    epoch = start_epoch
    num_epochs = param.epochs
    checkpoint_interval = param.checkpoint_interval
    eval_interval = param.eval_interval
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[param.m_min,param.m_med,param.m_max],gamma=0.1)
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    with LogWriter(logdir="../logs/wav2lip/train") as writer:
        while epoch < num_epochs:
            running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
            running_disc_real_loss, running_disc_fake_loss = 0., 0.
            prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False)

            for step, (x, indiv_mels, mel, gt) in prog_bar:
                disc.train()
                model.train()

                x = x.to(device)
                mel = mel.to(device)
                indiv_mels = indiv_mels.to(device)
                gt = gt.to(device)

                ### Train generator now. Remove ALL grads.
                optimizer.zero_grad()
                disc_optimizer.zero_grad()

                g = model(indiv_mels, x)

                if float(param.syncnet_wt) > 0.:
                    sync_loss = get_sync_loss(mel, g)
                else:
                    sync_loss = torch.tensor(0, dtype=torch.float)

                if float(param.disc_wt) > 0.:
                    perceptual_loss = disc.perceptual_forward(g)
                else:
                    perceptual_loss = torch.tensor(0, dtype=torch.float)
                # print ("g:{}|gt:{}".format(g.shape,gt.shape))
                l1loss = recon_loss(g, gt)

                loss = float(param.syncnet_wt) * sync_loss + float(param.disc_wt) * perceptual_loss + \
                       (1. - float(param.syncnet_wt) - float(param.disc_wt)) * l1loss

                loss.backward()
                optimizer.step()


                # Remove all gradients before Training disc
                disc_optimizer.zero_grad()

                pred = disc(gt)
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones(pred.size(0), 1).to(device))
                disc_real_loss.backward()

                pred = disc(g.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros(pred.size(0), 1).to(device))
                disc_fake_loss.backward()

                disc_optimizer.step()

                running_disc_real_loss += disc_real_loss.item()
                running_disc_fake_loss += disc_fake_loss.item()

                if global_step % checkpoint_interval == 0:
                    collage = save_sample_images(x, g, gt, global_step, checkpoint_dir)
                    for batch_idx, c in enumerate(collage):
                        for t in range(0,c.shape[0]-1):
                            x = cv2.cvtColor(c[t], cv2.COLOR_RGB2BGR)
                            writer.add_image(tag='train/sample', img=x/ 255., step=global_step)

                global_step += 1

                running_l1_loss += l1loss.item()
                if float(param.syncnet_wt) > 0.:
                    running_sync_loss += sync_loss.item()
                else:
                    running_sync_loss += 0.

                if param.disc_wt > 0.:
                    running_perceptual_loss += perceptual_loss.item()
                else:
                    running_perceptual_loss += 0.

                if global_step == 1 or global_step % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, epoch)
                    save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, epoch, prefix='disc_')

                if global_step % eval_interval == 0:
                    with torch.no_grad():
                        average_sync_loss = eval_model(test_data_loader, model, disc)
                        writer.add_scalar(tag='train/eval_loss', step=global_step, value=average_sync_loss)
                        if average_sync_loss < .75:
                            param.set_param('syncnet_wt', 0.01)
                prog_bar.set_description('Syncnet Train Epoch [{0}/{1}]'.format(epoch, num_epochs))
                prog_bar.set_postfix(Step=global_step,
                                     L1=running_l1_loss / (step + 1),
                                     lr=lr,
                                     Sync=running_sync_loss / (step + 1),
                                     Percep=running_perceptual_loss / (step + 1),
                                     Fake=running_disc_fake_loss / (step + 1),
                                     Real=running_disc_real_loss / (step + 1))
                writer.add_scalar(tag='train/L1_loss', step=global_step, value=running_l1_loss / (step + 1))
                writer.add_scalar(tag='train/Sync_loss', step=global_step, value=running_sync_loss / (step + 1))
                writer.add_scalar(tag='train/Percep_loss', step=global_step, value=running_perceptual_loss / (step + 1))
                writer.add_scalar(tag='train/Real_loss', step=global_step, value=running_disc_real_loss / (step + 1))
                writer.add_scalar(tag='train/Fake_loss', step=global_step, value=running_disc_fake_loss / (step + 1))
            #scheduler.step()
            #lr = optimizer.state_dict()['param_groups'][0]['lr']
            epoch += 1


def main():
    args = parse_args()

    # 创建checkpoint目录
    checkpoint_dir = args.checkpoint_dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    disc_checkpoint_path = args.disc_checkpoint_path
    syncnet_checkpoint_path = args.syncnet_checkpoint_path
    train_type = args.train_type

    train_dataset = FaceDataset(args.data_root, run_type=train_type, img_size=param.img_size)
    test_dataset = FaceDataset(args.data_root, run_type='eval', img_size=param.img_size)

    train_data_loader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True,
                                   num_workers=param.num_works, drop_last=True)

    test_data_loader = DataLoader(test_dataset, batch_size=param.batch_size,
                                  num_workers=param.num_works, drop_last=True)

    model = FaceCreator()
    disc = Discriminator()

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(param.init_learning_rate), betas=(0.5, 0.999))

    disc_optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                lr=float(param.init_learning_rate), betas=(0.5, 0.999))
    start_step = 0
    start_epoch = 0

    if checkpoint_path is not None:
        model, start_step, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    if disc_checkpoint_path is not None:
        disc, start_step, start_epoch = load_checkpoint(disc_checkpoint_path, disc, disc_optimizer,
                                                        reset_optimizer=False)
    cuda_ids = [int(d_id) for d_id in os.environ.get('CUDA_VISIBLE_DEVICES').split(',')]
    model = MyDataParallel(model,device_ids=cuda_ids)
    model.to(device)

    disc = MyDataParallel(disc,device_ids=cuda_ids)
    disc.to(device)


    # 装在sync_net
    load_checkpoint(syncnet_checkpoint_path, syncnet, None, reset_optimizer=True)

    train(model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=checkpoint_dir, start_step=start_step, start_epoch=start_epoch)


def parse_args():
    # parse args and config

    parser = argparse.ArgumentParser(description='code to train the wav2lip with visual quality discriminator')
    parser.add_argument('--data_root', help='Root folder of the preprocessed datasets', required=True, type=str)
    parser.add_argument('--checkpoint_dir', help='checkpoint files will be saved to this directory', required=True,
                        type=str)
    parser.add_argument('--syncnet_checkpoint_path', help='Load he pre-trained Expert discriminator', required=True,
                        type=str)
    parser.add_argument('--checkpoint_path', help='Load he pre-trained ', required=False,
                        type=str)
    parser.add_argument('--checkpoint', help='Resume generator from this checkpoint', default=None, type=str)
    parser.add_argument('--disc_checkpoint_path', help='Resume qulity disc from this checkpoint', default=None,
                        type=str)
    parser.add_argument('--train_type', help='the train tyep train or test', default='train',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
