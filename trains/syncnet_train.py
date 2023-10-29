from pathlib import Path

from models.SyncNetModel import SyncNetModel
from tqdm import tqdm
from wldatasets.SyncNetDataset import SyncNetDataset

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from process_util.ParamsUtil import ParamsUtil
import torch.multiprocessing
import argparse
from visualdl import LogWriter

param = ParamsUtil()

def load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer):

    print("load checkpoint from: {}".format(checkpoint_path))
    if param.gpunum > 0:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(checkpoint_path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["global_step"]
    epoch = checkpoint["global_epoch"]

    return model, step, epoch


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    path = Path(checkpoint_dir+'/sync_checkpoint_step{:09d}.pth'.format(step))
    optimizer_state = optimizer.state_dict() if param.save_optimizer_state else None
    torch.save({
        "state_dict":model.state_dict(),
        "optimizer":optimizer_state,
        "global_step":step,
        "global_epoch":epoch,
    },path)
    print("save the checkpoint step {}".format(path))


def eval_model(val_dataloader, global_step,device, model, checkpoint_dir):
    eval_steps = global_step
    losses = []
    with LogWriter(logdir="../logs/syncnet_train/eval") as writer:
        while 1:
            for vstep,(x,mel,y) in enumerate(val_dataloader):
                model.eval()

                x = x.to(device)
                mel = mel.to(device)
                y = y.to(device)

                a,v = model(mel,x)

                logloss = nn.BCELoss()
                d = F.cosine_similarity(a,v)
                loss = logloss(d.unsqueeze(1),y)

                losses.append(loss.item())

                if vstep > eval_steps: break

            averaged_loss = sum(losses)/len(losses)
            writer.add_scalar(tag='eval/loss', step=eval_steps, value=averaged_loss)
            print(averaged_loss)
            return


def train(device, model, train_dataloader, val_dataloader, optimizer, checkpoint_dir,start_step,start_epoch):

    gloab_step = start_step
    epoch = start_epoch
    numepochs = param.epochs
    checkpoint_interval = param.syncnet_checkpoint_interval
    eval_interval = param.syncnet_eval_interval

    with LogWriter(logdir="../logs/syncnet_train/train") as writer:
        while epoch < numepochs:
            running_loss = 0
            prog_bar = tqdm(enumerate(train_dataloader),total=len(train_dataloader),leave = False)
            for step,(x,mel,y) in prog_bar:
                model.train()
                optimizer.zero_grad()

                #transform data to cuda
                x= x.to(device)
                mel = mel.to(device)
                a,v = model(mel,x)
                y = y.to(device)

                #计算loss
                logloss = nn.BCELoss()
                d = F.cosine_similarity(a,v)
                loss = logloss(d.unsqueeze(1),y)
                loss.backward()

                optimizer.step()

                gloab_step=gloab_step+1
                running_loss = loss.item()

                if gloab_step % checkpoint_interval == 0:
                    save_checkpoint(model,optimizer,gloab_step,checkpoint_dir,epoch)

                if gloab_step % eval_interval == 0:
                    with torch.no_grad():
                        eval_model(val_dataloader,gloab_step,device,model,checkpoint_dir)

                prog_bar.set_description('Syncnet Train Epoch [{0}/{1}]'.format(epoch,numepochs))
                prog_bar.set_postfix(train_loss=running_loss/(step + 1),gloab_step=gloab_step)
                writer.add_scalar(tag='train/running_loss', step=gloab_step, value=running_loss)
                writer.add_scalar(tag='train/loss', step=gloab_step, value=running_loss / (step + 1))
            epoch +=1

def main():
    args = parse_args()

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = SyncNetDataset(args.data_root, run_type='train', img_size=288)
    val_dataset = SyncNetDataset(args.data_root, run_type='eval', img_size=288)

    train_dataloader = DataLoader(train_dataset, batch_size=param.syncnet_batch_size, shuffle=True,
                                  num_workers=param.num_works)
    val_dataloader = DataLoader(val_dataset, batch_size=param.syncnet_batch_size, shuffle=True,
                                num_workers=param.num_works)

    if args.gpunum > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = SyncNetModel().to(device)

    print("SyncNet Model's Total trainable params {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=param.syncnet_learning_rate)

    start_step = 0
    start_epoch =0

    if checkpoint_path is not None:
        model, start_step, start_epoch = load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    torch.multiprocessing.set_start_method('spawn')
    
    train(device,model,train_dataloader,val_dataloader,optimizer,checkpoint_dir,start_step,start_epoch)


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(
        description="train the syncnet model for wav2lip")
    parser.add_argument("--data_root", help='Root folder of the preprocessed dataset', required=True)
    parser.add_argument("--checkpoint_dir", help='Save checkpoints to this directory', required=True, type=str)
    parser.add_argument("--checkpoint_path", help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--config_file', help='The train config file', default='../configs/train_config_288.yaml',
                        required=True, type=str)
    parser.add_argument('--gpunum', help='Resume qulity disc from this checkpoint', default=0, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
