import os, argparse

from torch import optim
from torch.utils.data import DataLoader

from models.FaceCreator import FaceCreator
from process_util.ParamsUtil import ParamsUtil
from wavdatas.FaceDataset import FaceDataset

parser = argparse.ArgumentParser(description='code to train the wav2lip with visual quality discriminator')
parser.add_argument('--data_root',help='Root folder of the preprocessed datasets',required=True,type=str)
parser.add_argument('--checkpoint_dir',help='checkpoint files will be saved to this directory',required=True,type=str)
parser.add_argument('syncnet_checkpoint_path',help='Load he pre-trained Expert discriminator',required=True,type=str)

parser.add_argument('checkpoint',help='Resume generator from this checkpoint',default=None,type=str)
parser.add_argument('--disc_checkpoint_path',help='Resume qulity disc from this checkpoint',default=None,type=str)

parser.add_argument('--gpunum',help='Resume qulity disc from this checkpoint',default=0,type=int)
args = parser.parse_args()
param = ParamsUtil

if __name__ == "__main__":
    # checkpoint setup
    checkpoint_dir = args.checkpoint_dir


    #dataset
    train_dataset = FaceDataset(args.data_root,type='train',img_size=param.img_size)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=param.batch_size,
                                   shuffle=True,
                                   num_workers=param.num_wokers)
    if args.gpunum == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    model = FaceCreator().to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=param.init_learning_rate, betas=(0.5,0.999))
