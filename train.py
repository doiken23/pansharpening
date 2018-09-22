import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import transforms

from src.models import PanUNet, resnet18
from src.models import Criterion
from src.utils import PansharpenDataset
from torchcv.transforms import NPSegRandomFlip, NPSegRandomRotate

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, choices=['unet', 'resnet'])
parser.add_argument('data', type=str, help='root data dir')
parser.add_argument('--log', type=str, default='out', help='output path')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--alpha', type=float, default=1.0, help='parameter of loss weight')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# set torch parameters
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

# prepare dataset
statistics = joblib.load(Path(args.data).joinpath('statistic', 'statistic.pkl'))
train_trans = transforms.Compose([
    NPSegRandomFlip(),
    NPSegRandomRotat(),
    transforms.Normalize(statistics['mean'], statistics['std'])
    )
val_trans = trnsforms.Normalize(statistics['mean'], statistics['std'])
train_dataset, val_dataset = make_pansharpen_dataset(Path(args.data).joinpath('out'), train_trasforms=train_trans, val_transforms=val_trans)
train_loader = data_utils.DataLoader(
    dataset=train_dataset,
    batch_size=args.batchsize, shuffle=True,
    drop_last=True,
    num_workers=2)
val_loader = data_utils.DataLoader(
    dataset=val_dataset,
    batch_size=args.batchsize,
    drop_last=True,
    num_workers=2)

# training settings
if args.model == 'resnet':
    net = resnet18(4, 3, batch_norm=True)
elif args.model == 'unet':
    net = PanUNet(4, 3)
net.to(device)

criterion = Criterion(net, 3, alpha=args.alpha).to(device)
optimzer = optim.Adam(criterion.parameters(), lr=args.lr)

def train(epoch):
    criterion.train()
    epoch_loss = 0
    for data in train_loader:
        rgb, b8 = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        loss = criterion(rgb, b8)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('=== Epoch {}/{} ==='.format(epoch, args.epochs))
    print('Train Avg. Loss: {:.4f}'.format(epoch_loss / len(training_data_loader)))

    return epoch_loss / len(train_loader)

def test():
    criterion.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in val_loader:
            rgb, b8 = data[0].to(device), data[1].to(device)
            
            loss = criterion(rgb, b8)
            epoch_loss += loss.item()

    print("Test Avg. Loss: {:.4f}".format(epoch_loss / len(val_loader)))

    return epoch_loss / len(val_loader)

# set evaluation score
min_loss = 999.999
min_loss_model = net
loss_his = [[], []]
for epoch in range(1, args.epochs+1):
    train_loss = train(epoch)
    loss_his[0].append(train_loss)
    val_loss = test()
    loss_his[1].append(val_loss)
    if val_loss < min_loss:
        torch.save(model, Path(args.out).joinpath('min_loss_model.pth'))
        min_loss = val_loss
np.save(Path(args.out).joinpath('training_history.npy'), np.array(loss_his))    
