from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import numpy as np
from random import choice
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--data', default='data/ImageNet', metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch_size', default=50, type=int, metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--size', default=224, type=int, metavar='N', help='the shape of resized image')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')

torch.backends.cudnn.benchmark = False
def main():
    global args
    feature = []
    # Data loading code
    args = parser.parse_args()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    for i, (x, y) in tqdm(enumerate(val_loader)):
        front_20 = x[:20].cuda()
        feature.append(front_20.detach().cpu().numpy())
    feature = np.array(feature)
    np.save('/data/ImageNet_gallery.npy', feature)

if __name__ == "__main__":
    main()
