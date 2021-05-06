from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import random
import numpy as np
import os
import math
import argparse
from random import choice
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from Normalize import Normalize
import cv2
import math

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--data', default='data/ImageNet_chosen_dn-res-incv3-vgg19_5perclass', metavar='DIR', help='path to dataset')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument('--eps', default=0.07, type=float, metavar='N', help='epsilon for attack perturbation')
parser.add_argument('--decay', default=1.0, type=float, metavar='N', help='decay for attack momentum')
parser.add_argument('--iteration', default=20, type=int, metavar='N', help='number of attack iteration')
parser.add_argument('-b', '--batchsize', default=1, type=int, metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--size', default=224, type=int, metavar='N', help='the size of image')
parser.add_argument('--resize', default=299, type=int, metavar='N', help='the resize of image')
parser.add_argument('--prob', default=0.5, type=float, metavar='N', help='probability of using diverse inputs.')
parser.add_argument('--num', '--data_num', default=5000, type=int, metavar='N', help='the num of test images')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--kernel_type', type=str, default='gussi',  help='the type of kernel')
parser.add_argument('--c', type=float, default=0.0,  help='the param of poly kernel')
parser.add_argument('--GAA', type=int, default=0, help='use GAA instead of PAA')
parser.add_argument('--kernel_for_furthe', type=str, default='l_gussi', help='kernel for search furthest feature')
parser.add_argument('--byRank', type=int, default=1, help='choose target class by rank, if not, randomly choose')
parser.add_argument('--method', type=int, default=1, help='1 for PAA, 2 for GAA, 3 for AA')
parser.add_argument('--targetcls', type=int, default=2, help='select the target class indix, 2,10,100,500,1000')

class PAA_g(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 1):
        super(PAA_g, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
        batch_size = source.shape[0]
        channel = source.shape[1]
        h = source.shape[2]
        w = source.shape[3]
        source = source.view(batch_size, channel, h*w)
        target = target.view(batch_size, channel, h*w)
        source = source.permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        n_samples = int(source.size()[1])+int(target.size()[1])
        total = torch.cat([source, target], dim=1)
        total0 = total.unsqueeze(1).expand(batch_size, int(total.size(1)), int(total.size(1)), int(total.size(2)))
        total1 = total.unsqueeze(2).expand(batch_size, int(total.size(1)), int(total.size(1)), int(total.size(2)))
        L2_distance = ((total0-total1)**2).sum((3))
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance, axis=(1, 2), keepdim=True) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        n,c,h,w = source.shape
        batch_size = h*w
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:, :batch_size, :batch_size]
        YY = kernels[:, batch_size:, batch_size:]
        XY = kernels[:, :batch_size, batch_size:]
        YX = kernels[:, batch_size:, :batch_size]
        loss = torch.sum((XX + YY - XY -YX), dim=(1,2))/(c*c*w*w*h*h)
        return loss

def split_even_odd(x):
    """
    split a list into two different lists by the even and odd entries
    :param x: the list
    :return: two lists with even and odd entries of x respectively
    """
    n, M, c = x.size()
    # split even, odd
    n0 = M - M % 2
    return x[:, range(0, n0, 2), :], x[:, range(1, n0, 2), :], n0

def gaussian_kernel(diff_, gamma):
    """
    compute a Gaussian kernel for vector x and y
    :param x: data list
    :param y: data list
    :param gamma: parameter for the Gaussian kernel
    :return: the Gaussian kernel
    """
    # e^(-a * |x - y|^2)
    return torch.exp(-gamma * diff_)

def h(x_odd, y_odd, x_even, y_even, n0):
    """
    helper function for the MMD O(n) computation
    :param x_i: odd entries of x
    :param y_i: odd entries of y
    :param x_j: even entries of x
    :param y_j: even entries of y
    :param n0: the parameter for the Gaussian kernel
    :return: the value for the Gaussian kernel
    """
    # use variance as gamma
    diffx = torch.sum((x_even - x_odd)**2, axis=(2))
    diffy = torch.sum((y_even - y_odd)**2, axis=(2))
    diffxy = torch.sum((x_even - y_odd)**2, axis=(2))
    diffyx = torch.sum((y_even - x_odd)**2, axis=(2))
    gamma = n0 * 2 /(torch.sum(diffx, axis=1, keepdim=True) + torch.sum(diffy, axis=1, keepdim=True) + torch.sum(diffxy, axis=1, keepdim=True) + torch.sum(diffyx, axis=1, keepdim=True))
    # compute kernel values
    s1 = gaussian_kernel(diffx, gamma)
    s2 = gaussian_kernel(diffy, gamma)
    s3 = gaussian_kernel(diffxy, gamma)
    s4 = gaussian_kernel(diffyx, gamma)

    # return result of h
    s = s1 + s2 - s3 - s4
    return s

def PAA_g_l(x, y):
    """
    compute the linear time O(n) approximation of the PAA_g
    :param x: source_feature
    :param y: target_feature
    :param alpha:
    :return:
    """
    # split tensors x and y channel-wise based on its index
    n, c, h_, w = x.size()
    x = x.view(n, c, h_*w)
    y = y.view(n, c, h_*w)
    # permute shape to [n, h*w, c]
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    x_even, x_odd, n0 = split_even_odd(x)
    y_even, y_odd, n0 = split_even_odd(y)
    return torch.abs(torch.sum(h(x_odd, y_odd, x_even, y_even, n0), axis=1))/(c*c*h_*h_*w*w)

def PAA_p(source, target, c):
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)
    # permute shape to [n, h*w, c]
    source = source.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    diffx = torch.sum((source.transpose(1,2).bmm(source))**2, axis=(1,2)) + 2*c*torch.sum((source.transpose(1,2).bmm(source)), axis=(1,2))
    diffy = torch.sum((target.transpose(1,2).bmm(target))**2, axis=(1,2)) + 2*c*torch.sum(target.transpose(1,2).bmm(target), axis=(1,2)) 
    diffxy = torch.sum((source.bmm(target.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(source.bmm(target.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - 2*diffxy
    return diff/(4.0*h_*h_*w*w*ch*ch)

def PAA_p_l(source, target, c):
    d = 2
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)

    idx = torch.randperm(ch)
    idy = torch.randperm(ch)
    source = source[:, idx, ...]
    target = target[:, idy, ...]
    source = source.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    x_even, x_odd, n0 = split_even_odd(source)
    y_even, y_odd, n0 = split_even_odd(target)
    diffx = torch.sum((x_even.transpose(1,2).bmm(x_odd))**2, axis=(1,2)) + 2*c*torch.sum((x_even.transpose(1,2).bmm(x_odd)), axis=(1,2))
    diffy = torch.sum((y_even.transpose(1,2).bmm(y_odd))**2, axis=(1,2)) + 2*c*torch.sum(y_even.transpose(1,2).bmm(y_odd), axis=(1,2)) 
    diffxy = torch.sum((x_even.bmm(y_odd.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(x_even.bmm(y_odd.transpose(1,2)), axis=(1,2))
    diffyx = torch.sum((x_odd.bmm(y_even.transpose(1,2)))**2, axis=(1,2)) + 2*c*torch.sum(x_odd.bmm(y_even.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - diffxy - diffyx

    return diff/(4.0*h_*h_*w*w*ch*ch)

def PAA_line_l(source, target):
    n, ch, h_, w = source.size()
    source = source.view(n, ch, h_*w)
    target = target.view(n, ch, h_*w)
    idx = torch.randperm(ch)
    idy = torch.randperm(ch)
    source = source[:, idx, ...]
    target = target[:, idy, ...]
    source = source.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    x_even, x_odd, n0 = split_even_odd(source)
    y_even, y_odd, n0 = split_even_odd(target)
    diffx = torch.sum((x_even.transpose(1,2).bmm(x_odd)), axis=(1,2))
    diffy = torch.sum((y_even.transpose(1,2).bmm(y_odd)), axis=(1,2))
    diffxy = torch.sum((x_even.bmm(y_odd.transpose(1,2))), axis=(1,2))
    diffyx = torch.sum((x_odd.bmm(y_even.transpose(1,2))), axis=(1,2))
    diff = diffx + diffy - diffxy - diffyx

    return diff/(h_*h_*w*w*ch*ch)

def PAA_line(source, target):
    batch, ch, h_, w = source.size()
    source = source.view(batch, ch, h_*w)
    target = target.view(batch, ch, h_*w)
    # permute shape to [n, h*w, c]
    x = source.permute(0, 2, 1)
    y = target.permute(0, 2, 1)
    diffx = torch.sum(x.bmm(x.transpose(1,2)), axis=(1,2))
    diffy = torch.sum(y.bmm(y.transpose(1,2)), axis=(1,2))
    diffxy = torch.sum(x.bmm(y.transpose(1,2)), axis=(1,2))
    diff = diffx + diffy - 2 * diffxy
    return diff/(h_*h_*w*w*ch*ch)

def GAA(source, target):
    batch, ch, h_, w = source.size()
    source = source.view(batch, ch, h_*w)
    target = target.view(batch, ch, h_*w)
    # permute shape to [n, h*w, c]
    x = source.permute(0, 2, 1)
    y = target.permute(0, 2, 1)
    n = 2*h_*w
    p = ch
    ux = torch.sum(x, axis=1)*2.0/n
    uy = torch.sum(y, axis=1)*2.0/n
    diffu = torch.sum((ux-uy)**2, axis=1)
    vx = torch.sqrt(torch.sum((x-torch.unsqueeze(ux, 1))**2, axis=1)*2.0/n)
    vy = torch.sqrt(torch.sum((y-torch.unsqueeze(uy, 1))**2, axis=1)*2.0/n)
    diffv = torch.sum((vx-vy)**2, axis=1)
    diff = ((diffu+diffv)/(p*p))*h_*w
    return diff

class GAA_Loss(nn.Module):
    """
    GAA Loss
    """
    def __init__(self, GAA):
        super(GAA_Loss, self).__init__()
        self.loss = 0
        self.GAA = GAA
    
    def forward(self, source_feature, target_feature):
        if self.GAA==1:
            self.loss = GAA(source_feature, target_feature)
        return self.loss

class PAA_Loss(nn.Module):
    """
    PAA Loss
    """
    def __init__(self, kernel, c):
        super(PAA_Loss, self).__init__()
        self.loss = 0
        self.kernel = kernel
        self.c = c

    def forward(self, source_feature, target_feature):
        if self.kernel == 'gussi':
            PAA_g = PAA_g()
            self.loss = PAA_g(source_feature, target_feature)
        elif self.kernel == 'linear':
            self.loss = PAA_line(source_feature, target_feature)
        elif self.kernel == 'poly':
            self.loss = PAA_p(source_feature, target_feature, self.c)
        elif self.kernel == 'l_poly':
            self.loss = PAA_p_l(source_feature, target_feature, self.c)
        elif self.kernel == 'l_gussi':
            self.loss = PAA_g_l(source_feature, target_feature)
        elif self.kernel == 'l_linear':
            self.loss = PAA_line_l(source_feature, target_feature)
        return self.loss

def PAA_furthest(s, t, kernel, c):
    """PAA, find the furthest feature for each input feature respectively."""
    batch_size = args.batchsize
    loss = PAA_Loss(kernel, c)
    index = []
    distance = loss(s, t)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def GAA_furthest(s, t, GAA):
    """GAA, find the furthest feature for each input feature respectively."""
    batch_size = args.batchsize
    loss = GAA_Loss(GAA)
    index = []
    distance = loss(s, t)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def rn_select(y, num, batch_size):
    target = []
    y = y.numpy().tolist()
    for i in range(batch_size):
        target.append(choice([j for j in range(0, num) if j != y[i]]))
    return np.array(target)

def furthest(s, t):
    """AA, find the furthest feature for each input feature respectively."""
    batch_size = args.batchsize
    index = []
    distance = l2_norm(t - s)
    for i in range(batch_size):
        index.append(torch.argmax(distance[i * 20: (i + 1) * 20], dim=0) + i * 20)
    if len(t.shape) == 2:
        t = t[:, None, None, :]
    return t[index, :, :, :]

def attack_fp(x, t_f, model, kernel, GAA, c, method):
    batch_size = x.shape[0]
    alpha = args.eps / args.iteration
    momentum = torch.zeros([batch_size, 3, args.size, args.size]).cuda()
    if GAA == 1:
        loss = GAA_Loss(GAA)
    else:
        loss = PAA_Loss(kernel, c)
    for i in range(args.iteration):
        ori_out = model(x)
        s_f = mid_outputs
        if method==1:
            # PAA
            loss_ = loss(s_f, t_f).sum()
        elif method==2:
            # GAA
            loss_ = loss(s_f, t_f).sum()
        elif method==3:
            # AA
            loss_ = l2_norm(t_f - s_f).sum()
        loss_.backward()
        noise = x.grad.data
        l1_noise = torch.sum(torch.abs(noise), dim=(1, 2, 3))
        l1_noise = l1_noise[:, None, None, None]
        noise = noise / l1_noise
        momentum = momentum * args.decay + noise
        x = x - alpha * torch.sign(momentum)
        assert not torch.any(torch.isnan(x))
        x = torch.clamp(x, 0, 1).detach()
        x.requires_grad = True
    return x

def attack_mi(x, t_y, model):
    alpha = args.eps / args.iteration
    momentum = torch.zeros([args.batch_size, 3, args.size, args.size]).cuda()
    for i in range(args.iteration):
        pred_logit = model(x)
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        l1_noise = torch.sum(torch.abs(noise), dim=(1, 2, 3))
        l1_noise = l1_noise[:, None, None, None]
        noise = noise / l1_noise
        momentum = momentum * args.decay + noise
        x = x - alpha * torch.sign(momentum)
        x = torch.clamp(x, 0, 1).detach()
        x.requires_grad = True
    return x

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    # stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel

def attack_Ti(x, t_y, model, images_min, images_max):
    alpha = args.eps / args.iteration
    T_kern = torch.from_numpy(gkern(15, 3)).cuda()
    for i in range(args.iteration):
        pred_logit = model(x)
        ce_loss = F.cross_entropy(pred_logit.cuda(), t_y.cuda(), reduction='sum').cuda()
        ce_loss.backward()
        noise = x.grad.data
        noise = F.conv2d(noise, T_kern, padding = (7, 7), groups=3)
        x = x - alpha * torch.sign(noise)
        x = clip_by_tensor(x, images_min, images_max) 
        x = torch.autograd.Variable(x, requires_grad = True)
    return x.detach()

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def squared_l2_norm(x):
    flattened = x.reshape([x.shape[0], -1]).contiguous()
    flattened = flattened ** 2
    return torch.sum(flattened, dim=1)

def l2_norm(x):
    return squared_l2_norm(x) ** 0.5

@torch.no_grad()
def test(x_adv, y, target_y, ResNet_50, Vgg_19, Inc_v3, DenseNet_121, list_50, list_y, num, utr, tsuc, ttr):
    x_adv = x_adv.cuda()
    y = y.cuda()
    pred_adv_vgg = torch.argmax(Vgg_19(x_adv), dim=1)
    pred_adv_incv3 = torch.argmax(Inc_v3(x_adv), dim=1)
    pred_adv_121 = torch.argmax(DenseNet_121(x_adv), dim=1)
    pred_adv_50 = torch.argmax(ResNet_50(x_adv), dim=1)

    # White Box Model
    num[0] += torch.sum(pred_adv_50 != y)
    tsuc[0] += torch.sum(pred_adv_50 == target_y)
    idx_50 = pred_adv_50 != y
    idx_50_t = pred_adv_50 == target_y
    # Save White Box Model Tsuc List
    for img in x_adv[idx_50_t]:
        list_50.append(img.detach().cpu().numpy())
    for t_y in target_y[idx_50_t]:
        list_y.append(t_y.detach().cpu().numpy())
    
    # Black Box Model
    num[1] += torch.sum(pred_adv_vgg  != y)
    tsuc[1] += torch.sum(pred_adv_vgg  == target_y)
    idx_vgg = pred_adv_vgg != y
    idx_vgg_t = pred_adv_vgg == target_y
    num[2] += torch.sum(pred_adv_incv3 != y)
    tsuc[2] += torch.sum(pred_adv_incv3 == target_y)
    idx_incv3 = pred_adv_incv3 != y
    idx_incv3_t = pred_adv_incv3 == target_y
    num[3] += torch.sum(pred_adv_121 != y)
    tsuc[3] += torch.sum(pred_adv_121 == target_y)
    idx_121 = pred_adv_121 != y
    idx_121_t = pred_adv_121 == target_y
    utr[0] += torch.sum(idx_50 & idx_vgg)
    utr[1] += torch.sum(idx_50 & idx_incv3)
    utr[2] += torch.sum(idx_50 & idx_121)
    ttr[0] += torch.sum(idx_50_t & idx_vgg_t)
    ttr[1] += torch.sum(idx_50_t & idx_incv3_t)
    ttr[2] += torch.sum(idx_50_t & idx_121_t)

    return list_50, list_y, num, utr, tsuc, ttr 

def save_img(save_path, img):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = (img * 255).permute(0, 2, 3, 1).detach().cpu()
    # print(img.shape[0])
    for i in range(img.shape[0]):
        img_name = os.path.join(save_path, str(i) + '.png')
        Image.fromarray(np.array(img[i].squeeze(0)).astype('uint8')).save(img_name)

def getTarClas(x, y, model, rank=1):
    source_out = model(x)
    if rank==1:
        softMax = torch.softmax(source_out, -1)  #[1,1000]
        _, indices = torch.topk(softMax, k=args.targetcls, dim=-1, sorted=True)
        target_y = indices[:, -1]
    else:
        target_y = rn_select(y, 1000, x.shape[0])
        target_y = torch.from_numpy(target_y).cuda()
    return target_y

def getFurthestFeature(x, model, target_y, library, method, kernel, GAA, c):
    batchsize = x.shape[0]
    target_feature = []
    source_feature_tiled = []
    source_out = model(x)
    source_feature = mid_outputs
    for j in target_y:
        target_out = model(library[j].cuda())  
        target_feature.append(mid_outputs)
    target_feature = torch.cat(target_feature, axis=0)
    for j in range(batchsize):
        source_feature_tiled.append(torch.unsqueeze(source_feature[j], 0).repeat(20, 1, 1, 1))
    source_feature_tiled = torch.cat(source_feature_tiled, dim=0).cuda()
    if method == 1: 
        # PAA
        furthest_feature = PAA_furthest(source_feature_tiled, target_feature, kernel, c)
    elif method == 2:
        # GAA
        furthest_feature = GAA_furthest(source_feature_tiled, target_feature, GAA)
    elif method == 3:
        # AA
        furthest_feature = furthest(source_feature_tiled, target_feature)
    return furthest_feature

def main():
    global args
    # print('loading feature library...')
    library = np.load('data/ImageNet_image_library.npy')
    library = torch.from_numpy(library)
    # print('loading feature library done...')
    # Data loading
    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )
    val_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(args.data, transforms.Compose([
                    transforms.ToTensor(),
                ])),
    batch_size=args.batchsize, shuffle=False,
    num_workers=args.workers, pin_memory=True)
    densenet_121 = torch.nn.Sequential(Normalize(args.mean, args.std), models.densenet121(pretrained=True).eval().cuda())
    resnet_50 = torch.nn.Sequential(Normalize(args.mean, args.std), models.resnet50(pretrained=True).eval().cuda())
    vgg_19 = torch.nn.Sequential(Normalize(args.mean, args.std), models.vgg19_bn(pretrained=True).eval().cuda())
    inc_v3 = torch.nn.Sequential(Normalize(args.mean, args.std), models.inception_v3(pretrained=True).eval().cuda())
    global mid_outputs
    mid_outputs = None
    incv3_layer_list = []
    def get_mid_output(m, i, o):
        global mid_outputs 
        mid_outputs = o
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    incv3_layer_list = ['1.Conv2d_3b_1x1', '1.Conv2d_4a_3x3', '1.maxpool2', '1.Mixed_5b', '1.Mixed_5c', '1.Mixed_5d', '1.Mixed_6a', '1.Mixed_6b', '1.Mixed_6c', '1.Mixed_6d', '1.Mixed_6e', '1.AuxLogits', '1.Mixed_7a', '1.Mixed_7b', '1.Mixed_7c', '1.avgpool', '1.fc']

    for layer_name in incv3_layer_list:
        handlers = []
        for (name, module) in inc_v3.named_modules():
            if name == layer_name:
                print(name)
                handlers.append(module.register_forward_hook(get_mid_output))
        list_incv3 = []
        list_y = []
        num = [0]*4
        tsuc = [0]*4
        utr = [0]*3
        ttr = [0]*3
        for i, (x, y) in (enumerate(val_loader)):
            if i != args.num // args.batchsize:
                x = Variable(x.cuda(), requires_grad=True)
                with torch.no_grad():
                    target_y = getTarClas(x, y, inc_v3, args.byRank)
                    furthest_feature = getFurthestFeature(x, inc_v3, target_y, library, args.method, args.kernel_for_furthe, args.GAA, args.c)
                x_adv = attack_fp(x, furthest_feature, inc_v3, args.kernel_type, args.GAA, args.c, args.method).detach_()
                # save adv img
                # save_path = os.path.join(root, str(i))
                # save_img(save_path, x_adv)
                list_incv3, list_target_y, num, utr, tsuc, ttr = test(x_adv, y, target_y, inc_v3, vgg_19, densenet_121, resnet_50, list_incv3, list_y, num, utr, tsuc, ttr)
            else:
                break
        D_adv = float(args.num)
        print('Error for inc_v3: %10.4f' % float(100 * (float(num[0]) / D_adv)))
        print('tSuc for inc_v3: %10.4f' % float(100 * (float(tsuc[0]) / D_adv)))
        print('Error for vgg19: %10.4f' % float(100 * (float(num[1]) / D_adv)))
        print('uTR for vgg19: %10.4f' % float(100 * (float(utr[0]) / float(num[0]))))
        print('tSuc for vgg19: %10.4f' % float(100 * (float(tsuc[1]) / D_adv)))
        print('tTR for vgg19: %10.4f' % float(100 * (float(ttr[0]) / float(tsuc[0]))))
        print('Error for dense121: %10.4f' % float(100 * (float(num[2]) / D_adv)))
        print('uTR for dense121: %10.4f' % float(100 * (float(utr[1]) / float(num[0]))))
        print('tSuc for dense121: %10.4f' % float(100 * (float(tsuc[2]) / D_adv)))
        print('tTR for dense121: %10.4f' % float(100 * (float(ttr[1]) / float(tsuc[0]))))
        print('Error for res50: %10.4f' % float(100 * (float(num[3]) / D_adv)))
        print('uTR for res50: %10.4f' % float(100 * (float(utr[2]) / float(num[0]))))
        print('tSuc for res50: %10.4f' % float(100 * (float(tsuc[3]) / D_adv)))
        print('tTR for res50: %10.4f' % float(100 * (float(ttr[2]) / float(tsuc[0]))))
        del list_incv3, list_y, num, utr, tsuc, ttr, x_adv
        for h in handlers:
            h.remove()

if __name__ == "__main__":
    main()