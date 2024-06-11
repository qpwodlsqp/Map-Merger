import sys
import cv2
import torch
import numpy as np
import random

from lie_util import *
from models import MapTransformer, MapCNN, MergeNet
from smm import SMM
from MyMap import MyMap

import os
import time
import itertools
import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Map Inference',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Experiment Misc
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1557080, metavar='S',
                    help='random seed')
parser.add_argument('--map_viz', action='store_true', default=False,
                    help='generate merged maps')
parser.add_argument('--plot_viz', action='store_true', default=False,
                    help='generate merged maps')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.plot_viz:
    plt.rc('font', size=15)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 8))

def to_numpy(item):

    cam, tar, tar_tilde, pose_delta = item
    H = cam.shape[-2]; W = cam.shape[-1]
    cam = 255 * cam; tar = 255 * tar; tar_tilde = 255 * tar_tilde
    cam = cam.numpy().astype(np.uint8).reshape(H, W)
    tar = tar.numpy().astype(np.uint8).reshape(H, W)
    tar_tilde = tar_tilde.numpy().astype(np.uint8).reshape(H, W)
    return (cam, tar, tar_tilde, pose_delta.numpy().flatten())

def _discretize(M):

    empty = np.where(M >= 0.805, 1., 0.)
    unknown = np.where(np.logical_and(M < 0.805, M > 0.3), 0.5, 0)
    return empty + unknown


def merge_gt(cam, tar):

    cam = cam.astype(np.float64)/255; tar = tar.astype(np.float64)/255
    cam = _discretize(cam); tar = _discretize(tar)
    gray = (cam == 0.5) & (tar == 0.5)
    black = (cam == 0.0) | (tar == 0.0)
    merged = np.ones(cam.shape)
    gray_mask = 0.5 * np.ones(cam.shape)
    black_mask = np.zeros(cam.shape)
    merged = np.where(gray, gray_mask, merged)
    merged = np.where(black, black_mask, merged)
    return (merged * 255).astype(np.uint8)


def map_consistency(M_0, M_1):

    agree = np.logical_and(M_0>=200, M_1>=200).sum() + np.logical_and(M_0<=75, M_1<=75).sum()
    if agree == 0:
        return 0.
    else:
        disagree = np.logical_and(M_0<=75, M_1>=200).sum() + np.logical_and(M_0>=200, M_1<=75).sum()
        return agree / (agree+disagree)

def overlap(cam, tar):

    H, W = cam.shape
    cam = cam.astype(np.float64)/255; tar = tar.astype(np.float64)/255
    cam = _discretize(cam); tar = _discretize(tar)
    overlap = np.logical_and(np.where(cam != 0.5, 1, 0), np.where(tar != 0.5, 1, 0)).sum()
    map_info = np.logical_or(np.where(cam != 0.5, 1, 0), np.where(tar != 0.5, 1, 0)).sum()
    return overlap / map_info

def CI(err_list, c=1.96):
    return np.std(err_list) * c / np.sqrt(len(err_list))

def baselines():

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.map_viz:
        os.makedirs(os.path.join('viz_result', 'smm'), exist_ok=True)
    mymap = MyMap(test=True, batch=1, workers=0)
    testloader = mymap.dataloader
    w_diff = []
    h_diff = []
    yaw_diff = []
    over = []
    for idx, item in enumerate(tqdm(testloader, desc='Hough SMM', position=0, leave=False, file=sys.stdout)):
        cam, tar, tar_tilde, pose_delta = to_numpy(item)
        H, W = cam.shape
        # Hough Spectral Map Merging (S. Carpin, 2008)
        merged_smm, delta = SMM(cam, tar_tilde, 360, 360, 384)
        w_diff.append(np.abs(delta[0] - pose_delta[0]) / 128 * 384 * 0.05)
        h_diff.append(np.abs(delta[1] - pose_delta[1]) / 128 * 384 * 0.05)
        yaw_diff.append(np.abs(delta[2] - pose_delta[2]) * np.pi / 180)
        over.append(overlap(cam, tar))
        if args.map_viz:
            merged_gt  = merge_gt(cam, tar)
            merged_til = merge_gt(cam, tar_tilde)
            cv2.imwrite(f'viz_result/smm/{idx}.png', np.hstack((merged_gt, merged_til, merged_smm)))

    print(f'\nHough SMM Avg Error\nDelta_X: {np.mean(h_diff):.4f} pm {CI(h_diff):.4f}\tDelta_Y: {np.mean(w_diff):.4f} pm {CI(w_diff):.4f}\tDelta_Theta: {np.mean(yaw_diff):.4f} pm {CI(yaw_diff):.4f}')
    if args.plot_viz:
        zipped = zip(w_diff, h_diff, yaw_diff, over)
        zipped = sorted(zipped, key=lambda x : x[-1])
        w, h, yaw, over = zip(*zipped)
        axes[0].scatter(over, h, marker='.', facecolors='none', edgecolors='red', alpha=0.8)
        axes[1].scatter(over, w, marker='.', facecolors='none', edgecolors='red', alpha=0.8)
        axes[2].scatter(over, yaw, marker='.', facecolors='none', edgecolors='red', alpha=0.8, label='Hough')
    return
 
def infer(arch, rec, lie):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # model init
    use_lie = 'lie-o' if lie else 'lie-x'
    use_rec = 'rec-o' if rec else 'rec-x'
    if args.map_viz:
        os.makedirs(os.path.join('viz_result', f'{arch}_{use_rec}_{use_lie}'), exist_ok=True)
    ckpt = torch.load(f'./weight/mergenet_{arch}_{use_rec}_{use_lie}.pth')
    args_load = ckpt['args']
    if arch == 'cnn':
        model = MapCNN()
    else:
        model = MapTransformer(args_load.size, args_load.patch_size, args_load.dim,
                               args_load.depth, args_load.heads, args_load.mlp_dim,
                               args_load.pool, args_load.channels, args_load.dim_head,
                               args_load.dropout, args_load.emb_dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    # Inference Dataset
    mymap = MyMap(test=True, batch=1, workers=0)
    testloader = mymap.dataloader
    # Wrapper for Torch Model to handle Numpy array
    mergenet = MergeNet(model, device)

    map_loss = []
    pose_diff = []
    time_list = []
    w_diff = []
    h_diff = []
    yaw_diff = []
    over = []

    with torch.no_grad():
        pbar = tqdm(testloader, desc=f'{arch}_{use_rec}_{use_lie}', position=1, leave=False, file=sys.stdout)
        for idx, item in enumerate(pbar):
            # These datas are PyTorch tensors, because they should be output from PyTorch util Dataset
            # Turn them back to Numpy
            cam, tar, tar_tilde, pose_delta = to_numpy(item)
            H, W = cam.shape
            # Mergenet Model Inference
            merged, theta = mergenet.merge(cam, tar_tilde, args_load.use_lie_regress)
            w_diff.append(np.abs(theta[0] - pose_delta[0]) / 128 * 384 * 0.05)
            h_diff.append(np.abs(theta[1] - pose_delta[1]) / 128 * 384 * 0.05)
            yaw_diff.append(np.abs(theta[2] - pose_delta[2]) * np.pi / 180)
            over.append(overlap(cam, tar))
            if args.map_viz:
                merged_gt  = merge_gt(cam, tar)
                merged_til = merge_gt(cam, tar_tilde)
                nick = f'viz_result/{arch}_{use_rec}_{use_lie}'
                cv2.imwrite(f'{nick}/{idx}.png', np.hstack((merged_gt, merged_til, merged)))

    print(f'\nModel {arch}_{use_rec}_{use_lie} Avg Error\nDelta_X: {np.mean(h_diff):.4f} pm {CI(h_diff):.4f}\tDelta_Y: {np.mean(w_diff):.4f} pm {CI(w_diff):.4f}\tDelta_Theta: {np.mean(yaw_diff):.4f} pm {CI(yaw_diff):.4f}')
    del model
    if args.plot_viz and (arch == 'vit' and rec and lie):
        zipped = zip(w_diff, h_diff, yaw_diff, over)
        zipped = sorted(zipped, key=lambda x : x[-1])
        w, h, yaw, over = zip(*zipped)
        axes[0].scatter(over, h, marker='.', facecolors='none', edgecolors='blue', alpha=0.5)
        axes[1].scatter(over, w, marker='.', facecolors='none', edgecolors='blue', alpha=0.5)
        axes[2].scatter(over, yaw, marker='.', facecolors='none', edgecolors='blue', alpha=0.5, label='Ours')
    return


def test_all():

    arch_list = ['cnn', 'vit']
    use_lie = [False, True]
    use_rec = [False, True]
    model_iter = itertools.product(arch_list, use_rec, use_lie)
    baselines()
    for arch, rec, lie in model_iter:
        infer(arch, rec, lie)
    if args.plot_viz:
        axes[0].set_title('Absolute Translation Error on X axis (m)')
        axes[1].set_title('Absolute Translation Error on Y axis (m)')
        axes[2].set_title('Absolute Rotation Error (radian)')
        axes[2].set_xlabel('Overlap between Input Local Maps')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss_plot.png')
    return


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = False # For reproducibility
    test_all()



