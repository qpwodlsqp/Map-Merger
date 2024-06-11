import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from models import MapTransformer, MapCNN
from MyMap import MyMap
from lie_util import *
from tqdm import tqdm

import argparse
import datetime
import os
import shutil


parser = argparse.ArgumentParser(description='Map Transformer',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model Hyperparam
parser.add_argument('--model_type', type=str, default='vit',
                    help='Frame Error Estimator Model Type in [\'vit\', \'cnn\']')
parser.add_argument('--size', type=int, default=128, metavar='N',
                    help='image size. assuming square')
parser.add_argument('--patch_size', type=int, default=16, metavar='N',
                    help='patch size for image')
parser.add_argument('--dim', type=int, default=256, metavar='N',
                    help='output dim for encoder')
parser.add_argument('--depth', type=int, default=6, metavar='N',
                    help='num of transformer blocks')
parser.add_argument('--heads', type=int, default=8, metavar='N',
                    help='num of heads in attention layer')
parser.add_argument('--mlp_dim', type=int, default=512, metavar='N',
                    help='output dim for MLP layer in ViT')
parser.add_argument('--pool', type=str, default='cls', metavar='N',
                    help='pooling method')
parser.add_argument('--channels', type=int, default=2, metavar='N',
                    help='input image channel number for ViT')
parser.add_argument('--dim_head', type=int, default=32, metavar='N',
                    help='head num for wat?')
parser.add_argument('--dropout', type=float, default=0., metavar='N',
                    help='dropout rate')
parser.add_argument('--emb_dropout', type=float, default=0., metavar='N',
                    help='dropout rate for wat?')
parser.add_argument('--use_lie_regress', action='store_true', default=False,
                    help='Lie Regression Mode')
parser.add_argument('--use_rec_loss', action='store_true', default=False,
                    help='enables recurrent loss')
parser.add_argument('--omega', type=float, default=0.1, metavar='N',
                    help='hyperparam weight for recurrent loss')

# Training Hyperparam
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                    help='number of epochs to train')

# Experiment Misc
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=15571601, metavar='S',
                    help='random seed')
parser.add_argument('--log_every', type=int, default=1900, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_every', type=int, default=25, metavar='N',
                    help='how many epochs to wait to save models')
parser.add_argument('--exp', type=str, default='MyMap',
                    help='distinguishable experiment name')

args = parser.parse_args()
assert args.model_type in ['vit', 'cnn']
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


def train():

    # model init
    if args.model_type == 'vit':
        merger_net = MapTransformer(args.size,     args.patch_size, args.dim,         args.depth,
                                    args.heads,    args.mlp_dim,    args.pool,        args.channels,
                                    args.dim_head, args.dropout,    args.emb_dropout)
    elif args.model_type == 'cnn':
        merger_net = MapCNN()
    else:
        raise NotImplementedError
    # Data Loader
    mapsplit = MyMap(test=False,
                     batch=args.batch_size,
                     workers=4)
    trainloader = mapsplit.dataloader
    # Optimization
    criterion = nn.SmoothL1Loss(beta=9.0)
    optimizer = optim.Adam(merger_net.parameters(), lr=args.lr)
    merger_net.to(device)
    merger_net.train()

    i_step = 0 # iteration counter
    for epoch in tqdm(range(args.epochs), desc='total epoch', position=0, file=sys.stdout):
        running_loss_list = []
        rec_loss_list = []
        pbar = tqdm(trainloader, desc='current epoch', position=1, leave=False, file=sys.stdout)
        for item in pbar:
            cam, tar, tar_tilde, pose_delta = item
            try:
                optimizer.zero_grad()
                delta_hat, rec_loss = merger_net(cam.to(device), tar_tilde.to(device),
                                                 args.use_rec_loss, args.use_lie_regress)
                if args.use_rec_loss and args.use_lie_regress:
                    alg = target_to_algebra(pose_delta)
                    loss = lie_loss(delta_hat, alg.to(device), torch.from_numpy(mapsplit.dataset.COVAR))
                    loss += args.omega * rec_loss
                    rec_loss_list.append(rec_loss.item())
                    fake_loss = criterion(algebra_to_target(delta_hat.to('cpu')), pose_delta)
                    running_loss_list.append(fake_loss.item()) # only for scalable comparison
                    pbar.set_postfix(loss=fake_loss.item())
                elif args.use_rec_loss and not args.use_lie_regress:
                    loss = criterion(delta_hat, pose_delta.to(device))
                    pbar.set_postfix(loss=loss.item())
                    rec_loss_list.append(rec_loss.item())
                    running_loss_list.append(loss.item())
                    loss += args.omega * rec_loss
                elif not args.use_rec_loss and args.use_lie_regress:
                    alg = target_to_algebra(pose_delta)
                    loss = lie_loss(delta_hat, alg.to(device), torch.from_numpy(mapsplit.dataset.COVAR))
                    fake_loss = criterion(algebra_to_target(delta_hat.to('cpu')), pose_delta)
                    pbar.set_postfix(loss=fake_loss.item())
                    running_loss_list.append(fake_loss.item()) # only for scalable comparison
                else:
                    loss = criterion(delta_hat, pose_delta.to(device))
                    pbar.set_postfix(loss=loss.item())
                    running_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()

                if (i_step+1) % args.log_every == 0:
                    print(f'\n[Epoch {epoch+1}]')
                    if args.use_rec_loss:
                        print(f'Average Loss: {np.mean(running_loss_list):.6f}')
                        print(f'Average Rec Reg Loss: {np.mean(rec_loss_list):.6f}')
                    else:
                        print(f'Average Loss: {np.mean(running_loss_list):.6f}')
                    running_loss_list = []
                    rec_loss_list = []
                i_step += 1
            except InterruptedError:
                 continue

        if (epoch + 1) % args.save_every == 0: 
            torch.save({
                        'args': args,
                        'model_state_dict': merger_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                       }, os.path.join('result', nick, f"merger_net_{epoch+1}.pth"))

    torch.save({
                'args': args,
                'model_state_dict': merger_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, os.path.join('result', nick, f"merger_net_last.pth"))
    # Copy the parameters of last epoch to weight directory
    os.makedirs(os.path.join(os.getcwd(), 'weight'), exist_ok=True)
    rec_nick = 'rec-o' if args.use_rec_loss else 'rec-x'
    lie_nick   = 'lie-o' if args.use_lie_regress else 'lie-x'
    weight_file_name = f'mergernet_{args.model_type}_{rec_nick}_{lie_nick}.pth'
    shutil.copy(os.path.join('result', nick, f"merger_net_last.pth"), os.path.join('weight', weight_file_name))
    return


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    now = datetime.datetime.now()
    now_str = now.strftime('%m-%d-%H-%M-%S')
    recon_nick = 'recon-O' if args.use_rec_loss else 'recon-X'
    lie_nick   = 'lie-O' if args.use_lie_regress else 'lie-X'
    nick = f"{args.model_type}_{recon_nick}_{lie_nick}_{now_str}"
    os.makedirs(os.path.join('result', nick), exist_ok=False)
    train()


