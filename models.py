import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from vit_pytorch import ViT
from lie_util import *

class MapTransformer(nn.Module):

    def __init__(self, size=128, patch_size=16, dim=256, depth=4, heads=8,
                 mlp_dim=512, pool='cls', channels=2, dim_head=32, dropout=0.1, emb_dropout=0.):
        super().__init__()
        self.vit = ViT(image_size=size,
                       patch_size=patch_size,
                       dim=dim,
                       depth=depth,
                       heads=heads,
                       mlp_dim=mlp_dim,
                       pool=pool,
                       channels=channels,
                       dim_head=dim_head,
                       dropout=dropout,
                       emb_dropout=emb_dropout)
        self.fc = nn.Sequential(nn.Linear(dim, dim // 4),
                                nn.ReLU(),
                                nn.Linear(dim // 4, 3))


    def recurrent_loss(self, cam, tar, pose_delta, lie):

        N, _, H, W = cam.shape
        # Convert torchvision coordinate updates to STN coordinates update
        # translation: opposite (away from top-left => toward top-left), pixel value to [-1, 1] 
        # orientation: CW to CCW, deg to radian
        # Thankfully: center rotation and post-rotation translation still applied ...
        if lie:
            # convert algebra coordinates to group coordinates
            pose_delta = algebra_to_target(pose_delta)
        N_tw = pose_delta[:, 0:1]; N_th = pose_delta[:, 1:2]; N_ttheta = pose_delta[:, 2:]
        N_tw = -N_tw / W * 2; N_th = -N_th / H * 2; N_ttheta = -N_ttheta * np.pi / 180.
        # To (N, 2, 3) affine marices
        N_cos = torch.cos(N_ttheta); N_sin = torch.sin(N_ttheta)
        affine_matrices = torch.cat([N_cos, -N_sin, N_tw, N_sin, N_cos, N_th], dim=-1).reshape(N, 2, 3)
        # Transform cam to tar_tilde frame by STN
        grid = F.affine_grid(affine_matrices, cam.shape, align_corners=False)
        cam = F.grid_sample(cam, grid, mode='nearest', padding_mode='border', align_corners=False)
        # Recurrent propagation
        # This should be zero vector
        recur_delta, _ = self.forward(cam, tar)
        return torch.linalg.norm(recur_delta, 2, -1).mean()


    def forward(self, cam, tar, recurr=False, lie=False):

        img = torch.cat((cam, tar), dim=1) # <(B x 1 x H x W), (B x 1 x H x W)> => (B x 2 x H x W)
        theta_latent = self.vit(img)
        theta = self.fc(theta_latent)
        if not self.training:
            return theta
        if recurr:
            loss = self.recurrent_loss(cam, tar, theta, lie)
            return theta, loss
        else:
            return theta, None


class MapCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(2, 16, 4, 4), # (B x 2 x 128 x 128) => (B x 16 x 32 x 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B x 16 x 32 x 32) => (B x 16 x 16 x 16)
            nn.Conv2d(16, 64, 3), # (B x 16 x 16 x 16) => (B x 64 x 14 x 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B x 64 x 14 x 14) => (B x 64 x 7 x 7)
            nn.Conv2d(64, 256, 3), # (B x 64 x 7 x 7) => (B x 256 x 5 x 5)
            nn.ReLU(),
            nn.Conv2d(256, 256, 3), # (B x 256 x 5 x 5) => (B x 256 x 3 x 3)
            nn.ReLU(),
            nn.MaxPool2d(3, 3) # (B x 256 x 3 x 3) => (B x 256 x 1 x 1)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )


    def recurrent_loss(self, cam, tar, pose_delta, lie):

        N, _, H, W = cam.shape
        # Convert torchvision coordinate updates to STN coordinates update
        # translation: opposite (away from top-left => toward top-left), pixel value to [-1, 1] 
        # orientation: CW to CCW, deg to radian
        # Thankfully: center rotation and post-rotation translation still applied ...
        if lie:
            # convert algebra coordinates to group coordinates
            pose_delta = algebra_to_target(pose_delta)
        N_tw = pose_delta[:, 0:1]; N_th = pose_delta[:, 1:2]; N_ttheta = pose_delta[:, 2:]
        N_tw = -N_tw / W * 2; N_th = -N_th / H * 2; N_ttheta = -N_ttheta * np.pi / 180
        # To (N, 2, 3) affine marices
        N_cos = torch.cos(N_ttheta); N_sin = torch.sin(N_ttheta)
        affine_matrices = torch.cat([N_cos, -N_sin, N_tw, N_sin, N_cos, N_th], dim=-1).reshape(N, 2, 3)
        # Transform cam to tar_tilde frame by STN
        grid = F.affine_grid(affine_matrices, cam.shape, align_corners=False)
        cam = F.grid_sample(cam, grid, mode='nearest', padding_mode='border', align_corners=False)
        # Recurrent propagation
        # This should be zero vector
        recur_delta, _ = self.forward(cam, tar)
        return torch.linalg.norm(recur_delta, 2, -1).mean()


    def forward(self, cam, tar, recurr=False, lie=False):
        img = torch.cat((cam, tar), dim=1) # <(B x 1 x H x W), (B x 1 x H x W)> => (B x 2 x H x W)
        feature = self.cnn_layers(img) # (B x 2 x 28 x 28) => (B x 128 x 1 x1)
        feature = feature.flatten(1, -1)
        theta = self.fc_layers(feature)
        if not self.training:
            return theta
        if recurr:
            loss = self.recurrent_loss(cam, tar, theta, lie)
            return theta, loss
        else:
            return theta, None


class MergeNet:

    def __init__(self, network, device):
        super().__init__()
        self.network = network
        self.device = device

    def tensor_map(self, cv_map):

        H, W = cv_map.shape
        map_tensor = (torch.from_numpy(cv_map) / 255.).reshape(1, 1, H, W)
        one_mask = torch.ones(map_tensor.shape)
        zero_mask = torch.zeros(map_tensor.shape)
        empty = torch.where(map_tensor > 0.804, one_mask, zero_mask)
        unknown = torch.where((map_tensor <= 0.804) & (map_tensor >= 0.35), 0.5*one_mask, zero_mask)
        return TF.resize(empty+unknown, 128, interpolation=transforms.InterpolationMode.NEAREST)

    def _discretize(self, M):

        empty = np.where(M >= 0.805, 1., 0.)
        unknown = np.where(np.logical_and(M < 0.805, M > 0.3), 0.5, 0)
        return empty + unknown

    def _merge(self, cam, tar, theta):

        H, W = cam.shape
        tw = theta[0] * W / 128; th = theta[1] * H / 128; ttheta = -theta[2]
        R = cv2.getRotationMatrix2D((W//2, H//2), int(-ttheta), 1.0)
        T = np.float32([[1, 0, -tw], [0, 1, -th]])
        tar = cv2.warpAffine(tar, T, (H, W), borderValue=127, flags=0)
        tar = cv2.warpAffine(tar, R, (H, W), borderValue=127, flags=0)
        cam = self._discretize(cam/255); tar = self._discretize(tar/255)
        
        gray = (cam == 0.5) & (tar == 0.5)
        black = (cam == 0.0) | (tar == 0.0)
        merged = np.ones(cam.shape)
        gray_mask = 0.5 * np.ones(cam.shape)
        black_mask = np.zeros(cam.shape)
        merged = np.where(gray, gray_mask, merged)
        merged = np.where(black, black_mask, merged)
        return (merged * 255).astype(np.uint8)

    def merge(self, cam, tar, lie=False):

        cam_tensor = self.tensor_map(cam).to(self.device)
        tar_tensor = self.tensor_map(tar).to(self.device)
        theta = self.network(cam_tensor, tar_tensor)
        if lie:
            theta = algebra_to_target(theta)
        theta = theta.to('cpu').numpy().flatten()
        return self._merge(cam, tar, theta), theta






