import os
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyMapDataset(Dataset):

    def __init__(self, test):

        self.test = test
        if self.test:
            self.root_path = os.path.join(os.getcwd(), 'MyMap', 'test')
        else:
            self.root_path = os.path.join(os.getcwd(), 'MyMap', 'train')
        name_list = os.listdir(self.root_path)
        name_list.sort()
        num = len(name_list) // 2
        self.idx_list = []
        for i in range(num):
            cam_name = name_list[2*i]
            tar_name = name_list[2*i+1]
            assert cam_name.split('_')[0] == tar_name.split('_')[0]
            self.idx_list.append(cam_name.split('_')[0])

        # Kernel Density # DEPRECATED
        loc_x_list = []
        loc_y_list = []
        yaw_list = []
        self.COVAR = np.eye(3)

    def __len__(self):
        return len(self.idx_list)

    def map_to_tensor(self, map_np):

        map_tensor = (torch.from_numpy(map_np)/255.).unsqueeze(0)
        one_mask = torch.ones(map_tensor.shape)
        zero_mask = torch.zeros(map_tensor.shape)
        empty = torch.where(map_tensor > 0.804, one_mask, zero_mask)
        unknown = torch.where((map_tensor <= 0.804) & (map_tensor >= 0.35), 0.5*one_mask, zero_mask)
        # The test code requires the original resolution
        if self.test:
            return empty + unknown
        else:
            return TF.resize(empty+unknown, 128, interpolation=transforms.InterpolationMode.NEAREST)

    def map_augmentation(self, cam, tar):

        H, W = cam.shape
        trans = [int(random.uniform(-W/20, W/20)), int(random.uniform(-H/20, H/20))]
        theta = int(random.uniform(-180, 180))
        R = cv2.getRotationMatrix2D((W//2, H//2), theta, 1.0)
        T = np.array([[1., 0., trans[0]], [0., 1., trans[1]]])
        cam = cv2.warpAffine(cam, R, cam.shape, borderValue=127, flags=0)
        cam = cv2.warpAffine(cam, T, cam.shape, borderValue=127, flags=0)
        tar = cv2.warpAffine(tar, R, tar.shape, borderValue=127, flags=0)
        tar = cv2.warpAffine(tar, T, tar.shape, borderValue=127, flags=0)
        return cam, tar

    def tar_affine(self, tar):
        # Also output different tilde errors by split, considering the image resolution
        # dividing by 0.05, which is the resolution
        if self.test:
            tw = random.uniform(-2., 2.) * 20
            th = random.uniform(-2., 2.) * 20
            t_theta = random.uniform(-np.pi/4, np.pi/4) * 180 / np.pi

            tar_tilde = TF.affine(tar, 
                                  t_theta,
                                  (tw, th),
                                  scale=1.,
                                  shear=0.,
                                  interpolation=transforms.InterpolationMode.NEAREST,
                                  fill=0.5)
            return tar_tilde, torch.tensor([tw * 128 / 384, th * 128 / 384, t_theta], dtype=torch.float32)
        else:
            tw = random.uniform(-2., 2.) * 20 * 128 / 384
            th = random.uniform(-2., 2.) * 20 * 128 / 384
            t_theta = random.uniform(-np.pi/4, np.pi/4) * 180 / np.pi

            tar_tilde = TF.affine(tar, 
                                  t_theta,
                                  (tw, th),
                                  scale=1.,
                                  shear=0.,
                                  interpolation=transforms.InterpolationMode.NEAREST,
                                  fill=0.5)
            return tar_tilde, torch.tensor([tw, th, t_theta], dtype=torch.float32)

    def __getitem__(self, idx):

        data_idx = self.idx_list[idx]
        cam_np = cv2.imread(os.path.join(self.root_path, f'{data_idx}_cam.png'), cv2.IMREAD_GRAYSCALE)
        tar_np = cv2.imread(os.path.join(self.root_path, f'{data_idx}_tar.png'), cv2.IMREAD_GRAYSCALE)
        assert cam_np.shape == tar_np.shape

        cam_np, tar_np = self.map_augmentation(cam_np, tar_np)
        cam = self.map_to_tensor(cam_np)
        tar = self.map_to_tensor(tar_np)
 
        tar_tilde, pose_delta = self.tar_affine(tar) 
        return cam, tar, tar_tilde, pose_delta


class MyMap:

    def __init__(self, test=False, batch=16, workers=0):

        self.dataset = MyMapDataset(test)
        def seed_worker(worker_id):

            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size = batch,
                                     shuffle = not test,
                                     num_workers = workers,
                                     worker_init_fn = seed_worker,
                                     generator = g)
        return


