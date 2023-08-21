import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class SRNDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, scn_idx=0, **kwargs):
        super().__init__(root_dir, split, downsample, scn_idx)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
            lines = f.readlines()
            fx = fy = float(lines[0].split()[0]) * self.downsample
            h, w = map(lambda x: int(int(x)*self.downsample), lines[-1].split())

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb/*')))
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose/*.txt')))

        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            poses = np.array([x for i, x in enumerate(poses) if i%8!=0])
        elif split=='test':
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            poses = np.array([x for i, x in enumerate(poses) if i%8==0])

        print(f'Loading {len(img_paths)} {split} images ({self.scn_idx} th) ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):
            self.poses += [np.loadtxt(pose).reshape(4, 4)[:3]]

            img = read_image(img_path, self.img_wh)
            self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)