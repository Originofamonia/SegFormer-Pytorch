import json
import os
import numpy as np
import torch
from torch import Tensor
# from collections import namedtuple
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
from typing import Tuple, List, Any

from utils.augmentations import trav_train_augmentation, trav_val_augmentation

class Cityscapes(Dataset):
    """
    num_classes: 19
    """
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
               'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    PALETTE = torch.tensor(
        [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
         [250, 170, 30], [220, 220, 0], [107, 142, 35],
         [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
         [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    ID2TRAINID = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3,
                  13: 4, 14: 255, 15: 255, 16: 255,
                  17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
                  30: 255, 31: 16, 32: 17, 33: 18, -1: 255}

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        img_path = Path(root) / 'leftImg8bit' / split
        self.files = list(img_path.rglob('*.png'))

        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")


    def __len__(self) -> int:
        return len(self.files)


    def encode(self, label: np.array) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')
        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        if self.transform:
            image, label = self.transform(image, label)
        lbl = self.encode(label.squeeze().numpy()).long()
        return image, lbl


class IndoorTrav(Dataset):
    def __init__(self, root: str, split: str = 'train', scenes=[], transform=None) -> None:
        super().__init__()
        try:
            assert split in ['train', 'val', 'test']
        except AssertionError:
            print('Invalid split for mode! Please use split="train" or "val"')
        self.root = root
        self.split = split
        self.transform = transform
        self.scenes = scenes
        self.scene_dirs = [os.path.join(self.root, x) for x in self.scenes]
        if split == 'train':
            self.images_dirs = [os.path.join(x, 'positive') for x in self.scene_dirs]
        else:  # 'val'
            self.images_dirs = [os.path.join(x, 'challenging') for x in self.scene_dirs]
        
        self.images = []
        self.targets = []

        for scene in self.images_dirs:
            img_dir = os.path.join(scene, 'images')
            target_dir = os.path.join(scene, 'labels')

            for filename in os.listdir(img_dir):
                abs_img_path = os.path.join(img_dir, filename)
                target_name = filename.rstrip('.jpg') + '.png'
                abs_label_path = os.path.join(target_dir, target_name)
                if os.path.exists(abs_img_path) and os.path.exists(abs_label_path):
                    self.images.append(abs_img_path)
                    self.targets.append(abs_label_path)
    
    # @classmethod
    # def encode_target(cls, target):
    #     return cls.id_to_train_id[np.array(target)]
    
    def __getitem__(self, index) -> Any:
        try:
            image = io.read_image(self.images[index])  # [640,480]
            target = io.read_image(self.targets[index])  # [480, 640]
        except IOError:
            raise RuntimeError(f'Cannot open image: {self.images[index]} or label: {self.targets[index]}')

        if self.transform:
            image, target = self.transform(image, target)
        target = (target / 255).long()
        target = target.squeeze(0)
        return image, target, self.images[index]

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_transform = trav_train_augmentation([480, 640])
    train_set = IndoorTrav('/home/qiyuan/2023spring/segmentation_indoor_images', 'train', ['elb', 'erb', 'uc', 'woh'], )