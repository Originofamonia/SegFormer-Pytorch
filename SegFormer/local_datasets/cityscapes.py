import cv2
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
# from PIL import Image, ImageOps, ImageFilter
import argparse
from typing import Tuple, List, Any
import local_datasets.transform as transform
from utils.augmentations import trav_train_augmentation, trav_val_augmentation, make_trav_dataset

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


class EpisodicIndoorTrav(Dataset):
    def __init__(self, 
                #  root: str, split: str = 'train', scenes=[], transform=None
                 args, transform, class_list, split
                 ) -> None:
        super().__init__()
        try:
            assert split in ['train', 'val', 'test']
        except AssertionError:
            print('Invalid split for mode! Please use split="train" or "val"')
        self.args = args
        self.root = args.data_root
        self.split = split
        self.transform = transform
        self.scenes = args.scenes
        self.class_list = class_list
        self.data_list, self.sub_class_file_list = make_trav_dataset(args.data_root, args.scenes, split, self.class_list)
    
    def __getitem__(self, index) -> Any:
        # ====== Read query image + Chose class ======
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = np.load(label_path)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()

        new_label_class = []
        for c in label_class:
            if c in self.class_list:  # current list of classes to try
                new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        # ====== From classes in query image, chose one randomly ======
        class_chosen = np.random.choice(label_class)

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # ====== Build support ======
        # First, randomly choose indexes of support images
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        if self.args.random_shot:
            shot = np.random.randint(1, self.shot)
        else:
            shot = self.args.shot

        for k in range(shot):
            support_idx = np.random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path)
                  or support_idx in support_idx_list):
                support_idx = np.random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = [self.class_list.index(class_chosen) + 1]  # index of the chosen class in new_classes

        # Second, read support images and masks
        for k in range(shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = np.load(support_label_path)
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (
                    RuntimeError("Support Image & label shape mismatch: "
                                 + support_image_path + " " + support_label_path + "\n")
                )
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == shot and len(support_image_list) == shot

        # Original support images and labels
        support_labels = support_label_list.copy()

        # Forward images through transforms
        if self.transform is not None:  # 255 is padding, Resize is not cropping
            qry_img, target = self.transform(image, label)
            for k in range(shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # Reshape properly
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, spprt_imgs, spprt_labels, subcls_list, \
               [support_image_path_list, support_labels], [image_path, label]

    def __len__(self):
        return len(self.data_list)



def trav_train_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a episodic loader.
    """
    assert args.train_split in [0, 1, 2, 3]
    aug_dic = {
        'randscale': transform.RandScale([args.scale_min, args.scale_max]),
        'randrotate': transform.RandRotate(
            [args.rot_min, args.rot_max],
            padding=[0 for x in args.mean],
            ignore_label=255
        ),
        'hor_flip': transform.RandomHorizontalFlip(),
        'vert_flip': transform.RandomVerticalFlip(),
        'crop': transform.Crop(
            args.image_size, crop_type='rand',
            padding=[0 for x in args.mean], ignore_label=255
        ),
        'resize': transform.Resize(args.image_size)
    }

    train_transform = [aug_dic[name] for name in args.augmentations]
    train_transform += [transform.ToTensor(), transform.Normalize(mean=args.mean, std=args.std)]
    train_transform = transform.Compose(train_transform)

    class_list = [0,1]

    # ====== Build loader ======
    train_set = EpisodicIndoorTrav(args, train_transform, class_list, 'train')

    # world_size = torch.distributed.get_world_size()
    if args.DDP:
        sampler = DistributedSampler(train_set)
    else:
        sampler = RandomSampler(train_set)
    # train_sampler = DistributedSampler(train_data) if args.distributed else None
    # batch_size = int(args.batch_size / world_size) if args.distributed else args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    return train_loader, sampler


def trav_val_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']
    val_transform = transform.Compose([
            transform.Resize(args.image_size),
            transform.ToTensor(),
            transform.Normalize(mean=args.mean, std=args.std)
    ])
    val_sampler = None

    # ====== Filter out classes seen during training ======
    class_list = [0,1]

    # ====== Build loader ======
    val_data = EpisodicIndoorTrav(
        args, val_transform, class_list, 'val'
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    return val_loader, val_transform


if __name__ == '__main__':
    train_transform = trav_train_augmentation([480, 640])
    train_set = IndoorTrav('/home/qiyuan/2023spring/segmentation_indoor_images', 'train', ['elb', 'erb', 'uc', 'woh'], )
