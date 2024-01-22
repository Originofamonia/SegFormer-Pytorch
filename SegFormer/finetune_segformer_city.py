"""
https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024
https://huggingface.co/blog/fine-tune-segformer
single gpu version
"""
import os
import re
import torch
import json
from huggingface_hub import cached_download, hf_hub_url
import argparse
import yaml
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
import evaluate
from engine import city_train_one_epoch, eval
from local_datasets import *
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import get_loss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp


def get_argparser():
    parser = argparse.ArgumentParser('Pytorch SegFormer Models training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/qiyuan/2023spring/superpixel-align/data/cityscapes',help="path to Dataset")
    parser.add_argument("--image_size", type=int, default=[512, 512], help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")
    parser.add_argument("--pin_mem", type=bool, default=True, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=4,help='batch size (default: 4)') # consume approximately 3G GPU-Memory
    parser.add_argument("--val_batch_size", type=int, default=4,help='batch size for validation (default: 2)')

    # SegFormer Options
    parser.add_argument("--model", type=str, default='make_SegFormerB1', help='model name')

    # Train Options
    parser.add_argument("--amp", type=bool, default=False, help='auto mixture precision') # There may be some problems when loading weights, such as: ComplexFloat
    parser.add_argument("--epochs", type=int, default=2, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda:1', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=3,
                        help='num_workers, set it equal 0 when run programs in win platform')
    parser.add_argument("--DDP", type=bool, default=False)
    parser.add_argument("--train_print_freq", type=int, default=50)
    parser.add_argument("--val_print_freq", type=int, default=50)

    # Loss Options
    parser.add_argument("--loss_fn_name", type=str, default='CrossEntropy', choices=['OhemCrossEntropy', 'CrossEntropy'])

    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    parser.add_argument("--lr_scheduler", type=str, default='WarmupPolyLR')
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--lr_warmup", type=int, default=10)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1)

    # save checkpoints
    parser.add_argument("--save_weights_dir", default='./save_weights', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--pretrained_path", type=str, default='/home/qiyuan/2024spring/SegFormer-Pytorch/pretrained/segformer.b1.1024x1024.city.160k.pt')

    return parser

def main(args):
    train_transform = get_train_augmentation(args.image_size, seg_fill=args.ignore_label)
    val_transform = get_val_augmentation(args.image_size)
    repo_id = "huggingface/label-files"
    filename = "cityscapes-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    train_set = CityScapes(args.data_root, 'train', train_transform)
    valid_set = CityScapes(args.data_root, 'val', val_transform)
    sampler = RandomSampler(train_set)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem)
    loss_fn = get_loss(args.loss_fn_name, train_set.ignore_label, None)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
                                                            num_labels=args.num_classes, 
                                                            id2label=id2label, 
                                                            label2id=label2id)
    metric = evaluate.load("mean_iou")
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    iters_per_epoch = len(train_set) // args.batch_size
    scheduler = get_scheduler(args.lr_scheduler, optimizer, args.epochs * iters_per_epoch, args.lr_power,
                              iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)
    scaler = GradScaler(enabled=args.amp) if torch.cuda.is_bf16_supported() else None
    model = model.to(args.device)
    for epoch in range(args.epochs):

        mean_loss, lr = city_train_one_epoch(args, model, optimizer, loss_fn, trainloader, sampler, scheduler,
                                     epoch, args.device, args.train_print_freq, scaler)

        confmat = eval(args, model, valloader, args.device, args.val_print_freq)

        val_info = str(confmat)
        print(val_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch SegFormer Models training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
