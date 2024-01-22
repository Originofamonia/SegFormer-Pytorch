"""
https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024
https://huggingface.co/blog/fine-tune-segformer
torchrun --standalone --nproc_per_node=gpu finetune_segformer_city_ddp.py
"""
import os
import re
import torch
import json
from huggingface_hub import hf_hub_download
import argparse
import yaml
from transformers import SegformerFeatureExtractor, SegformerConfig
from transformers import SegformerForSemanticSegmentation
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch.distributed import init_process_group, destroy_process_group
# from torchvision.datasets import Cityscapes
import evaluate

from local_datasets import Cityscapes
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import get_loss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
import utils.distributed_utils as utils


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_argparser():
    parser = argparse.ArgumentParser('Pytorch SegFormer Models training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/qiyuan/2023spring/superpixel-align/data/cityscapes',help="path to Dataset")
    parser.add_argument("--image_size", type=int, default=[512, 1024], help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',choices=['cityscapes', 'traversability'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")
    parser.add_argument("--pin_mem", type=bool, default=True, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=4,help='batch size (default: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,help='batch size for validation (default: 2)')

    # SegFormer Options
    parser.add_argument("--model", type=str, default='make_SegFormerB1', help='model name')

    # Train Options
    parser.add_argument("--amp", type=bool, default=False, help='auto mixture precision, do not use') # There may be some problems when loading weights, such as: ComplexFloat
    parser.add_argument("--epochs", type=int, default=2, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda:1', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=4,
                        help='num_workers, set it equal 0 when run programs in win platform')
    parser.add_argument("--DDP", type=bool, default=True)
    parser.add_argument("--train_print_freq", type=int, default=50)
    parser.add_argument("--val_print_freq", type=int, default=50)

    # Loss Options
    parser.add_argument("--loss_fn_name", type=str, default='CrossEntropy', choices=['OhemCrossEntropy', 'CrossEntropy'])

    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--lr", type=float, default=1e-3,
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
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    parser.add_argument("--eval_interval", type=int, default=2, help="evaluation interval")
    parser.add_argument("--save_every", type=int, default=2, help="evaluation interval")
    parser.add_argument("--load_pretrained", type=bool, default=True)
    parser.add_argument("--pretrained_path", type=str, default='nvidia/segformer-b3-finetuned-cityscapes-1024-1024')

    return parser

class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        # init train objects
        if args.DDP:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = args.device
        train_transform = get_train_augmentation(args.image_size)
        val_transform = get_val_augmentation(args.image_size)
        repo_id = "huggingface/label-files"
        id2label_filename = "cityscapes-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, id2label_filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}

        train_set = Cityscapes(args.data_root, split='train', transform=train_transform)
        valid_set = Cityscapes(args.data_root, split='val', transform=val_transform)
        if args.DDP:
            self.sampler = DistributedSampler(train_set)
        else:
            self.sampler = RandomSampler(train_set)
        self.trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                drop_last=True, pin_memory=args.pin_mem, sampler=self.sampler)

        self.valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                            drop_last=False, pin_memory=args.pin_mem)
        # self.loss_fn = get_loss(args.loss_fn_name, train_set.ignore_label, None)
        # config = SegformerConfig()
        # config.semantic_loss_ignore_index = 255
        self.model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_path,
                                                                    #   config=config,
                                                                num_labels=args.num_classes, 
                                                                id2label=id2label, 
                                                                label2id=label2id).to(self.gpu_id)
        self.metric = evaluate.load("mean_iou")
        self.optimizer = get_optimizer(self.model, args.optimizer, args.lr, args.weight_decay)
        iters_per_epoch = len(train_set) // args.batch_size
        self.scheduler = get_scheduler(args.lr_scheduler, self.optimizer, args.epochs * iters_per_epoch, args.lr_power,
                                iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)
        self.scaler = GradScaler(enabled=args.amp) if torch.cuda.is_bf16_supported() else None
        self.metric_logger = utils.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        self.confmat = utils.ConfusionMatrix(args.num_classes)

        if args.DDP:
            self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def train_epoch(self, epoch):
        self.model.train()
        if self.args.DDP:
            self.trainloader.sampler.set_epoch(epoch)
        # header = 'Epoch: [{}]'.format(epoch)
        for iter, (img, lbl) in enumerate(tqdm(self.trainloader)):
            img = img.to(self.gpu_id)
            lbl = lbl.to(self.gpu_id)
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast(enabled=args.amp):
                    outputs = self.model(img, lbl)
                    loss, logits = outputs.loss, outputs.logits
                    # loss = loss_fn(logits, lbl)
            else:
                outputs = self.model(img, lbl)
                loss, logits = outputs.loss, outputs.logits
                # loss = loss_fn(logits, lbl)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            torch.cuda.synchronize()

            loss_value = loss.item()
            lr = self.optimizer.param_groups[0]["lr"]

            self.metric_logger.update(loss=loss_value, lr=lr)

        torch.cuda.empty_cache()
        return self.metric_logger.meters["loss"].global_avg, lr

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self,):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            if epoch % self.args.eval_interval == 0:
                confmat = self.eval()
                val_info = str(confmat)
                print(val_info)

        confmat = self.eval()
        val_info = str(confmat)
        print(val_info)
    
    @torch.no_grad()
    def eval(self,):
        self.model.eval()
        header = 'Test: '
        for i, (images, labels) in enumerate(tqdm(self.valloader)):
            images = images.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            outputs = self.model(images, labels)
            loss, logits = outputs.loss, outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            self.confmat.update(labels.flatten(), predicted.flatten())

        self.confmat.reduce_from_all_processes()
        return self.confmat


def main(args):
    if args.DDP:
        ddp_setup()
    trainer = Trainer(args)
    trainer.train()
    if args.DDP:
        destroy_process_group()
    # model = model.to(args.device)
    # for epoch in range(args.epochs):
    #     mean_loss, lr = city_train_one_epoch(args, model, optimizer, loss_fn, trainloader, sampler, scheduler,
    #                                  epoch, args.device, args.train_print_freq, scaler)

    #     confmat = eval(args, model, valloader, args.device, args.val_print_freq)

    #     val_info = str(confmat)
    #     print(val_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch SegFormer Models training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    main(args)
