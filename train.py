#!/usr/bin/python
# -*- encoding: utf-8 -*-
## CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py

import sys
sys.path.insert(0,'./')
from logger import setup_logger
from cityscapes_edge import CityScapes

from loss import OhemCrossEntropy2d, SemsegCrossEntropy
from loss import OhemCELoss
from loss import LossEdge
from val import evaluate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse

from csrnet_feat import resnet18
from csrnet_seg import SemsegModel

from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser

import torch.optim as optim



respth = './model/cityscapes'
if not osp.exists(respth): os.makedirs(respth)
logger = logging.getLogger()

def parse_args():
    parser = ArgumentParser(description='CSRNet')
    parser.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    # model and dataset
    parser.add_argument('--model', type=str, default="CSRNet", help="model name: (default CSRNet)")
    parser.add_argument('--dataset', type=str, default="cityscapes", help="dataset: cityscapes or camvid") ##
    parser.add_argument('--input_size', type=list, default=[1024, 1024], help="input size of model") ##
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=19,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--ignore', type=int, default=255,
                        help="the index of ignore classes in the dataset. 255 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="train", ##
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    # training hyper params
    parser.add_argument('--max_it', type=int, default=30001,
                        help="the number of iterations: 300 for train set, 350 for train+val set")
    parser.add_argument('--eval_it', type=int, default=20000,
                        help="the number of iterations: 300 for train set, 350 for train+val set")  

    parser.add_argument('--lr', type=float, default=4e-4, help="initial learning rate")
    parser.add_argument('--lr_min', type=float, default=1e-6, help="initial learning rate")
    parser.add_argument('--epochs', type=int, default=250, help="epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="the batch size is set to 16 for 2 GPUs") ##
    parser.add_argument('--wd', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--fine_tune_factor', type=float, default=4, help="weight decay")

    parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.5 to 2")
    parser.add_argument('--optim',type=str.lower,default='sgd',choices=['sgd','adam','radam','ranger'],help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='warmpoly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=0, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False, help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False, help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False,help=' FocalLoss2d for cityscapes dataset')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--distinct_wd', type=bool, default=False, help="distinct_wd")

    args = parser.parse_args()


    return args

def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0:
        print(args)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:33241',
                world_size = torch.cuda.device_count(),
                rank=args.local_rank
                )
    setup_logger(respth)

    ## dataset
    n_classes = args.classes
    n_img_per_gpu = args.batch_size
    n_workers = args.num_workers
    cropsize = args.input_size
    data_path = '/home/mybeast/xjj/cityscapes'
    train_dataset = CityScapes(data_path, cropsize=cropsize, mode=args.train_type) ##
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dl = DataLoader(train_dataset,
                    batch_size = n_img_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    ## model
    ignore_idx = args.ignore
    num_features = 128
    resnet = resnet18(pretrained=True, efficient=False, num_features=num_features)
    net = SemsegModel(resnet, n_classes)

    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank,
            find_unused_parameters=True
            )

    net.criterion = SemsegCrossEntropy(num_classes=n_classes, ignore_id=ignore_idx)
    lr = args.lr
    lr_min = args.lr_min
    fine_tune_factor = args.fine_tune_factor
    weight_decay = args.wd
    epochs = args.epochs

    optim_params = [
        {'params': net.module.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': net.module.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)

    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//8
    n_min_sma = n_img_per_gpu*cropsize[0]*cropsize[1]//(8*32*32)
    n_min_mid = n_img_per_gpu*cropsize[0]*cropsize[1]//(8*16*16)
    n_min_big = n_img_per_gpu*cropsize[0]*cropsize[1]//(8*8*8)

    score_thres = 0.7
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss_sma = OhemCELoss(thresh=score_thres, n_min=n_min_sma, ignore_lb=ignore_idx)
    Loss_mid = OhemCELoss(thresh=score_thres, n_min=n_min_mid, ignore_lb=ignore_idx)
    Loss_big = OhemCELoss(thresh=score_thres, n_min=n_min_big, ignore_lb=ignore_idx)
    ALoss_mid = nn.MSELoss()
    ALoss_sma1 = nn.MSELoss()
    ALoss_sma2 = nn.MSELoss()

    Loss_Edge = LossEdge(ignore_index=ignore_idx)


    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = args.lr

    max_iter = epochs * len(dl)
    power = args.poly_exp
    warmup_steps = args.warmup_iters
    warmup_start_lr = 1e-5
    distinct_wd = args.distinct_wd


    msg = ', '.join([
            'batch size: {batch_size}',
            'crop size: {crop_size}',
            'initial lr: {lr}',
            'num_features: {num_features}',
            'epochs: {epochs}',

        ]).format(
            batch_size = n_img_per_gpu,
            crop_size = cropsize,
            lr = lr,
            num_features = num_features,
            epochs = epochs
        )

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    best_mIoU = 0
    best_mIoU_epoch = 0
    IoU = []

    it = 0
    for epoch in range(epochs):
        net.train()

        lr_scheduler.step()
        for group in optimizer.param_groups:
            msg = 'LR: {:.4e}'.format(group['lr'])
            logger.info(msg)
        msg = 'Epoch: {} / {}'.format(epoch+1, epochs)
        logger.info(msg)
        batch_iterator = iter(enumerate(dl))

        for index, (im, lb, lb_edge) in batch_iterator:

            im = im.cuda()
            lb = lb.cuda()
            lb_edge = lb_edge.cuda()
            H, W = im.size()[2:]
            lb = torch.squeeze(lb, 1)

            optimizer.zero_grad()
            out = net.forward(im)
            
            loss = net.criterion(out, lb)
            loss.backward()
            optimizer.step()



            loss_avg.append(loss.item())
            it = it+1
            ## print training log message
            if (it+1)%msg_iter==0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                eta = int((max_iter - it) * (glob_t_intv / it))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join([
                        'it: {it}',
                        'loss: {loss:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]).format(
                        it = it+1,
                        loss = loss_avg,
                        time = t_intv,
                        eta = eta
                    )
                logger.info(msg)

                loss_avg = []
                st = ed
        if args.local_rank <= 0:    
            save_pth = osp.join(respth, 'now.pth')
            torch.save(net.module.state_dict(),save_pth)
            if epoch%20==0 or epoch >= 200:
                mean_IoU = evaluate(checkpoint=save_pth, respth=data_path)
                IoU.append(mean_IoU)
                if mean_IoU > best_mIoU:
                    save_pth_best = osp.join(respth, 'best.pth')
                    torch.save(net.module.state_dict(),save_pth_best)
                    best_mIoU = mean_IoU
                    best_mIoU_epoch = epoch+1

                msg = 'MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, Best_mIoU_epoch: {}'.format(mean_IoU, best_mIoU, best_mIoU_epoch)
                logger.info(msg)



if __name__ == "__main__":
    train()
