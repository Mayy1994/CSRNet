#!/usr/bin/python
# -*- encoding: utf-8 -*-
from logger import setup_logger
from cityscapes_edge import CityScapes


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from csrnet_feat import resnet18
from csrnet_seg import SemsegModel
from matplotlib import pyplot as plt


class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [ 1.0],
            n_classes = 19,
            lb_ignore = 255,
            cropsize = 1024,
            flip = False,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)
            if len(out)>1:
                out = out[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
        return prob


    def crop_eval(self, im):
        prob = self.eval_chip(im)
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def get_confusion_matrix(self, label, pred, size, num_class, ignore):
        """
        Calcute the confusion matrix by given label and pred
        """
        output = pred.cpu().numpy().transpose(0, 2, 3, 1)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        seg_gt = np.asarray(
        label.numpy()[:, :size[-2], :size[-1]], dtype=np.int)
        ignore_index = seg_gt != ignore
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]

        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                    i_pred] = label_count[cur_index]
        return confusion_matrix
        


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        confusion_matrix = np.zeros((n_classes, n_classes))
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            size = label.size()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))

            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)

        return mIOU

def evaluate(respth='/home/mybeast/xjj/mbnet/0.7401_ori/github/model/cityscapes/pretrained/best.pth', dspth='/home/mybeast/xjj/cityscapes', checkpoint=None):

    ## logger
    logger = logging.getLogger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 19

    num_features = 128
    resnet = resnet18(pretrained=True, efficient=False, num_features=num_features)
    net = SemsegModel(resnet, n_classes)

    if checkpoint is None:
        save_pth = respth
    else:
        save_pth = checkpoint
    logger.info('load model: {}'.format(save_pth))


    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 1
    n_workers = 1
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=[1.0],flip=False)
    ## eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))
    return mIOU



if __name__ == "__main__":
    setup_logger('./model')
    evaluate()
