# -*- coding: utf-8 -*-
from ast import parse
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import random
import os
import glob
import re
import time
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def increment_path(path, exist_ok=False, sep='-', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = ''
        if os.path.isfile(path):
            suffix = path.suffix
            path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s = f"CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)"
            print(s)

    return torch.device('cuda:0' if cuda else 'cpu')

def make_dirname(args):
    if args.classify_mode == 'subtype':
        dir_name = f'subtype_classify'
        dir_name = f'{dir_name}_lr-{args.lr}_mag-{args.mag}'

        if args.fc:
            dir_name = f'fc_{dir_name}'
        if args.dropout:
            dir_name = f'do_{dir_name}'
        dir_name = f'{args.model}_{dir_name}'
        
        if args.data:
            if args.reduce:
                dir_name = f'reduce_{dir_name}'
            dir_name = f'{args.data}_{dir_name}'
    else:
        dir_name = f'{args.classify_mode}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{args.constant}'
        if args.loss_mode == 'focal' or args.loss_mode == 'focal-weight':
            dir_name = f'{dir_name}-{args.gamma}'
        dir_name = f'{dir_name}_lr-{args.lr}_mag-{args.mag}'
        
        if args.fc:
            dir_name = f'fc_{dir_name}'
        if args.dropout:
            dir_name = f'do_{dir_name}'
        dir_name = f'{args.model}_{dir_name}'
        
        if args.augmentation:
            dir_name = f'aug_{dir_name}'
        if args.data:
            if args.reduce:
                dir_name = f'reduce_{dir_name}'
            dir_name = f'{args.data}_{dir_name}'

        dir_name = f'{dir_name}/depth-{args.depth}_leaf-{args.leaf}'
        if args.yolo_ver:
            dir_name = f'{dir_name}_weight-ver{args.yolo_ver}'
        if args.detect_obj:
            dir_name = f'{dir_name}_detect-{args.detect_obj}'
        if args.multistage>1:
            dir_name = f'{dir_name}_stage-{args.multistage}'

    return dir_name

def make_filename(args):
    if args.classify_mode == 'subtype':
        filename = f'subtype_classify'
        filename = f'{filename}_lr-{args.lr}_mag-{args.mag}'
        if args.fc:
            filename = f'fc_{filename}'
        if args.dropout:
            filename = f'do_{filename}'
        filename = f'{args.model}_{filename}'
        
        if args.data:
            if args.reduce:
                filename = f'reduce_{filename}'
            filename = f'{args.data}_{filename}'
    else:
        filename = args.classify_mode
        filename = f'{filename}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            filename = f'{filename}-{args.constant}'
        if args.loss_mode == 'focal' or args.loss_mode == 'focal-weight':
            filename = f'{filename}-{args.gamma}'
        filename = f'{filename}_lr-{args.lr}_mag-{args.mag}'
        
        if args.fc:
            filename = f'fc_{filename}'
        if args.dropout:
            filename = f'do_{filename}'
        filename = f'{args.model}_{filename}'
        
        if args.augmentation:
            filename = f'aug_{filename}'
        if args.data:
            if args.reduce:
                filename = f'reduce_{filename}'
            filename = f'{args.data}_{filename}'
        filename = f'{filename}_depth-{args.depth}_leaf-{args.leaf}'
        if args.yolo_ver:
            filename = f'{filename}_weight-ver{args.yolo_ver}'
        if args.detect_obj:
            filename = f'{filename}_detect-{args.detect_obj}'
        if args.multistage>1:
            filename = f'{filename}_stage-{args.multistage}'
            
    return filename

def get_max_depth():
    weight_list = glob.glob('./YoloWeights/depth*best.pt')
    return len(weight_list)

def get_slideID_name():
    file_name = '../KurumeTree/add_data/Data_SimpleName_svs.csv'
    csv_data = np.loadtxt(file_name, delimiter=',', dtype='str')
    name_list = {}

    for i in range(1,csv_data.shape[0]):
        if csv_data[i,1] == 'N/A':
            csv_data[i,1] = 'NA'
        if csv_data[i,1] == 'CLL/SLL':
            csv_data[i,1] = 'CLL-SLL'
        if 'N/A' in csv_data[i,1]:
            csv_data[i,1] = csv_data[i,1].replace('N/A','NA')
        if 'CLL/SLL' in csv_data[i,1]:
            csv_data[i,1] = csv_data[i,1].replace('CLL/SLL','CLL-SLL')
        name_list[csv_data[i,0]] = csv_data[i,1]

    return name_list

def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return

def send_email(body:str):
    MAIL_ADDRESS = 'nitech28114106@gmail.com'
    PASSWORD = 'yrxqmyxhkglnpfsq'
    TO_ADDRESS1 = 'yyph.fam@gmail.com'
    TO_ADDRESS2 = 'ckv14106@nitech.jp'

    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.ehlo()
    smtpobj.starttls()
    smtpobj.ehlo()
    smtpobj.login(MAIL_ADDRESS, PASSWORD)

    msg = make_msg(MAIL_ADDRESS, TO_ADDRESS1, body)
    smtpobj.sendmail(MAIL_ADDRESS, TO_ADDRESS1, msg.as_string())

    msg = make_msg(MAIL_ADDRESS, TO_ADDRESS2, body)
    smtpobj.sendmail(MAIL_ADDRESS, TO_ADDRESS2, msg.as_string())
    
    smtpobj.close()

def make_msg(from_addr, to_addr, body_msg):
    subject = 'inform finished program'
    msg = MIMEText(body_msg)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Date'] = formatdate()

    return msg

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def scale_coords(coords, img1_shape=[224,224], img0_shape=[224,224], ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
            
            
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y
