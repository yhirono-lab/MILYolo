# -*- coding: utf-8 -*-
import re
import csv
import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import dataloader_svs_hirono
from yolo import non_max_suppression
from utils import scale_coords, colorstr, select_device

def parse_opt(known=False):    
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('-n', '--number', default='', help='choose training number')
    parser.add_argument('--depth', default='2', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    parser.add_argument('--data', default='add', choices=['', 'add'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--num_gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='new_tree', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('-C', '--constant', default=0)
    parser.add_argument('-g', '--gamma', default=0)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('-r', '--restart', default='')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    parser.add_argument('--save_bbox', action='store_true')
    parser.add_argument('--exist_ok', action='store_true')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    opt.valid = split_dict[opt.train]
    
    if opt.data == 'add':
        opt.reduce = True

    if opt.classify_mode != 'subtype':
        if opt.depth == None:
            print(f'mode:{opt.classify_mode} needs depth param')
            exit()
    
    if opt.loss_mode == 'LDAM' and not opt.constant:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()
    
    if (opt.loss_mode == 'focal' or opt.loss_mode == 'focal-weight') and not opt.gamma:
        print(f'when loss_mode is focal, input gamma param')
        exit()
        
    return opt

split_dict = {
    '123':'5',
    '234':'1',
    '345':'2',
    '451':'3',
    '512':'4',
}

def paths_sorted(paths):
    return sorted(paths, key = lambda x: int(x.name.split('-')[-1]) if 'train' not in x.name.split('-')[-1] else 1)

def main(opt):
    # Checks
    print(colorstr('test: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    opt.save_dir = Path('./runs/' + utils.make_dirname(opt)) / f'train{opt.train}'
    if opt.number:
        opt.save_dir = Path(str(opt.save_dir) + f'-{opt.number}')
    else:
        dir_list = paths_sorted(list(opt.save_dir.parent.glob(fr'train{opt.train}*')))
        opt.save_dir = dir_list[-1]
    print(colorstr(opt.save_dir) + ': test this directory')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)