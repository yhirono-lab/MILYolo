# -*- coding: utf-8 -*-
import csv
import random
import os
import yaml
import time
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ast import parse
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict as edict

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch import nn, optim
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from utils import colorstr

def train_model_yolo(opt, label_num, rank, save_dir):
    from model_yolo import feature_extractor, class_predictor, yolo_extractor, MILYOLO
    # 各ブロック宣言
    feature_extractor = feature_extractor(opt.model)
    class_predictor = class_predictor(label_num, dropout=opt.dropout, detect_obj=opt.detect_obj, yolo_stage=opt.multistage)
    
    yolo_extractor_list = []
    for s in range(opt.multistage):
        yolo = yolo_extractor()
        if opt.multistage == 1:
            yolo_weight = f'./YoloWeights/depth{int(opt.depth)}-ver{int(opt.yolo_ver)}.pt'
        else:
            yolo_weight = f'./YoloWeights/depth{int(opt.depth)+s}-best.pt'
        ckpt_data = torch.load(yolo_weight)  # load checkpoint
        csd = ckpt_data['model'].state_dict()
        yolo.load_state_dict(csd, strict=False)  # load
        yolo_extractor_list.append(yolo)
        print(colorstr('load yolo weights')+f' : {yolo_weight}') if rank==0 else None
        
    # model構築
    model = MILYOLO(feature_extractor, class_predictor, yolo_extractor_list)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if 'yolo_extractor' in k:
            # print(f'freezing {k}')
            v.requires_grad = False
    print(colorstr('yolo_extractors are freezed')) if rank==0 else None
    # print(colorstr('model:\n'), model) if rank==0 else None

    # 途中で学習が止まってしまったとき用
    if opt.restart:
        model_params_dir = f'{save_dir}/model_params'
        if os.path.exists(model_params_dir) and os.listdir(model_params_dir):
            model_params_list = sorted(os.listdir(model_params_dir))
            model_params_file = f'{model_params_dir}/{model_params_list[-1]}'
            model.load_state_dict(torch.load(model_params_file))
            restart_epoch = len(model_params_list)
            print(colorstr('restart') + f': load {model_params_list[-1]}')
        else:
            restart_epoch = 0
    
    # Optimizerの設定
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
            
    optimizer = SGD(g0, lr=opt.lr, momentum=opt.momentum, nesterov=True)
    optimizer.add_param_group({'params': g1, 'weight_decay': opt.weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    print(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias") if rank==0 else None
    del g0, g1, g2

    return model, optimizer

def test_model_yolo(opt, label_num, best_param):
    # model読み込み
    from model_yolo import feature_extractor, class_predictor, yolo_extractor, MILYOLO
    # 各ブロック宣言
    feature_extractor = feature_extractor(opt.model)
    class_predictor = class_predictor(label_num, dropout=opt.dropout, detect_obj=opt.detect_obj, yolo_stage=opt.multistage)
    yolo_extractor_list = []
    for s in range(opt.multistage):
        yolo = yolo_extractor()
        yolo_extractor_list.append(yolo)
    # YOLOMIL構築
    model = MILYOLO(feature_extractor, class_predictor, yolo_extractor_list)
    model.load_state_dict(best_param)
    
    return model

# MILYOLO に通常のAMILをオプションで変更可能にしようとしたけど，結果の出力の違いでエラーが出そうだったので断念
# モデルの構築までは動きます
# def train_model_amil(opt, label_num, rank, save_dir):
#     # model読み込み
#     from model_mil import feature_extractor, class_predictor, MIL
#     # 各ブロック宣言
#     feature_extractor = feature_extractor(opt.model)
#     class_predictor = class_predictor(label_num)
#     # model構築
#     model = MIL(feature_extractor, class_predictor)

#     # 途中で学習が止まってしまったとき用
#     if opt.restart:
#         model_params_dir = f'{save_dir}/model_params'
#         if os.path.exists(model_params_dir):
#             model_params_list = sorted(os.listdir(model_params_dir))
#             model_params_file = f'{model_params_dir}/{model_params_list[-1]}'
#             model.load_state_dict(torch.load(model_params_file))
#             restart_epoch = len(model_params_list)
#             print(colorstr('restart') + f': load {model_params_list[-1]}')
#         else:
#             restart_epoch = 0
    
#     # Optimizerの設定
#     optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=0.0001)

#     return model, optimizer

# def test_model_amil(opt, label_num, best_param):
#     # model読み込み
#     from model_mil import feature_extractor, class_predictor, MIL
#     # 各ブロック宣言
#     feature_extractor = feature_extractor(opt.model)
#     class_predictor = class_predictor(label_num)
#     # DAMIL構築
#     model = MIL(feature_extractor, class_predictor)
#     model.load_state_dict(best_param)

#     return model