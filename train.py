# -*- coding: utf-8 -*-
import csv
import random
import os
# import yaml
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
from utils import colorstr, increment_path, select_device, one_cycle
import dataloader_svs

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    # winではncclは使えないので、gloo
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label[0])
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def model_train(rank, model, train_loader, loss_fn, optimizer, scaler, epoch):
    model.train() #訓練モードに変更
    train_class_loss = 0.0
    correct_num = 0
    total_num = 0

    nb = len(train_loader)
    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=nb)
    for i, (input_tensor, slideID, class_label) in pbar:
        optimizer.zero_grad()
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        batch_loss = 0

        for bag_num in range(input_tensor.shape[0]):
            input = input_tensor[bag_num]
            label = class_label[bag_num]
            class_prob, class_hat, A, _ = model(input)
            
            # 各loss計算
            class_loss = loss_fn(class_prob, label)
            batch_loss += class_loss
            train_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label)
            total_num += 1
            del input, label

        # Backward
        batch_loss /= input_tensor.shape[0]
        scaler.scale(batch_loss).backward()
        
        # Optimize
        scaler.step(optimizer)  # optimizer.step
        scaler.update()
        
        # Log
        accuracy = correct_num / total_num
        loss = train_class_loss / total_num
        pbar.set_description(('%10s' + '%10.4g' * 2) % (f'epoch:{epoch}', accuracy, loss))

    return train_class_loss, correct_num

def model_valid(rank, model, valid_loader, loss_fn, epoch):
    model.eval() #訓練モードに変更
    valid_class_loss = 0.0
    correct_num = 0
    total_num = 0

    nb = len(valid_loader)
    pbar = enumerate(valid_loader)
    pbar = tqdm(pbar, total=nb)
    for i, (input_tensor, slideID, class_label) in pbar:
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A, _ = model(input_tensor[bag_num])
            
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            valid_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])
            total_num += 1
            
        # Log
        accuracy = correct_num / total_num
        loss = valid_class_loss / total_num
        pbar.set_description(('%10s' + '%10.4g' * 2) % (f'epoch:{epoch}', accuracy, loss))

    return valid_class_loss, correct_num

def train(rank, world_size, opt):
    setup(rank, world_size)

    save_dir = Path(opt.save_dir)
    epochs = opt.epoch

    w = save_dir / 'model_params' # weights directory
    w.mkdir(parents=True, exist_ok=True) # make directory
    last, best, ckpt = w / 'last.pt', w / 'best.pt', w / 'ckpt.pt'

    # Save run settings or load run settings
    if not opt.restart or not os.path.exists(save_dir / 'opt.yaml'):
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)
            print(colorstr('save opt.yaml') + ':' + str(opt)) if rank==0 else None
    elif opt.restart:
        with open(save_dir / 'opt.yaml') as f:
            opt = edict(yaml.safe_load(f))
            print(colorstr('load opt.yaml') + ':' + str(opt)) if rank==0 else None

    data_dict = None
    cuda = True

    # # 訓練用と検証用に症例を分割
    colorstr('data load:')
    import dataset_kurume as ds
    if opt.classify_mode == 'normal_tree' or opt.classify_mode == 'kurume_tree':
        train_dataset, valid_dataset, label_num = ds.load_leaf(opt, rank)
    elif opt.classify_mode == 'subtype':
        train_dataset, valid_dataset, label_num = ds.load_svs(opt, rank)

    label_count = np.zeros(label_num)
    for i in range(label_num):
        label_count[i] = len([d for d in train_dataset if d[1] == i])
    if rank == 0:
        print(f'train split:{opt.train} train slide count:{len(train_dataset)}')
        print(f'valid split:{opt.valid}   valid slide count:{len(valid_dataset)}')
        print(f'train label count:{label_count}')
   
    log = f'{save_dir}/log_{opt.mag}_{opt.lr}_train-{opt.train}.csv'
    if rank == 0 and (not os.path.exists(log) or not opt.restart):
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from make_model import train_model_yolo
    model, optimizer = train_model_yolo(opt, label_num, rank, save_dir)
    model = model.to(rank)
    print(colorstr('model created')) if rank==0 else None
    
    lf = one_cycle(1, opt.lr*20, epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    # Resume
    start_epoch, best_fitness = 0, 100.0
    if opt.restart:
        ckpt_dir = Path(f'{save_dir}/model_params')
        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
            ckpt_file = f'{ckpt_dir}/ckpt.pt'
            
            ckpt_data = torch.load(ckpt_file, map_location=device)  # load checkpoint
            # Optimizer
            if ckpt_data.get('optimizer'):
                optimizer.load_state_dict(ckpt_data['optimizer'])
                best_fitness = ckpt_data['best_fitness']

            # Epochs
            start_epoch = ckpt_data['epoch'] + 1
            if epochs < start_epoch:
                epochs += ckpt_data['epoch']  # finetune additional epochs

            del ckpt_data

    
    if len(opt.device.split(',')) > 1:
        # # SyncBatchNorm
        process_group = torch.distributed.new_group([i for i in range(world_size)])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
        # # DDP mode
        model = DDP(model, device_ids=[rank])
    

    last_opt_step = -1
    results = (0, 0, 0, 0, 0) # train_acc, train_loss, val_acc, val_loss
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    
    # 損失関数
    from model_yolo import CEInvarse, LDAMLoss, FocalLoss
    loss_fn = None
    if opt.loss_mode == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(rank)
    if opt.loss_mode == 'ICE':
        loss_fn = CEInvarse(rank, label_count).to(rank)
    if opt.loss_mode == 'LDAM':
        loss_fn = LDAMLoss(rank, label_count, Constant=float(opt.constant)).to(rank)
    if opt.loss_mode == 'focal':
        loss_fn = FocalLoss(rank, label_count, gamma=float(opt.gamma)).to(rank)
    if opt.loss_mode == 'focal-weight':
        loss_fn = FocalLoss(rank, label_count, gamma=float(opt.gamma), weight_flag=True).to(rank)
        
    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad() #勾配初期化
        
        if rank == 0:
            print(f'epoch:{epoch}')
        
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        data_train = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=train_dataset,
            class_count=label_num,
            mag=opt.mag,
            bag_num=50,
            bag_size=100
        )
        
        #Datasetをmulti GPU対応させる
        #下のDataLoaderで設定したbatch_sizeで各GPUに分配
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=os.cpu_count()//world_size,
            sampler=train_sampler
        )
        train_loader.sampler.set_epoch(epoch)
        
        class_loss, correct_num = model_train(rank, model, train_loader, loss_fn, optimizer, scaler, epoch)
        
        train_loss += class_loss
        train_acc += correct_num
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        
        data_valid = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            class_count=label_num,
            mag=opt.mag,
            bag_num=50,
            bag_size=100
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        valid_sampler = torch.utils.data.distributed.DistributedSampler(data_valid, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        valid_loader = torch.utils.data.DataLoader(
            data_valid,
            batch_size=opt.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=os.cpu_count()//world_size,
            sampler=valid_sampler
        )

        # 学習
        class_loss, correct_num = model_valid(rank, model, valid_loader, loss_fn, epoch)
        
        valid_loss += class_loss
        valid_acc += correct_num
        
        # GPU１つあたりの精度を計算
        train_loss /= float(len(train_loader.dataset))/float(world_size)
        train_acc /= float(len(train_loader.dataset))/float(world_size)
        valid_loss /= float(len(valid_loader.dataset))/float(world_size)
        valid_acc /= float(len(valid_loader.dataset))/float(world_size)

        if valid_loss < best_fitness:
            best_fitness = valid_loss

        # epochごとにlossなどを保存
        if rank == 0:
            f = open(log, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])
            f.close()
            
        # epochごとにmodelのparams保存
        if rank == 0:
            model_params_dir = f'{save_dir}/model_params/{opt.mag}_train-{opt.train}_epoch-{epoch}.pt'
            torch.save(model.module.state_dict(), model_params_dir)
            if best_fitness == valid_loss:
                torch.save(model.module.state_dict(), best)
            
                ckpt_data = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'optimizer': optimizer.state_dict(),
                            }
                torch.save(ckpt_data, ckpt)
                del ckpt_data
            
    torch.cuda.empty_cache()


def parse_opt(known=False):    
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--yolo_ver', default=None, help='choose weight version')
    parser.add_argument('--data', default='2nd', choices=['1st', '2nd', '3rd'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--multistage', default=1, type=int)
    parser.add_argument('--detect_obj', default=None, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='kurume_tree', choices=['normal_tree', 'kurume_tree', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('-C', '--constant', default=0)
    parser.add_argument('-g', '--gamma', default=0)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('-r', '--restart', default='')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    
    opt.valid = split_dict[opt.train]
    
    """
    1st < 2nd < 3rd の順でデータが増えるが，
    両方の葉のデータを増やすとデータの偏りが大きくなるので
    もとから多いほうの葉は少ないデータ(1st)で行うためのフラグ
    """
    if opt.data == '2nd' or opt.data == '3rd':
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

# cross validation の訓練に対する検証データの設定
split_dict = {
    '123':'4',
    '234':'5',
    '345':'1',
    '451':'2',
    '512':'3',
}

def main(opt):
    # Checks
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    # 同じ実験設定のパスが存在するなら，番号付けして新規作成する
    if opt.restart:
        opt.save_dir = str(Path('./runs/' + utils.make_dirname(opt)) / opt.restart)
    else:
        opt.save_dir = str(increment_path(Path('./runs/' + utils.make_dirname(opt) + '/train' + opt.train)))
    print(colorstr(opt.save_dir) + ': maked save directory')
    
    # DDP mode
    device = select_device(opt.device)
    num_gpu = len(opt.device.split(','))

    # Train
    mp.spawn(train, args=(num_gpu, opt), nprocs=num_gpu, join=True)
    # train(opt, device)
    if WORLD_SIZE > 1 and RANK == 0:
        _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)