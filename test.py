# -*- coding: utf-8 -*-
import re
import csv
import os
import sys
import argparse
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
import dataloader_svs
from yolo import non_max_suppression
from utils import scale_coords, colorstr, select_device

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    # winではncclは使えないので、gloo
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def cal_bbox(opt, yolo_out, slideID, pos_list):
    bbox_list = [] * yolo_out.shape[0]
    yolo_boxes = non_max_suppression(yolo_out)
    for i, det in enumerate(yolo_boxes):
        pos = pos_list[i].tolist()
        det[:, :4] = scale_coords(det[:, :4]).round()
        det = det.to('cpu').detach().tolist()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            pos = pos + [*xyxy, conf]
        bbox_list.append(pos)
    return bbox_list
    

def model_test(rank, opt, model, test_loader, output_file):
    model.eval() #テストモードに変更

    pred_label = []
    nb = len(test_loader)
    pbar = enumerate(test_loader)
    pbar = tqdm(pbar, total=nb)
    for i, (input_tensor, slideID, class_label, pos_list) in pbar:
        input_tensor = input_tensor.to(rank)

        # MILとバッチ学習のギャップを吸収
        for bag_num in range(input_tensor.shape[0]):
            # print(slideID[bag_num], pos_list[bag_num].shape, input_tensor[bag_num].shape, pos_list[bag_num,0], rank)
            with torch.no_grad():
                class_prob, class_hat, A, yolo_out = model(input_tensor[bag_num])
                
            # if opt.save_bbox:
            bbox_list = cal_bbox(opt, yolo_out, slideID[bag_num], pos_list[bag_num])
            
            pred_label.append([class_label, class_hat])

            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist() # listに変換

            # bagの分類結果と各パッチのattention_weightを出力
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = ['', slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax # [slideID, 真のラベル, 予測ラベル] + [y_prob[0], y_prob[1], y_prob[2]]
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = pos_list[bag_num,:,0].tolist()
            pos_y = pos_list[bag_num,:,1].tolist()
            f_writer.writerow(['pos_x']+pos_x) # 座標書き込み
            f_writer.writerow(['pos_y']+pos_y) # 座標書き込み
            attention_weights = A.cpu().squeeze(0) # 1次元目削除[1,100] --> [100]
            att_list = attention_weights.tolist()
            f_writer.writerow(['attention']+att_list) # 各instanceのattention_weight書き込み
            f.close()
            
            # bbox_listを保存する
            f = open(output_file.parent/'bbox_coords'/f'{slideID[bag_num]}.txt', 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            f_writer.writerows(bbox_list)
            f.close()

def test(rank, world_size, opt):
    setup(rank, world_size)
    
    save_dir = Path(opt.save_dir)
    result = opt.result
    
    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # 訓練用と検証用に症例を分割
    colorstr('data load:')
    import dataset_kurume as ds
    if opt.classify_mode == 'normal_tree' or opt.classify_mode == 'kurume_tree':
        _, test_dataset, label_num = ds.load_leaf(opt, rank)
    elif opt.classify_mode == 'subtype':
        _, test_dataset, label_num = ds.load_svs(opt, rank)
        
    params_dir = save_dir / 'model_params'
    if os.path.exists(params_dir) and os.listdir(params_dir):
        best_file = params_dir / 'best.pt'
        ckpt_file = params_dir / 'ckpt.pt'
        
        best_param = torch.load(best_file, map_location='cuda')  # load best parameter
        ckpt_data = torch.load(ckpt_file) # load checkpoint file
        
        best_epoch = ckpt_data['epoch']
        del ckpt_data
    else:
        print('There is not best.pt file') if rank==0 else None
        exit()

    # model読み込み
    from make_model import test_model_yolo
    model = test_model_yolo(opt, label_num, save_dir)
    model = model.to(rank)
    
    # # DDP mode
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    model = DDP(model, device_ids=[rank])

    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    data_test = dataloader_svs.Dataset_svs(
        train=False,
        transform=transform,
        dataset=test_dataset,
        class_count=label_num,
        mag=opt.mag,
        bag_num=50,
        bag_size=100
    )
    
    #下のDataLoaderで設定したbatch_sizeで各GPUに分配
    test_sampler = torch.utils.data.distributed.DistributedSampler(data_test, rank=rank)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        sampler=test_sampler
    )

    # 学習
    model_test(rank, opt, model, test_loader, result)

def parse_opt(known=False):    
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('-n', '--number', default='', help='choose training number')
    parser.add_argument('--depth', default='1', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    parser.add_argument('--yolo_ver', default=None, help='choose weight version')
    parser.add_argument('--data', default='2nd', choices=['1st', '2nd', '3rd'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--multistage', default=1, type=int)
    parser.add_argument('--detect_obj', default=None, type=int)
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='kurume_tree', choices=['normal_tree', 'kurume_tree', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.0005, type=float)
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
    if RANK in [-1, 0]:
        print(colorstr('test: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    opt.save_dir = Path('./runs/' + utils.make_dirname(opt)) / f'train{opt.train}'
    if opt.number:
        opt.save_dir = Path(str(opt.save_dir) + f'-{opt.number}')
    else:
        dir_list = paths_sorted(list(opt.save_dir.parent.glob(fr'train{opt.train}*')))
        opt.save_dir = dir_list[-1]
    print(colorstr(opt.save_dir) + ': test this directory')
    
    # DDP mode
    device = select_device(opt.device)
    num_gpu = len(opt.device.split(','))
    
    # resultファイルの作成
    if opt.save_dir.exists():
        save_dir = Path(opt.save_dir)
        save_dir = save_dir / 'test_result' / 'bbox_coords'
        save_dir.mkdir(parents=True, exist_ok=True)
        opt.result = save_dir.parent / f'test_{opt.mag}_{opt.lr}_train-{opt.train}_best.csv'
        if os.path.exists(opt.result) and not opt.exist_ok:
            print(f'[{opt.result}] has been already done')
            exit()
        f = open(opt.result, 'w')
        f.close()

        # Test
        mp.spawn(test, args=(num_gpu, opt), nprocs=num_gpu, join=True)
    else:
        print(colorstr(opt.save_dir)+' is not exists')
        exit()
        
    if WORLD_SIZE > 1 and RANK == 0:
        _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
