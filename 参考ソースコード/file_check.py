import csv
import os
import sys
import argparse

import numpy as np
from pathlib import Path

def read_leafCSV(filepath, label):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}') and row[1] != 'OI_ILPD':
        # if row[1] != 'OI_ILPD':
            file_data.append([row[0], label, row[1]])
        elif row[1] == 'OI_ILPD':
            # print(f'slideID{row[0]}はOI_ILPDです')
            continue
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data

def reduce_data(data):
    flag_list = {}
    csv_data = open('../KurumeTree/add_data/add_flag_list.csv')
    reader = csv.reader(csv_data)
    for row in reader:
        flag_list[row[0]]=int(row[1])
    csv_data.close()
    
    reduced_data = []
    for d in data:
        if flag_list[d[0]] == 0:
           reduced_data.append([d[0], d[1], d[2]])

    return reduced_data 

def load_leaf(args):
    if args.data:
        args.data = f'{args.data}_'
    if args.classify_mode == 'leaf':
        dir_path = f'../KurumeTree/{args.data}result/{args.name}/unu_depth{args.depth}/leafs_data'
    if args.classify_mode == 'new_tree':
        dir_path = f'../KurumeTree/{args.data}result_teacher/FDC/{args.name}/unu_depth{args.depth}/leafs_data'

    if args.leaf != None:
        if int(args.leaf[0])%2!=0 or int(args.leaf[1])-int(args.leaf[0])!=1:
            print('隣接した葉ではありません')
            exit()

        leaf_data = []
        for num in args.leaf:
            data = read_leafCSV(f'{dir_path}/leaf_{num}.csv', int(num)%2)
            leaf_data.append(data)
        min_leaf = np.argmin([len(leaf_data[0]), len(leaf_data[1])])
        max_leaf = np.argmax([len(leaf_data[0]), len(leaf_data[1])])
        ratio = len(leaf_data[max_leaf])//len(leaf_data[min_leaf])
        print(f'{min_leaf}:{len(leaf_data[min_leaf])},{max_leaf}:{len(leaf_data[max_leaf])}')
        if args.reduce:
            leaf_data[max_leaf] = reduce_data(leaf_data[max_leaf])
            min_leaf = np.argmin([len(leaf_data[0]), len(leaf_data[1])])
            max_leaf = np.argmax([len(leaf_data[0]), len(leaf_data[1])])
            ratio = len(leaf_data[max_leaf])//len(leaf_data[min_leaf])
            print(f'many data reduced \n{min_leaf}:{len(leaf_data[min_leaf])},{max_leaf}:{len(leaf_data[max_leaf])}')

        dataset = [[] for i in range(5)]
        for num in args.leaf:
            for idx, slide in enumerate(leaf_data[int(num)]):
                dataset[idx%5].append(slide+[0])

                # if args.augmentation and int(num) == min_leaf:
                #     for aug_i in range(np.min([7, ratio-1])):
                #         dataset[idx%5].append(slide+[aug_i+1])
                        
        return dataset

    else: 
        print('Please input leaf number')
        exit()

def parse_opt(known=False):    
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--yolo_ver', default=None, help='choose weight version')
    parser.add_argument('--data', default='add', choices=['', 'add'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--num_gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='new_tree', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
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
    '123':'4',
    '234':'5',
    '345':'1',
    '451':'2',
    '512':'3',
}

opt = parse_opt()

dataset = load_leaf(opt)
if opt.data:
    SAVE_DIR = Path(f'./data/{opt.data}_depth{opt.depth}/')
    if opt.reduce:
        SAVE_DIR = Path(f'./data/{opt.data}_reduce_depth{opt.depth}/')
else:
    SAVE_DIR = Path(f'./data/depth{opt.depth}/')
SAVE_DIR.mkdir(exist_ok=True, parents=True)

print(dataset[0],len(dataset[0]))

for split in range(5):
    data = dataset[split]
    file_path = SAVE_DIR / f'split{split}.csv'
    with open(file_path, 'a') as f:
        f_writer = csv.writer(f, lineterminator='\n')
        for d in data:
            f_writer.writerow(d)