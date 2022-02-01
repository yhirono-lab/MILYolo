from make_log_Graphs import load_logfile
import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import openslide
import torch
import torchvision
from tqdm import tqdm
import utils

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_slideID_name():
    file_name = '../KurumeTree/add_data/Data_SimpleName_svs.csv'
    csv_data = np.loadtxt(file_name, delimiter=',', dtype='str')
    name_dict = {}

    for i in range(1,csv_data.shape[0]):
        if csv_data[i,1] == 'N/A':
            csv_data[i,1] = 'NA'
        if csv_data[i,1] == 'CLL/SLL':
            csv_data[i,1] = 'CLL-SLL'
        if 'N/A' in csv_data[i,1]:
            csv_data[i,1] = csv_data[i,1].replace('N/A','NA')
        if 'CLL/SLL' in csv_data[i,1]:
            csv_data[i,1] = csv_data[i,1].replace('CLL/SLL','CLL-SLL')
        name_dict[csv_data[i,0]] = csv_data[i,1]

    return name_dict

# Slide単位でattentionを読み込む
def load_bagatt_data(args, target_list=None):
    dir_name = args.dir_name
    test_fn_list = os.listdir(f'./test_result/{dir_name}')
    test_fn_list = [test_fn for test_fn in test_fn_list if args.mag in test_fn and str(args.lr) in test_fn and 'epoch' in test_fn]
    print(test_fn_list)

    bagatt_data_list = {}
    for test_fn in test_fn_list:
        csv_data = open(f'./test_result/{dir_name}/{test_fn}')
        reader = csv.reader(csv_data)
        row_number = 0
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                row_number = 0
                slideID = row[1]
                true_label = row[2]
                pred_label = row[3]

                slide_name = slideID_name_dict[slideID]

                if slideID not in bagatt_data_list and true_label == args.label and slide_name in target_list:
                    bagatt_data_list[slideID] = [row[2:],[[],[],[]]]
            else:
                if true_label == pred_label and true_label == args.label and slide_name in target_list:
                    bagatt_data_list[slideID][1][row_number] += row[1:]
                row_number += 1
    print(len(bagatt_data_list))

    return bagatt_data_list


def save_high_low_patches(args, data_list):
    print('make high&low patch tiles')

    dir_name = args.dir_name
    save_dir = f'./yolo_data/{dir_name}-{args.label}/{args.mag}_{args.lr}'
    makedir(save_dir)

    bar = tqdm(total = len(data_list))
    print(data_list.keys())
    for slideID in data_list:
        bar.set_description(slideID)
        bar.update(1)
        
        data = data_list[slideID]
        label = data[0][0]
        true_data = data[1]
        att = [float(a) for a in true_data[2]]
        if len(att) > 0:
            att_max = max(att)
            att_min = min(att)
            
            true_data[0] = [int(x) for x in true_data[0]]
            true_data[1] = [int(y) for y in true_data[1]]
            true_data[2] = [(float(a)-att_min)/(att_max-att_min) for a in true_data[2]]

            save_patch(slideID, args.mag, label, true_data, save_dir)
        
def save_patch(slideID, mag, label, data, save_dir):
    b_size = 224
    img_num = 100

    svs_fn = [s for s in svs_fn_list if slideID in s[:11]]
    svs = openslide.OpenSlide(f'/Raw/Kurume_Dataset/svs/{svs_fn[0]}')

    sort_idx = np.argsort(data[2])[::-1]
    for i in range(img_num):
        idx = sort_idx[i]
        pos = [data[0][idx], data[1][idx]]
        att = data[2][idx]

        if mag == '40x':
            b_img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
        elif mag == '20x':
            b_img = svs.read_region((pos[0]-(int(b_size/2)),pos[1]-(int(b_size/2))),0,(b_size*2,b_size*2)).convert('RGB')
        elif mag == '10x':
            b_img = svs.read_region((pos[0]-(int(b_size*3/2)),pos[1]-(int(b_size*3/2))),1,(b_size,b_size)).convert('RGB')
        elif mag == '5x':
            b_img = svs.read_region((pos[0]-(int(b_size*7/2)),pos[1]-(int(b_size*7/2))),1,(b_size*2,b_size*2)).convert('RGB')

        b_img = svs.read_region((pos[0], pos[1]), 0, (b_size,b_size)).convert('RGB')
        img_name = f'{slideID}_{label}_{slideID_name_dict[slideID]}_{pos[0]}_{pos[1]}'
        b_img.save(f'{save_dir}/{img_name}.tif')


DATA_PATH = '/Dataset/Kurume_Dataset'

svs_fn_list = os.listdir(f'/Raw/Kurume_Dataset/svs')
slideID_name_dict = get_slideID_name()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default='2', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    parser.add_argument('--label', default='0')
    parser.add_argument('--data', default='add', choices=['', 'add'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='', choices=['', 'vgg11'])
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='new_tree', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='myinvarse', choices=['normal','myinvarse','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-g', '--gamma', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()
    
    if args.data == 'add':
        args.data = 'add_'
        args.reduce = True

    if args.classify_mode != 'subtype':
        if args.depth == None:
            print(f'mode:{args.classify_mode} needs depth param')
            exit()

    if args.loss_mode == 'LDAM' and args.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()

    if args.loss_mode == 'focal' and args.gamma == None:
        print(f'when loss_mode is focal, input gamma param')
        exit()

    target_name_list = ['AITL', 'ATLL', 'PTCL_NOS']
    args.dir_name = utils.make_dirname(args)
    print(args.dir_name)
    
    data_list = load_bagatt_data(args, target_name_list)
    print(data_list.keys())

    save_high_low_patches(args, data_list)


    # print(SlideID_name_dict)
    # print(args.label)
    # print(len(list(data.keys())))

