import os
import re
import sys
import time
import glob
import csv
import shutil
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageStat, ImageDraw

import cv2
import openslide
import torch
import torchvision

from utils import colorstr, make_dirname, make_filename

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

def load_bbox(bbox_file):
    with open(bbox_file, 'r') as f:
        data = [x.split(',') for x in f.read().strip().splitlines() if len(x)]
    
    bbox_list = {}
    for d in data:
        key = f'{d[0]}_{d[1]}'
        bbox_list[key] = np.array(d[2:], dtype=np.float32).reshape((-1,5))
    
    return bbox_list

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=True):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        self.pil = pil
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            s = sum(self.im.size) / 2  # mean shape
            f = font_size or max(round(s * 0.035), 12)
            try:
                self.font = ImageFont.truetype(font, size=f)
            except Exception as e:  # download TTF if missing
                print(f'WARNING: Annotator font {font} not found: {e}')
                url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/" + font
                torch.hub.download_url_to_file(url, font)
                print(f'Annotator font successfully downloaded from {url} to {font}')
                self.font = ImageFont.truetype(font, size=f)
            self.fh = self.font.getsize('a')[1] - 3  # font height
        else:  # use cv2
            self.im = im
        s = sum(im.shape) / 2  # mean shape
        self.lw = line_width or max(round(s * 0.003), 2)  # line width

    def box_label(self, box, label='', width='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        lw = width if width else self.lw
        if self.pil:
            self.draw.rectangle(box, width=lw, outline=color)  # box
            if label:
                w = self.font.getsize(label)[0]  # text width
                self.draw.rectangle([box[0], box[1] - self.fh, box[0] + w + 1, box[1] + 1], fill=color)
                self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')
        else:  # cv2
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
                c2 = c1[0] + w, c1[1] - h - 3
                cv2.rectangle(self.im, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf,
                            lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

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

# Slide単位でattentionを読み込む
def load_att_data(opt):
    test_dir_list = [Path(f'./runs/{opt.dir_name}/{best}/test_result') for best in opt.best_list]
    test_fn_list = []
    for test_dir in test_dir_list:
        test_fn = list(test_dir.glob(r'*.csv'))
        if test_fn:
            test_fn_list.append(test_fn[0])
            print(colorstr(f'{test_fn[0].parent.name}/{test_fn[0].name}'))
        else:
            print(colorstr(test_dir)+' not exist')

    att_data_list = {}
    for test_fn in test_fn_list:
        csv_data = open(test_fn)
        reader = csv.reader(csv_data)
        row_number = 0
        for row in reader:
            if row_number%4==0:
                row_number = 0
                slideID = row[1]
                if slideID not in att_data_list:
                    att_data_list[slideID] = [row[2:],[],[],[]]
            else:
                att_data_list[slideID][row_number] += row[1:]
            row_number += 1
    print(len(att_data_list))
    
    return att_data_list

# bag単位でattentionを読み込む
def load_bagatt_data(opt):
    test_dir_list = [Path(f'./runs/{opt.dir_name}/{best}/test_result') for best in opt.best_list]
    test_fn_list = []
    for test_dir in test_dir_list:
        test_fn = list(test_dir.glob(r'*.csv'))
        if test_fn:
            test_fn_list.append(test_fn[0])
            print(colorstr(f'{test_fn[0].parent.name}/{test_fn[0].name}'))
        else:
            print(colorstr(test_dir)+' not exist')

    bagatt_data_list = {}
    for test_fn in test_fn_list:
        csv_data = open(test_fn)
        reader = csv.reader(csv_data)
        row_number = 0
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                row_number = 0
                slideID = row[1]
                true_label = row[2]
                pred_label = row[3]

                if slideID not in bagatt_data_list:
                    bagatt_data_list[slideID] = [row[2:],[[],[],[]],[[],[],[]],test_fn.parent]
            else:
                if true_label == pred_label:
                    bagatt_data_list[slideID][1][row_number] += row[1:]
                elif true_label != pred_label:
                    bagatt_data_list[slideID][2][row_number] += row[1:]
                row_number += 1
    print(len(bagatt_data_list))

    return bagatt_data_list

def draw_heatmap(opt):
    print(colorstr('make attention heat map'))
    b_size = 224
    t_size = 4

    att_data_list = load_att_data(opt)
    save_dir = Path(f'./runs/{opt.dir_name}/attention_map')
    save_dir.mkdir(exist_ok=True)

    bar = tqdm(total = len(att_data_list))
    for slideID in att_data_list:
        bar.set_description(slideID)
        bar.update(1)
        pos_x = np.array([int(x) for x in att_data_list[slideID][1]])
        pos_y = np.array([int(y) for y in att_data_list[slideID][2]])
        att = np.array([float(a) for a in att_data_list[slideID][3]])

        # 極端なattentionとして上位50個はカット
        # attention mapの濃淡が極端になることを抑える
        sort_att = np.argsort(att)[::-1]
        sort_att = sort_att[10:-50]
        att_max = max(att[sort_att])
        att_min = min(att[sort_att])
        for i in range(len(att)):
            att[i] = (att[i] - att_min) / (att_max - att_min) #attentionを症例で正規化
        
        img = cv2.imread(f'{DATA_PATH}/hirono/svs_info_40x/{slideID}/{slideID}_thumb.tif')
        thumb = cv2.imread(f'{DATA_PATH}/hirono/svs_info_40x/{slideID}/{slideID}_thumb.tif')
        
        height, width = img.shape[0], img.shape[1]
        w_num = width // t_size
        h_num = height // t_size

        cmap = plt.get_cmap('jet')
        att_map = np.ones((h_num, w_num,3))*255
        for i in sort_att:
            x = pos_x[i]//b_size
            y = pos_y[i]//b_size

            cval = cmap(float(att[i]))
            att_map[y,x,:] = [cval[2]*255, cval[1]*255, cval[0]*255]
            color = (cval[2] * 255, cval[1] * 255, cval[0] * 255)
            if opt.mag == '40x':
                start = (int(pos_x[i]/b_size*t_size), int(pos_y[i]/b_size*t_size))
                end = (int(pos_x[i]/b_size*t_size+t_size), int(pos_y[i]/b_size*t_size+t_size))
                cv2.rectangle(img, start, end, color, thickness=-1)
            elif opt.mag == '20x':
                start = (int(pos_x[i]/b_size*t_size-t_size/2), int(pos_y[i]/b_size*t_size-t_size/2))
                end = (int(pos_x[i]/b_size*t_size+t_size*3/2), int(pos_y[i]/b_size*t_size+t_size*3/2))
                cv2.rectangle(img, start, end, color, thickness=-1)
            elif opt.mag == '10x':
                start = (int(pos_x[i]/b_size*t_size-t_size*3/2), int(pos_y[i]/b_size*t_size-t_size*3/2))
                end = (int(pos_x[i]/b_size*t_size+t_size*5/2), int(pos_y[i]/b_size*t_size+t_size*5/2))
                cv2.rectangle(img, start, end, color, thickness=-1)
            elif opt.mag == '5x':
                start = (int(pos_x[i]/b_size*t_size-t_size*7/2), int(pos_y[i]/b_size*t_size-t_size*7/2))
                end = (int(pos_x[i]/b_size*t_size+t_size*9/2), int(pos_y[i]/b_size*t_size+t_size*9/2))
                cv2.rectangle(img, start, end, color, thickness=-1)

        att_map = cv2.resize(np.uint8(att_map), (width, height))
        cv2.imwrite(f'{save_dir}/{slideID}_{slideID_name_dict[slideID]}_map.tif', att_map)
        cv2.imwrite(f'{save_dir}/{slideID}_{slideID_name_dict[slideID]}_blend.tif', img)
        cv2.imwrite(f'{save_dir}/{slideID}_{slideID_name_dict[slideID]}_thumb.tif', thumb)
    
def save_high_low_patches(opt):
    print(colorstr('make high&low patch tiles'))

    bagatt_data_list = load_bagatt_data(opt)
    save_dir = Path(f'./runs/{opt.dir_name}/attention_patch')
    save_dir.mkdir(exist_ok=True)

    bar = tqdm(total = len(bagatt_data_list))
    print(bagatt_data_list.keys())
    for slideID in bagatt_data_list:
        bar.set_description(slideID)
        bar.update(1)
        
        data = bagatt_data_list[slideID]
        label = data[0][0]
        true_data = data[1]
        false_data = data[2]
        test_dir = data[3]
        att = np.array([float(a) for a in true_data[2]+false_data[2]])

        att_max = max(att)
        att_min = min(att)
        
        true_data[0] = np.array([int(x) for x in true_data[0]])
        true_data[1] = np.array([int(y) for y in true_data[1]])
        true_data[2] = np.array([(float(a)-att_min)/(att_max-att_min) for a in true_data[2]])

        false_data[0] = np.array([int(x) for x in false_data[0]])
        false_data[1] = np.array([int(y) for y in false_data[1]])
        false_data[2] = np.array([(float(a)-att_min)/(att_max-att_min) for a in false_data[2]])

        save_patch(opt, slideID, label, true_data, test_dir, save_dir, 'correct')
        save_patch(opt, slideID, label, false_data, test_dir, save_dir, 'incorrect') 
        
def save_patch(opt, slideID, label, data, test_dir, save_dir, c_flag):
    if len(data[2]) > 0:
        svs_fn = [s for s in svs_fn_list if slideID in s[:11]]
        svs = openslide.OpenSlide(f'/Raw/Kurume_Dataset/svs/{svs_fn[0]}')

        sort_idx = np.argsort(data[2])[::-1]
        save_many_patch(opt, slideID, svs, sort_idx, label, data, test_dir, save_dir, c_flag, 'high')  

        sort_idx = np.argsort(data[2])
        save_many_patch(opt, slideID, svs, sort_idx, label, data, test_dir, save_dir, c_flag, 'low')

def save_many_patch(opt, slideID, svs, sort_idx, label, data, test_dir, save_dir, c_flag, hl_flag):
    b_size = 224
    img_num = 50
    nrow = 10
    write_bbox = False
    images = np.zeros((img_num, b_size, b_size, 3), np.uint8)
    fig, ax = plt.subplots()
    
    bbox_file = test_dir / 'bbox_coords'/ f'{slideID}.txt'
    if os.path.exists(bbox_file):
        bbox_list = load_bbox(bbox_file)
        write_bbox = True
    else:
        with open('./runs/error_log.txt', 'a') as f:
            f.write(f'No such file : {bbox_file.name}\n')
    
    for i in range(img_num):
        idx = sort_idx[i]
        pos = [data[0][idx], data[1][idx]]
        att = data[2][idx]

        if opt.mag == '40x':
            b_img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
        elif opt.mag == '20x':
            b_img = svs.read_region((pos[0]-(int(b_size/2)),pos[1]-(int(b_size/2))),0,(b_size*2,b_size*2)).convert('RGB')
        elif opt.mag == '10x':
            b_img = svs.read_region((pos[0]-(int(b_size*3/2)),pos[1]-(int(b_size*3/2))),1,(b_size,b_size)).convert('RGB')
        elif opt.mag == '5x':
            b_img = svs.read_region((pos[0]-(int(b_size*7/2)),pos[1]-(int(b_size*7/2))),1,(b_size*2,b_size*2)).convert('RGB')
        b_img = svs.read_region((pos[0], pos[1]), 0, (b_size,b_size)).convert('RGB')
        b_img = np.array(b_img, dtype=np.uint8)
        
        if write_bbox:
            annotator = Annotator(b_img, line_width=1, pil=False)
            boxes = bbox_list[f'{pos[0]}_{pos[1]}']
            for b in boxes:
                annotator.box_label(b[0:4], width=2, color=colors(b[0]+10, True))
    
        images[i] = b_img
    
    images = np.transpose(images, [0,3,1,2]) # NHWC -> NCHW に変換
    images_tensor = torch.as_tensor(images)
    joined_images_tensor = torchvision.utils.make_grid(images_tensor, nrow=nrow, padding=10)
    joined_images = joined_images_tensor.numpy()

    jointed = np.transpose(joined_images, [1,2,0]) # NCHW -> NHWCに変換
    plt.tick_params(color='white', labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    img_name = f'{slideID}_{label}_{c_flag}_{slideID_name_dict[slideID]}'
    plt.title(f'{img_name}')
    plt.imshow(jointed)

    save_dir = save_dir / f'{label}_{c_flag}_{hl_flag}' / 'attention'
    save_dir.mkdir(exist_ok=True, parents=True) 
    plt.savefig(save_dir.parent/f'{img_name}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=600)
    plt.clf()

    f = open(save_dir/f'{img_name}.csv', 'w')
    for d in data:
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow(d[0:50])
    f.close()

def make_histgram(opt):
    save_dir = Path(f'./runs/{opt.dir_name}/attention_patch')
    dir_list = [
        ['0_correct_high', '0_incorrect_high'],
        ['1_correct_high', '1_incorrect_high'],
    ]
    
    for i, dir in enumerate(dir_list):
        print(dir)
        dir_path = save_dir / dir[0]
        file_list = os.listdir(dir_path)
        correct_subtype_list = [slideID_name_dict[f.split('_')[0]] for f in file_list if os.path.isfile(f'{dir_path}/{f}')]
        count_dict = collections.Counter(correct_subtype_list)
        subtype_list = list(count_dict.keys())
        correct_count = list(count_dict.values())
        
        dir_path = save_dir / dir[1]
        file_list = os.listdir(dir_path)
        incorrect_subtype_list = [slideID_name_dict[f.split('_')[0]] for f in file_list if os.path.isfile(f'{dir_path}/{f}')]
        incorrect_count = [incorrect_subtype_list.count(subtype) for subtype in subtype_list]
        
        print(subtype_list)
        print(correct_count)
        print(incorrect_count)

        plt.rcParams['figure.subplot.bottom'] = 0.30
        fig = plt.bar(subtype_list, correct_count, align='edge', width=-0.3, label='correct')
        fig = plt.bar(subtype_list, incorrect_count, align='edge', width=0.3, label='incorrect')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(save_dir / f'hist_{i}.png')
        plt.clf()



DATA_PATH = '/Dataset/Kurume_Dataset'
SAVE_PATH = '.'

svs_fn_list = os.listdir(f'/Raw/Kurume_Dataset/svs')
slideID_name_dict = get_slideID_name()

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--yolo_ver', default=None, help='choose weight version')
    parser.add_argument('--data', default='add', choices=['', 'add'])
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--multistage', default=1, type=int)
    parser.add_argument('--detect_obj', default=None, type=int)
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='new_tree', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-g', '--gamma', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    opt = parser.parse_args()

    # if opt.data == 'add':
    #     opt.reduce = True

    if opt.classify_mode != 'subtype':
        if opt.depth == None:
            print(f'mode:{opt.classify_mode} needs depth param')
            exit()

    if opt.loss_mode == 'LDAM' and opt.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()
    
    if (opt.loss_mode == 'focal' or opt.loss_mode == 'focal-weight') and opt.gamma == None:
        print(f'when loss_mode is focal, input gamma param')
        exit()

    return opt

if __name__ == '__main__':
    opt = parse_args()

    opt.dir_name = make_dirname(opt)
    opt.file_name = make_filename(opt)

    with open(f'./runs/{opt.dir_name}/best_exps.txt') as f:
        opt.best_list = [s.strip() for s in f.readlines()]
    print(opt.best_list)

    draw_heatmap(opt)
    save_high_low_patches(opt)
    make_histgram(opt)
