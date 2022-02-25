from itertools import count

from torch.nn.modules import loss
from torch.nn.modules.conv import LazyConvTranspose2d
from dataloader_svs import Dataset_svs
import os
import csv
from PIL import Image, ImageStat
import numpy as np
import cv2
import openslide
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp
import torch.distributed as dist

""" spawnの同期の確認 """
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

#     # initialize the process group
#     # winではncclは使えないので、gloo
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
# def train(rank, num_gpu, opt):
#     setup(rank, num_gpu)
#     print(rank)
#     for i in range(100):
#         f = open('douki.csv', 'a')
#         f_writer = csv.writer(f, lineterminator='\n')
#         csv_header = [rank for j in range(10000)]
#         f_writer.writerow(csv_header)
#         f.close()
        

# def main():
#     f = open('douki.csv', 'w')
#     f.close()
    
#     num_gpu = 3
#     opt = 5
#     mp.spawn(train, args=(num_gpu, opt), nprocs=num_gpu, join=True)

# if __name__ == "__main__":
#     main()

# a = 0.123124535
# print('aaa_'+'{:.4f}'.format(a))

""" mail送信の確認 """
# import smtplib
# from email.mime.text import MIMEText
# from email.utils import formatdate
# MAIL_ADDRESS = 'nitech28114106@gmail.com'
# PASSWORD = 'yrxqmyxhkglnpfsq'

# smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
# smtpobj.ehlo()
# smtpobj.starttls()
# smtpobj.ehlo()
# smtpobj.login(MAIL_ADDRESS, PASSWORD)

# body_msg = 'test message'
# subject = 'test title'
# from_addr = 'ckv14106@nitech.jp'
# to_addr = 'yyph.fam@gmail.com'
# msg = MIMEText(body_msg)
# msg['Subject'] = subject
# msg['From'] = from_addr
# msg['To'] = to_addr
# msg['Date'] = formatdate()

# smtpobj.sendmail(from_addr, to_addr, msg.as_string())
# smtpobj.close()

""" softmaxとcross entropyの計算の確認 """
x = torch.tensor([[0.222,0.978],[0.3335,0.886]])
target = torch.tensor([0,1])
index = F.one_hot(target, 2).type(torch.uint8)

loss_fn = nn.CrossEntropyLoss()
x_soft = F.softmax(x, dim=1)
loss = -1. * index * torch.log(x_soft)
print(x_soft)

soft1 = torch.exp(x[0,0])/(torch.exp(x[0,0])+torch.exp(x[0,1]))
soft2 = torch.exp(x[1,1])/(torch.exp(x[1,0])+torch.exp(x[1,1]))
print(soft1,soft2)

loss1 = loss_fn(x, target)
loss2 = -x[0,target[0]]+torch.log(torch.exp(x[0,0])+torch.exp(x[0,1]))
loss3 = -x[1,target[1]]+torch.log(torch.exp(x[1,0])+torch.exp(x[1,1]))
loss4 = -torch.log(soft1)
loss5 = -torch.log(soft2)
print(loss1,loss2,loss3,(loss2+loss3)/2,loss4,loss5,(loss4+loss5)/2)
print(loss.sum(),loss.sum()/2)


# a = np.array([[1,2,3],[4,5,6]])
# b = np.array([[10,20,30],[40,50,60]])
# c = np.array([[100,200,300],[400,500,600]])
# A = [a,b,c]
# # A = np.array(A)
# A = np.stack([a,b,c])
# print(A)

# vgg = models.vgg16(pretrained=True)
# print(vgg)
# print(nn.Sequential(*list(vgg.children())[:-1]))

# x = torch.arange(196).view(1,14,14).float()
# y = nn.AdaptiveAvgPool2d((7,7))(x)
# print(x)
# print(y)

# DATA_PATH = '/Dataset/Kurume_Dataset' # データディレクトリ
# SVS_PATH = '/Raw/Kurume_Dataset'
# slideID = '180183'
# pos_list = np.loadtxt(f'{DATA_PATH}/svs_info/{slideID}/{slideID}.csv', delimiter=',', dtype='int')
# np.random.shuffle(pos_list)
# pos = pos_list[0,:].tolist()

# b_size = 224
# svs_list = os.listdir(f'{SVS_PATH}/svs')
# svs_fn = [s for s in svs_list if slideID in s[:11]]
# svs = openslide.OpenSlide(f'{SVS_PATH}/svs/{svs_fn[0]}')
# img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
# print(img)
# img.save(f'./patch_{pos[0]}_{pos[1]}.png')

# device = 'cuda:0'
# class_num_list = np.array([10,90])
# weights = torch.tensor([1/(10/100),1/(90/100)]).to(device)
# print(weights)
# x = torch.tensor([[0.3335,0.886],[0.3335,0.886]]).to(device)
# target = torch.tensor([0,1]).to(device)

# loss_fn = nn.CrossEntropyLoss().to(device)
# loss_fn_w = nn.CrossEntropyLoss(weight=weights).to(device)
# loss1 = loss_fn(x, target)
# loss2 = loss_fn_w(x, target)
# loss3 = -x[0,target[0]]+torch.log(torch.exp(x[0,0])+torch.exp(x[0,1]))
# loss4 = -x[1,target[1]]+torch.log(torch.exp(x[1,0])+torch.exp(x[1,1]))
# print(loss1, loss2, loss3, loss4, (weights[0]*loss3+weights[1]*loss4)/torch.sum(weights))

# sm = F.log_softmax(x, dim=1)
# print(F.nll_loss(sm, target, weight=weights))
# print(weights/torch.sum(weights))

# m_list = 1.0/np.sqrt(np.sqrt(class_num_list))
# m_list = m_list * 0.5
# m_list = torch.cuda.FloatTensor(m_list)
# m_list = m_list[None, :]
# index = F.one_hot(target, len(class_num_list)).type(torch.uint8)
# # index = index.type(torch.uint8)
# index_float = index.type(torch.cuda.FloatTensor)
# batch_m = torch.matmul(m_list, index_float.transpose(0,1))
# print(m_list,index,batch_m)
# x_m = x - batch_m
# print(x_m,x,index)
# output = torch.where(index, x_m, x)
# print(output)

# def check_patch(img, svs, pos_list):
#     bar = tqdm(total = len(pos_list))
#     count = 0
#     error = None
#     for pos in pos_list:
#         bar.update(1)
#         try:
#             patch = img.read_region((pos[0], pos[1]), 0, (b_size, b_size))
#         except:
#             if error is None:
#                 error = [svs,pos[0],pos[1]]
#                 print(error)
#             count += 1
#     if error is not None:
#        error = error + [len(pos_list), count]
#     return error

# DATA_PATH = '/Dataset/Kurume_Dataset' 
# SVS_PATH = '/Raw/Kurume_Dataset'

# b_size = 224

# svs_info_list = os.listdir(f'{DATA_PATH}/svs_info')
# svs_list = os.listdir(f'{SVS_PATH}/svs')

# split_num = 10
# args = sys.argv
# if len(args) > 1:
#     group = int(args[1])
#     split = len(svs_info_list)//split_num
#     if group<(split_num-1):
#         svs_info_list = svs_info_list[group*split : (group+1)*split]
#     if group==(split_num-1):
#         svs_info_list = svs_info_list[group*split : len(svs_info_list)]

# svs_info_list = ['180077','180146','180246','180091','180287','180619']
# print(svs_info_list)
# error_list = []

# for idx, svs in enumerate(svs_info_list):
#     if svs == 'error_list.txt':
#         continue
#     print(f'svs:{svs} {idx}/{len(svs_info_list)}')

#     svs_fn = [s for s in svs_list if svs in s[:11]]
#     pos_list = np.loadtxt(f'{DATA_PATH}/svs_info/{svs}/{svs}.csv', delimiter=',', dtype='int')
 
#     img = openslide.OpenSlide(f'{SVS_PATH}/svs/{svs_fn[0]}')

#     error = check_patch(img, svs, pos_list)
#     if error is not None:
#         error_list.append(error)

# if not os.path.exists('./error_list.csv'):
#     f = open('error_list.csv', 'w')
#     f_writer = csv.writer(f, lineterminator='\n')
#     f_writer.writerow(['SlideID','w_i', 'h_i', 'pos_count', 'error_count'])
#     f.close()

# f = open('error_list.csv', 'a')
# f_writer = csv.writer(f, lineterminator='\n')
# f_writer.writerows(error_list)
# f.close()
