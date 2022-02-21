import os
import glob
import shutil
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from re import S
from pathlib import Path
from statistics import harmonic_mean as hmean
from sklearn.base import MetaEstimatorMixin
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils import colorstr, make_dirname, make_filename, get_slideID_name

def load_logfile(save_dir):    
    log_fn = list(save_dir.glob(r'*.csv'))
    log = None

    if log_fn:
        log_fn = Path(log_fn[0])
        csv_file = np.loadtxt(log_fn, delimiter=',', dtype='str')

        if csv_file.ndim>1:
            print('load train log file :', f'{log_fn.parent.name}/{log_fn.name}:{csv_file.shape}')
            
            if csv_file[0,0].isalpha():
                log = csv_file[1:,1:].astype(np.float32)
            else:
                log = csv_file[0:,1:].astype(np.float32)
            
    return log

# バッグ単位の事後確率とラベルのリストを返す
def load_testresult(save_dir):
    test_fn = list(save_dir.glob(r'*.csv'))
    result_data = None

    if test_fn:
        test_fn = Path(test_fn[0])

        csv_data = []
        csv_file = open(test_fn)
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                csv_data.append([row[1], int(row[2]), int(row[3])])

        if len(csv_data)>0:
            result_data = np.asarray(csv_data)
            print('load test result :', f'{test_fn}:{result_data.shape}')

    return result_data

# # スライド単位の事後確率とラベルのリストを返す
# def load_slide_prob_label(opt):
#     dir_name = opt.dir_name
#     mag = opt.mag
#     lr = str(opt.lr)
    
#     pred_corpus = {}
#     label_corpus = {}
#     slide_id_list = []

#     result_fn_list = os.listdir(f'./test_result/{dir_name}')
#     result_fn_list = [result_fn for result_fn in result_fn_list if mag in result_fn and lr in result_fn and 'epoch' in result_fn]
#     print('load_bagresult:',result_fn_list)
    
#     split_num_list = []
#     for result_fn in result_fn_list:
#         csv_file = f'./test_result/{dir_name}/{result_fn}'
#         split_num = result_fn.split('-')[-2][0]
#         split_num_list.append(split_num)
#         with open(csv_file, 'r') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if(len(row)==6 or len(row)==9):
#                     slide_id = row[1]
#                     if len(row)==6:
#                         prob_list = [float(row[4]), float(row[5])]
#                     elif len(row)==9:
#                         prob_list = [float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])]
                    
#                     if(slide_id not in pred_corpus):
#                         pred_corpus[slide_id] = []
#                         label_corpus[slide_id] = int(row[2]) #正解ラベル

#                     pred_corpus[slide_id].append(prob_list)
#                     if((slide_id, split_num) not in slide_id_list):
#                         slide_id_list.append((slide_id, split_num))

#     # slide単位の事後確率計算
#     slide_prob = {f'split-{split_num}':[] for split_num in split_num_list}
#     slide_prob['all'] = []
#     slide_label_list = {f'split-{split_num}':[] for split_num in split_num_list}
#     slide_label_list['all'] = []

#     for slide_id, split_num in slide_id_list:
#         prob_list = pred_corpus[slide_id]
#         bag_num = len(prob_list) # Bagの数

#         total_prob_list = [0.0 for i in prob_list[0]]
#         for prob in prob_list:
#             total_prob_list = total_prob_list + np.log(prob)
#         total_prob_list = np.exp(total_prob_list / bag_num) 
        
#         label = [label_corpus[slide_id], np.argmax(total_prob_list)]

#         slide_prob[f'split-{split_num}'].append(list(total_prob_list))
#         slide_prob['all'].append(list(total_prob_list))
#         slide_label_list[f'split-{split_num}'].append(label)
#         slide_label_list['all'].append(label)
    
#     for key in slide_prob:
#         slide_prob[key] = np.array(slide_prob[key])
#         slide_label_list[key] = np.array(slide_label_list[key])

#     return slide_id_list, slide_prob, slide_label_list

def cal_log_ave(log_list, max_epoch):
    ave_log_data = np.zeros((max_epoch, 4))    
    for epoch in range(max_epoch):
        count = 0
        total_data = np.zeros(4)
        for key in log_list:
            log = log_list[key]
            if epoch < log.shape[0]:
                count += 1
                total_data += log[epoch]
        ave_data = total_data/count
        ave_log_data[epoch] += ave_data
    return ave_log_data

def save_split_graph(opt, data, exp, save_dir):
    result_dir = opt.result / 'graphs'
    result_dir.mkdir(exist_ok=True)
    
    file_name = opt.file_name
    # print(file_name)
    
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    x = np.array(range(1, data.shape[0]+1))
    train_loss = data[:,0]
    train_acc = data[:,1]
    valid_loss = data[:,2]
    valid_acc = data[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, train_loss, label='train_loss')
    ax.plot(x, train_acc, label='train_acc')
    ax.plot(x, valid_loss, label='valid_loss')
    ax.plot(x, valid_acc, label='valid_acc')

    ax.legend(loc='best')
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    plt.title(f'{file_name}_{exp}')

    plt.savefig(f'{save_dir}/acc_loss_graph_{exp}.png')
    if exp != 'average':
        plt.savefig(opt.result / 'graphs' / f'acc_loss_graph_{exp}.png')

    if exp == 'average':
        f = open(f'{save_dir}/average_log.csv', 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow(['epoch','train_loss','train_acc','valid_loss','valid_acc'])
        for idx, d in enumerate(data):
            f_writer.writerow([idx]+d.tolist())
        f.close()
    
def save_category_graph(opt, data_list, max_epoch):
    result_dir = opt.result / 'graphs'
    result_dir.mkdir(exist_ok=True)
    file_name = opt.file_name
    
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')
    
    x = np.array(range(1, max_epoch+1))
    cd_list = [[] for i in range(4)]
    cn_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
    key_list = []
    for key in data_list:
        data = data_list[key]
        key_list.append(key)
        cd_list[0].append(data[:max_epoch,0])
        cd_list[1].append(data[:max_epoch,1])
        cd_list[2].append(data[:max_epoch,2])
        cd_list[3].append(data[:max_epoch,3])
    
    for i in range(len(cn_list)):
        cn = cn_list[i]
        data_list = cd_list[i]
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for j, key in enumerate(key_list):
            ax.plot(x, data_list[j], label=key)

        ax.legend(loc='best')
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_ylim(0,)
        plt.title(cn)

        plt.savefig(f'{result_dir}/{cn}.png')
        

def save_test_cm(opt, test_result, exp, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = opt.file_name
    
    # print(test_result)
    cm = confusion_matrix(y_true=test_result[:,1], y_pred=test_result[:,2], labels=np.unique(test_result[:,1]).tolist())
    print(colorstr(f'{exp}\'s bag result\n'),cm)

    f = open(f'{save_dir}/test_analytics_{exp}.csv', 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    f_writer.writerow([file_name])
    f_writer.writerow(['Bag']+[f'pred:{i}' for i in range(len(cm))])
    total = 0
    correct = 0
    recall_list = []
    for i in range(len(cm)):
        row_total = 0
        for j in range(len(cm)):
            total += cm[i][j]
            row_total += cm[i][j]
        recall_list.append(cm[i][i]/row_total)
        correct += cm[i][i]
        f_writer.writerow([f'true:{i}']+cm[i].tolist())
    acc = correct/total
    score = hmean(recall_list)
    f_writer.writerow(['recall']+recall_list)
    f_writer.writerow(['total',total])
    f_writer.writerow(['accuracy',acc])
    f_writer.writerow(['score', score])
    f.close()

    if exp != 'total':
        shutil.copyfile(f'{save_dir}/test_analytics_{exp}.csv', opt.result/'graphs'/f'test_analytics_{exp}.csv')

    # f_writer.writerow([])

    # cm = confusion_matrix(y_true=slide_label[:,0], y_pred=slide_label[:,1], labels=np.unique(slide_label[:,0]).tolist())
    # print(f'{key}\'s slide result\n',cm)
    
    # f_writer.writerow(['Slide']+[f'pred:{i}' for i in range(len(cm))])
    # total = 0
    # correct = 0
    # recall_list = []
    # for i in range(len(cm)):
    #     row_total = 0
    #     for j in range(len(cm)):
    #         total += cm[i][j]
    #         row_total += cm[i][j]
    #     recall_list.append(cm[i][i]/row_total)
    #     correct += cm[i][i]
    #     f_writer.writerow([f'true:{i}']+cm[i].tolist())
    # acc = correct/total
    # f_writer.writerow(['recall']+recall_list)
    # f_writer.writerow(['total',total])
    # f_writer.writerow(['accuracy',acc])
    # f.close()

    print(classification_report(y_true=test_result[:,1], y_pred=test_result[:,2]))

def make_histgram(opt, total_data, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    print(total_data)
    label_split_data = [
        [SlideName_dict[data[0]] for data in total_data if data[1]=='0' and data[2]=='0'],
        [SlideName_dict[data[0]] for data in total_data if data[1]=='0' and data[2]=='1'],
        [SlideName_dict[data[0]] for data in total_data if data[1]=='1' and data[2]=='1'],
        [SlideName_dict[data[0]] for data in total_data if data[1]=='1' and data[2]=='0'],
    ]
    
    subtype_list = [
        np.unique(label_split_data[0]+label_split_data[1]),
        np.unique(label_split_data[2]+label_split_data[3]),
    ]
    
    count_list = [
        np.array([label_split_data[0].count(subtype) for subtype in subtype_list[0]]),
        np.array([label_split_data[1].count(subtype) for subtype in subtype_list[0]]),
        np.array([label_split_data[2].count(subtype) for subtype in subtype_list[1]]),
        np.array([label_split_data[3].count(subtype) for subtype in subtype_list[1]]),
    ]

    sort_list = [
        np.argsort(count_list[0])[::-1],
        np.argsort(count_list[2])[::-1],
    ]    
    
    for count in count_list:
        print(count)

    plt.rcParams['figure.subplot.bottom'] = 0.35
    for i in range(2):
        plt.figure()
        fig = plt.bar(subtype_list[i][sort_list[i]], count_list[i*2][sort_list[i]], align='edge', width=-0.3, label='correct', color='dodgerblue')
        fig = plt.bar(subtype_list[i][sort_list[i]], count_list[i*2+1][sort_list[i]], align='edge', width=0.3, label='incorrect', color='orange')
        plt.xticks(rotation=90)
        plt.legend()
        plt.savefig(save_dir / f'hist_{i}.png')


def make_log_graphs(opt):
    exp_list = list(opt.result.parent.glob(r'train*'))
    all_log_list = {}
    
    for exp in exp_list:
        log = load_logfile(exp)
        if log is not None:
            key = exp.name.split('-')[0]
            
            if key not in all_log_list:
                all_log_list[key] = [[exp, np.min(log[:, 2])]]
            else:
                all_log_list[key].append([exp, np.min(log[:, 2])])
            
            save_dir = exp / 'graphs'
            save_dir.mkdir(exist_ok=True)
            
            save_split_graph(opt, log, exp.name, save_dir)

    print(all_log_list)

    best_log_list = {}
    best_exp_list = []
    max_epoch = 30
    for key in all_log_list:
        exps = np.asarray(all_log_list[key])
        exp = exps[np.argmin(exps[:,1])][0]
        log = load_logfile(exp)
        best_log_list[exp.name] = log
        best_exp_list.append(exp)
        
        max_epoch = max_epoch if max_epoch < log.shape[0] else log.shape[0]
        
    best_log_list['average'] = cal_log_ave(best_log_list, max_epoch)
    save_split_graph(opt, best_log_list['average'], 'average', opt.result/'graphs')
    save_category_graph(opt, best_log_list, max_epoch)

    return best_exp_list

def make_test_analysis(opt, best_exp_list=None):
    exp_list = list(opt.result.parent.glob(r'train*'))
    total_result = []
    
    for exp in exp_list:
        test_result = load_testresult(exp/'test_result')
            
        if test_result is not None:
            save_dir = exp / 'graphs'
            save_dir.mkdir(exist_ok=True)

            save_test_cm(opt, test_result, exp.name, save_dir)

            if best_exp_list and exp in best_exp_list:
                total_result.append(test_result)

    if len(total_result) == len(best_exp_list):
        total_result = np.concatenate(total_result)
        print(total_result.shape)
        save_test_cm(opt, total_result, 'total', opt.result/'graphs')
        make_histgram(opt, total_result, opt.result/'graphs')
    

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--data', default='add', choices=['', 'add'])
    parser.add_argument('--mil_mode', default='yolo', choices=['amil', 'yolo'], help='flag to use normal AMIL')
    parser.add_argument('--yolo_ver', default=None, help='choose weight version')
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

SlideName_dict = get_slideID_name()
if __name__ == '__main__':
    opt = parse_args()
    
    opt.dir_name = make_dirname(opt)
    opt.file_name = make_filename(opt)
    opt.result = Path(f'./runs/{opt.dir_name}/total_result')
    opt.result.mkdir(exist_ok=True)
    print(colorstr(opt.dir_name))
    
    print(colorstr('make log graphs'))
    best_exp_list = make_log_graphs(opt)
    print(best_exp_list)

    print(colorstr('make test cross matrix'))
    make_test_analysis(opt, best_exp_list)

    best_exp_list = '\n'.join([str(best.name) for best in best_exp_list])
    print(best_exp_list)
    with open(f'./runs/{opt.dir_name}/best_exps.txt', 'w') as f:
        f.write(best_exp_list)
    