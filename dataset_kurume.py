# slide(症例)を訓練用とテスト(valid)用に分割
import os
import csv
import argparse
import numpy as np

from collections import Counter

def read_leafCSV(filepath, label):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}') and row[1] != 'OI_ILPD':
        # if row[1] != 'OI_ILPD':
            # OI_ILPDは別の病気？のための薬の副作用によるがんの発症なので無視する
            file_data.append([row[0], label, row[1]])
        elif row[1] == 'OI_ILPD':
            # print(f'slideID{row[0]}はOI_ILPDです')
            continue
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data

def reduce_data(data, args):
    flag_list = {}
    # ファイルの修飾子変更前の参照先
    # csv_data = open('../KurumeTree/add_data/add_flag_list.csv')
    csv_data = open(f'../KurumeTree/dataset/{args.data}/add_flag_list.csv')
    reader = csv.reader(csv_data)
    for row in reader:
        flag_list[row[0]]=int(row[1])
    csv_data.close()
    
    reduced_data = []
    for d in data:
        if flag_list[d[0]] == 0:
           reduced_data.append([d[0], d[1], d[2]])

    return reduced_data 

# 決定木の葉に沿って識別する時
def load_leaf(args, rank=0):
    # ファイルの修飾子変更前の参照先
    # if args.data:
    #     args.data = f'{args.data}_'
    # if args.classify_mode == 'leaf':
    #     dir_path = f'../KurumeTree/{args.data}result/{args.name}/unu_depth{args.depth}/leafs_data'
    # if args.classify_mode == 'new_tree':
    #     dir_path = f'../KurumeTree/{args.data}result_teacher/FDC/{args.name}/unu_depth{args.depth}/leafs_data'
    dir_path = f'../KurumeTree/results/{args.classify_mode}/{args.data}/{args.name}/unu_depth{args.depth}/leafs_data'

    if args.leaf != None:
        if int(args.leaf[0])%2!=0 or int(args.leaf[1])-int(args.leaf[0])!=1:
            print('隣接した葉ではありません')
            exit()

        dataset = []
        for num in args.leaf:
            leaf_data = read_leafCSV(f'{dir_path}/leaf_{num}.csv', int(num)%2)
            dataset.append(leaf_data)
        min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
        max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
        ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
        print(f'{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}') if rank==0 else None
        if args.reduce:
            dataset[max_leaf] = reduce_data(dataset[max_leaf], args)
            min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
            max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
            ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
            print(f'many data reduced \n{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}') if rank==0 else None

        train_dataset = []
        valid_dataset = []
        for num in args.leaf:
            for idx, slide in enumerate(dataset[int(num)]):
                if str((idx%5)+1) in args.train:
                    train_dataset.append(slide+[0])

                    if args.augmentation and int(num) == min_leaf:
                        for aug_i in range(np.min([7, ratio-1])):
                            train_dataset.append(slide+[aug_i+1])
                
                if str((idx%5)+1) in args.valid:
                    valid_dataset.append(slide+[0])
        
        return train_dataset, valid_dataset, 2

    else: 
        print('Please input leaf number')
        exit()


def read_CSV(filepath):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    name_data = {'DLBCL':0, 'FL':1, 'Reactive':2, 'CHL':3}
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}'):
            if row[1] in name_data:
                file_data.append([row[0], name_data[row[1]]])
            else:
                file_data.append([row[0], 4])
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data

# subtype別で識別する時
def load_svs(args, rank):
    # ファイルの修飾子変更前の参照先
    # file_name = f'./{args.data}data/Data_{args.name}Name.csv'
    file_name = f'../KurumeTree/dataset/{args.data}/Data_{args.name}Name.csv'
    svs_data = read_CSV(file_name)
    
    train_dataset = []
    valid_dataset = []
    for idx, slide in enumerate(svs_data):
        if str((idx%5)+1) in args.train:
            train_dataset.append(slide)
            
        if str((idx%5)+1) in args.valid:
            valid_dataset.append(slide)

    return train_dataset, valid_dataset, 5

def count_subtype(opt):
    # ファイルの修飾子変更前の参照先
    # if opt.data:
    #     opt.data = f'{opt.data}_'
    # if opt.classify_mode == 'leaf':
    #     dir_path = f'../KurumeTree/{opt.data}result/{opt.name}/unu_depth{opt.depth}/leafs_data'
    # if opt.classify_mode == 'new_tree':
    #     dir_path = f'../KurumeTree/{opt.data}result_teacher/FDC/{opt.name}/unu_depth{opt.depth}/leafs_data'
    dir_path = f'../KurumeTree/results/{opt.classify_mode}/{opt.data}/{opt.name}/unu_depth{opt.depth}/leafs_data'

    if opt.leaf:
        dataset = []
        for num in opt.leaf:
            leaf_data = read_leafCSV(f'{dir_path}/leaf_{num}.csv', int(num)%len(opt.leaf))
            dataset.append(np.array(leaf_data))
        print(dataset)
            
        min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
        max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
        ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
        print(f'{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}')
        
        if opt.reduce:
            dataset[max_leaf] = np.array(reduce_data(dataset[max_leaf], opt))
            min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
            max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
            ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
            print(f'many data reduced \n{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}')
            
        for data in dataset:
            count = Counter(data[:,2])
            print(len(data), count)
        
        data = np.concatenate(dataset)
        count = Counter(data[:,2])
        print(len(data), len(count), count)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default='1', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    parser.add_argument('--data', default='2nd', choices=['1st', '2nd', '3rd'])
    parser.add_argument('--reduce', action='store_true')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='kurume_tree', choices=['normal_tree', 'kurume_tree', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    opt = parser.parse_args()
    
    if opt.data == '2nd' or opt.data == '3rd':
        opt.reduce = True

    count_subtype(opt)



