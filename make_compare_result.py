import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## make_log_Graphs.pyで生成されるlog_averageとtest_analyticsをまとめるプログラム

DATA_PATH = './graphs/compare'

def load_recall_data(path):
    csv_data = open(path)
    reader = csv.reader(csv_data)
    for row in reader:
        if row[0] == 'recall':
            leaf0 = '{:.3f}'.format(float(row[1]))
            leaf1 = '{:.3f}'.format(float(row[2]))
            return [leaf0, leaf1]

def make_all_recall_list(args):
    recall_all = []
    for mode in model_name_list:
        recall_all += ['', mode, '']
    recall_all = np.array([recall_all+['']])
    
    for i, data in enumerate(data_list):
        data_name = data_name_list[i]

        recall_model = None
        for j, model in enumerate(model_list):
            model_name = model_name_list[j]
            recall_list = [['Hodgkin', 'other']]

            for loss in loss_list:
                path = f'graphs/{data}{model}new_tree{loss}/depth-{args.depth}_leaf-{args.leaf}/{args.mag}_{args.lr}/test_analytics.csv'

                if not os.path.exists(path):
                    recall_list.append(['', ''])
                else:
                    recall_list.append(load_recall_data(path))
            
            if recall_model is None:
                space = [[data_name]]+[[l] for l in loss_name_list]
                f_score = [['f_score']]+[['{:.3f}'.format(2*float(recall[0])*float(recall[1])/(float(recall[0])+float(recall[1])))] if recall[0]!='' else [''] for recall in recall_list[1:]]
                recall_model = np.concatenate((space, recall_list, f_score), axis=1)
            else:
                space = [[''] for a in range(recall_model.shape[0])]
                f_score = [['f_score']]+[['{:.3f}'.format(2*float(recall[0])*float(recall[1])/(float(recall[0])+float(recall[1])))] if recall[0]!='' else [''] for recall in recall_list[1:]]
                recall_model = np.concatenate((recall_model, recall_list, f_score), axis=1)

        if recall_all.shape[0]==1:
            recall_all = np.concatenate((recall_all, recall_model), axis=0)
        else:
            space = [['' for a in range(recall_model.shape[1])]]
            recall_all = np.concatenate((recall_all, space, recall_model), axis=0)

    print(recall_all)
    np.savetxt(f'./graphs/compare/depth{args.depth}/{args.mag}_{args.lr}/all_recall_list.csv', recall_all, delimiter=',', fmt='%s')

def make_all_log_graph(args):
    loss_all = {}
    graph_title_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']

    for i, loss in enumerate(loss_list):
        graph_label_list = []
        log_data_list = [] # axis=2 => [train_loss, train_acc, valid_loss, valid_acc]

        for j, data in enumerate(data_list):
            for k, model in enumerate(model_list):
                path = f'graphs/{data}{model}new_tree{loss}/depth-{args.depth}_leaf-{args.leaf}/{args.mag}_{args.lr}/average_log.csv'
                
                if os.path.exists(path):
                    if data_name_list[j] != '':
                        graph_label = f'{data_name_list[j]}_{model_name_list[k]}'
                    else:
                        graph_label = f'{model_name_list[k]}'
                    graph_label_list.append(graph_label)

                    log = np.loadtxt(path, delimiter=',', dtype='str')
                    log = log[1:21,1:].astype(np.float32)
                    if log.shape[0] == 20:
                        log_data_list.append(log)
        if len(log_data_list)>0:
            log_data_list = np.stack(log_data_list, axis=0)
            loss_all[loss_name_list[i]] = [log_data_list, graph_label_list]
    if len(loss_all)>0:
        fig = plt.figure()
        color = {
            'vgg16':'royalblue',
            'vgg16_fc':'darkorange',
            'vgg11':'green',
            'add_vgg16':'red',
            'add_vgg11':'purple'
        }
        weight = {
            'vgg16': 252/21.+252/231.,
            'vgg16_fc': 252/21.+252/231.,
            'vgg11': 252/21.+252/231.,
            'add_vgg16': 315/84.+315/231.,
            'add_vgg11': 315/84.+315/231.
        }
        # print(weight)
        for i,l in enumerate(loss_all):
            data = loss_all[l][0]
            data = data[:,:,2]
            x = np.array(range(data.shape[1]))+1
            label = loss_all[l][1]
            
            for j,d in enumerate(data):
                if l == 'CE-invarse':
                    d = d/weight[label[j]]
                plt.plot(x, d, label=label[j], color=color[label[j]])
            plt.legend(fontsize=14)
            plt.title(l,fontsize=16)
            plt.xlabel('epoch')
            plt.gca().set_ylim(top=1.2)
            plt.gca().set_ylim(bottom=-0.1)
            plt.grid()
            plt.savefig(f'./graphs/compare/depth{args.depth}/{args.mag}_{args.lr}/valid_loss_{l}.png')
            plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default='3', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()

    place = 'depth-1_leaf-01'

    model_list = ['', 'fc_', 'vgg11_']
    model_name_list = ['vgg16', 'vgg16_fc', 'vgg11']
    loss_list = ['', '_myinvarse', '_LDAM-0.1', '_LDAM-0.3', '_LDAM-0.5', '_focal-1.0', '_focal-2.0']
    loss_name_list = ['Cross_Entropy', 'CE-invarse', 'LDAM-0.1', 'LDAM-0.3', 'LDAM-0.5', 'focal-1.0', 'focal-2.0']
    data_list = ['', 'add_reduce_']
    data_name_list = ['', 'add']

    import utils
    utils.makedir(f'./graphs/compare/depth{args.depth}/{args.mag}_{args.lr}/')
    make_all_recall_list(args)
    make_all_log_graph(args)

    # utils.send_email(body=str(args))
                
                