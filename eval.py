# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sys

# スライド単位の事後確率とラベルのリストを返す
def get_slide_prob_label(csv_file):
    pred_corpus = {}
    label_corpus = {}
    slide_id_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row)==6):
                slide_id = row[0]
                prob_list = [float(row[3]), float(row[4]), float(row[5])] # [DLBCLの確率, FLの確率, RLの確率]
                if(slide_id not in pred_corpus):
                    pred_corpus[slide_id] = []
                    label_corpus[slide_id] = int(row[1]) #正解ラベル

                pred_corpus[slide_id].append(prob_list)
                if(slide_id not in slide_id_list):
                    slide_id_list.append(slide_id)
    # slide単位の事後確率計算
    slide_prob = []
    true_label_list = []
    pred_label_list = []
    max_prob = []

    for slide_id in slide_id_list:
        prob_list= pred_corpus[slide_id]
        bag_num = len(prob_list) # Bagの数

        add_DLBCL = 0.0
        add_FL = 0.0
        add_RL = 0.0
        for prob in prob_list:
            add_DLBCL = add_DLBCL + np.log(float(prob[0]))
            add_FL = add_FL + np.log(float(prob[1]))
            add_RL = add_RL + np.log(float(prob[2]))
        DLBCL_prob = np.exp(add_DLBCL / bag_num)
        FL_prob = np.exp(add_FL / bag_num)
        RL_prob = np.exp(add_RL / bag_num)
        slide_prob.append([DLBCL_prob, FL_prob, RL_prob])
        true_label_list.append(label_corpus[slide_id])

        tmp_prob = [DLBCL_prob, FL_prob, RL_prob]
        pred_label_list.append(tmp_prob.index(max(tmp_prob)))
        max_prob.append(max(tmp_prob))

    return slide_id_list, slide_prob, true_label_list, pred_label_list, max_prob

def cal_recall(true_label_list, pred_label_list):
    slide_num = len(true_label_list)
    correct = 0
    DLBCL_num = 0
    n_correct = 0
    n_DLBCL_num = 0
    for i in range(slide_num):
        if(true_label_list[i]==1):
            DLBCL_num += 1
            if(pred_label_list[i]==1):
                correct += 1
        if(true_label_list[i]==0):
            n_DLBCL_num += 1
            if(pred_label_list[i]==0):
                n_correct += 1
    return n_correct/n_DLBCL_num, correct/DLBCL_num

def cal_precision(true_label_list, pred_label_list):
    slide_num = len(true_label_list)
    correct = 0
    DLBCL_num = 0
    n_correct = 0
    n_DLBCL_num = 0
    for i in range(slide_num):
        if(pred_label_list[i]==1):
            DLBCL_num += 1
            if(true_label_list[i]==1):
                correct += 1
        if(pred_label_list[i]==0):
            n_DLBCL_num += 1
            if(true_label_list[i]==0):
                n_correct += 1
    return n_correct/n_DLBCL_num, correct/DLBCL_num

def cal_acc(true_label_list, pred_label_list):
    cor_num = 0
    slide_num = len(true_label_list)
    for i in range(slide_num):
        if(true_label_list[i]==pred_label_list[i]):
            cor_num += 1
    return cor_num / slide_num

def eval(csv_file, output_file):
    slide_id_list, slide_prob, true_label_list, pred_label_list, max_prob = get_slide_prob_label(csv_file)
    correct_slide = np.zeros((len(slide_id_list),4),dtype='object')
    for i in range (len(slide_id_list)):
        correct_slide[i,0] = slide_id_list[i]
        correct_slide[i,1] = str(true_label_list[i])
        correct_slide[i,2] = str(pred_label_list[i])
        correct_slide[i,3] = str(max_prob[i])
    np.savetxt(output_file, correct_slide, delimiter=',',fmt="%s")
    print('acc', cal_acc(true_label_list, pred_label_list))

if __name__ == "__main__":
    args = sys.argv
    mag = args[1]
    train_slide = args[2]
    result_file = f'test_result/test_{mag}_train-{train_slide}.csv'
    output_file = f'test_predict/predict_{mag}_train-{train_slide}.csv'

    eval(result_file, output_file)
