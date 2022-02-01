from numpy.core.defchararray import index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision.models.vgg import vgg11

from yolo import Model as yolo_extractor

class CEInvarse(nn.Module):
    def __init__(self, rank, class_num_list, reduction_flag=False):
        # 不均衡データに対してlossの重みを調整
        super(CEInvarse, self).__init__()
        n_RIGHT = class_num_list[0]
        n_LEFT = class_num_list[1]
        weight = torch.tensor([1/(n_RIGHT/(n_RIGHT+n_LEFT)), 1/(n_LEFT/(n_RIGHT+n_LEFT))])
        self.weight = weight.to(rank)
        self.class_num_list = class_num_list
        self.reduction = torch.sum(self.weight)
        self.reduction_flag = reduction_flag
        
    def forward(self, x, target):
        index = F.one_hot(target, len(self.class_num_list)).type(torch.uint8)
        weight = torch.sum(self.weight * index)
        loss = weight*F.cross_entropy(x, target)
        # loss1 = self.weight[target[0]]*F.cross_entropy(x, target)

        if self.reduction_flag:
            return loss / self.reduction
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, rank, class_num_list, gamma=1.0, weight_flag=False, reduction_flag=True):
        # ほとんどが識別に関係のない背景であることを考慮したLoss
        # 大きく間違っている場合は通常のCE-Lossと同じだが、
        # ほぼ正解している場合に対してはLossとして計上しない仕組み
        super(FocalLoss, self).__init__()
        self.rank = rank
        self.class_num_list = class_num_list
        self.gamma = gamma

        n_RIGHT = class_num_list[0]
        n_LEFT = class_num_list[1]
        weight = torch.tensor([1/(n_RIGHT/(n_RIGHT+n_LEFT)), 1/(n_LEFT/(n_RIGHT+n_LEFT))])
        self.weight_flag = weight_flag
        self.weight = weight.to(rank)
        self.reduction = torch.sum(self.weight)
        self.reduction_flag = reduction_flag
        
    def forward(self, x, target):
        index = F.one_hot(target, len(self.class_num_list)).type(torch.uint8)
        x_softmax = F.softmax(x, dim=1).to(self.rank)
        loss = -1. * index * torch.log(x_softmax) # cross entropy
        loss = loss * (1 - x_softmax) ** self.gamma # focal loss

        if self.weight_flag:
            if self.reduction_flag:
                loss = self.weight * loss / self.reduction
            else:
                loss = self.weight * loss

        # weight = torch.pow(1-x_softmax, self.gamma).to(self.rank)
        # print(x,x_softmax)
        # print(target,weight)
        # print(F.cross_entropy(x, target),loss)
        # print(weight[0,target[0]]*F.cross_entropy(x, target))
        # exit()
        # loss_handmade =  weight[0,target[0]]*F.cross_entropy(x, target)
        return loss.sum()


class LDAMLoss(nn.Module):
    def __init__(self, rank, class_num_list, Constant=0.5, s=1):
        super(LDAMLoss, self).__init__()
        m_list = 1.0/np.sqrt(np.sqrt(class_num_list))
        m_list = m_list * Constant
        m_list = torch.cuda.FloatTensor(m_list)
        self.class_num_list = class_num_list
        self.m_list = m_list[None,:].to(rank)
        self.s = s
    
    def forward(self, x, target):
        index = F.one_hot(target, len(self.class_num_list)).type(torch.uint8)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list, index_float.transpose(0,1))
        batch_m = batch_m.view(-1,1)
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target)

class feature_extractor(nn.Module):
    def __init__(self, model_select='vgg16'):
        super(feature_extractor, self).__init__()
        if model_select == 'vgg16' or not model_select:
            feature_ex = models.vgg16(pretrained=True)
        elif model_select == 'vgg11':
            feature_ex = models.vgg11(pretrained=True)
        self.feature_ex = nn.Sequential(*list(feature_ex.children())[:-1])
    
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.view(feature.size(0), -1)
        return feature

class class_predictor(nn.Module):
    def __init__(self, label_count, dropout=False, p=0.5, detect_obj=None, yolo_stage=1):
        super(class_predictor, self).__init__()
        self.detect_obj = detect_obj*yolo_stage if detect_obj else 3087*yolo_stage

        if dropout:
            # 次元圧縮
            self.feature_extractor_2 = nn.Sequential(
                nn.Linear(in_features=25088, out_features=2048),
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.Linear(in_features=2048, out_features=512),
                nn.ReLU()
            )
            # attention mechanism
            self.attention = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(512 + self.detect_obj, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
            # class classifier
            self.classifier = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(512, label_count),
            )
        else:
            # 次元圧縮
            self.feature_extractor_2 = nn.Sequential(
                nn.Linear(in_features=25088, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=512),
                nn.ReLU()
            )
            # attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(512 + self.detect_obj, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
            # class classifier
            self.classifier = nn.Sequential(
                nn.Linear(512, label_count),
            )

    def forward(self, input1, input2):
        x = input1.squeeze(0)
        H = self.feature_extractor_2(x)
        
        # 次元数3087は検出がない時にスパースな割合が高くなるので減らす
        input2, _ = torch.sort(input2, descending=True)
        input2 = input2[:,:self.detect_obj]

        # x_mix : x_feature + confidence
        x_mix = torch.cat((H, input2),1)
        A = self.attention(x_mix)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H)  # KxL
        class_prob = self.classifier(M)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))

        return class_prob, class_hat, A

class MILYOLO(nn.Module):
    def __init__(self, feature_ex, class_predictor, yolo_extractor_list):
        super(MILYOLO, self).__init__()
        self.feature_extractor = feature_ex
        self.yolo_extractor_list = nn.ModuleList(yolo_extractor_list)
        self.class_predictor = class_predictor

    def forward(self, input):
        x = input.squeeze(0)
        # 特徴抽出
        features = self.feature_extractor(x)

        yolo_conf_list, yolo_out_list = [], []
        for yolo_extractor in self.yolo_extractor_list:
            yolo_conf, yolo_out = yolo_extractor(x)
            yolo_conf_list.append(yolo_conf)
            yolo_out_list.append(yolo_out)
        yolo_conf = torch.cat(yolo_conf_list, 1)
        yolo_out = torch.cat(yolo_out_list, 1)
        # class分類
        class_prob, class_hat, A = self.class_predictor(features, yolo_conf)
        # 訓練時(mode='train')DANN適用
        return class_prob, class_hat, A, yolo_out

if __name__ == '__main__':
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    class_predictor = class_predictor(2,dropout=True)
    yolo_extractor = yolo_extractor()
    ckpt = torch.load('./YoloWeights/depth1.pt')  # load checkpoint
    csd = ckpt['model'].state_dict()
    yolo_extractor.load_state_dict(csd, strict=False)  # load
    
    # model構築
    model = MILYOLO(feature_extractor, class_predictor, yolo_extractor)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if 'yolo_extractor' in k:
            # print(f'freezing {k}')
            v.requires_grad = False
    # print([m for m in model.named_parameters()])
    model.eval()
    print(model)
    
    # Profile
    img = torch.rand(1, 100 if torch.cuda.is_available() else 1, 3, 224, 224)
    cls_p, cls_h, A, _ = model(img)
    print(cls_p, cls_h, A.shape)