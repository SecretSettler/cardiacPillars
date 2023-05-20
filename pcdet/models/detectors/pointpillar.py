import copy

import numpy as np
import torch
from torch import nn
from torch.nn import Softmax
from .detector3d_template import Detector3DTemplate


def log_softmax(x):
    return x - torch.logsumexp(x, dim=1, keepdim=True)


def CrossEntropyLoss(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]
    return - torch.sum(outputs) / num_examples


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # demo.py中调用的是models中__init__.py中的build_network(),返回的是该网络的类
        # 这里调用的是Detector3DTemplate中的build_networks(),
        # 差一个s，这里返回的是各个模块的列表
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # Detector3DTemplate构造好所有模块
        # 这里根据模型配置文件生成的配置列表逐个调用
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # 如果在训练模式下，则获取loss
        disp_dict = {}
        if self.training:
            outputs = batch_dict['cls_output']
            labels = batch_dict['label']
            loss_func = nn.CrossEntropyLoss(reduction='none')
            ce_loss = loss_func(outputs, labels)
            pt = torch.exp(-ce_loss)
            alpha = 0.25
            gamma = 2.0
            focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
            batch_dict['loss'] = focal_loss
            return batch_dict, disp_dict
        else:
            return batch_dict, disp_dict
