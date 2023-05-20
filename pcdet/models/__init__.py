# coding = utf-8
# -*- coding:utf-8 -*-

from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def build_network(model_cfg, num_class, dataset):
    """
    调用detectors中__init__.py中的build_detector构建网络模型
    """
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    """
    跳过元信息和标定数据，同时根据数据类型转换数据类型，再放到gpu上
    """
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray) or key in ['name']:
            continue
        if key in ['use_lead_xyz', 'label']:
            batch_dict[key] = torch.from_numpy(val).cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'batch_dict', 'disp_dict'])

    def model_func(model, batch_dict):

        load_data_to_gpu(batch_dict)

        batch_dict, disp_dict = model(batch_dict)

        loss = batch_dict['loss']

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, batch_dict, disp_dict)

    return model_func
