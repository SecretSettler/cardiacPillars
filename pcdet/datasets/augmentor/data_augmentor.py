from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):

        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):

                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue

            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        points = data_dict['points']

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']

            points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                points,
            )

        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE'] # [-0.78539816, 0.78539816]

        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        points = augmentor_utils.global_rotation(
            data_dict['points'], rot_range=rot_range
        )

        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """
        随机缩放
        """
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        # 调用augmentor_utils中的函数缩放box和点云
        points = augmentor_utils.global_scaling(
            data_dict['points'], config['WORLD_SCALE_RANGE'] # [0.95, 1.05]
        )

        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 遍历增强队列，逐个增强器做数据增强
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
