# coding = utf-8
# -*- coding:utf-8 -*-
import os
import re
import shutil
from collections import defaultdict
from math import *
import random
import fire
import numpy as np
import pathlib
import scipy.spatial.transform as f
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from pcdet.utils import common_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from sklearn.utils import class_weight


def read_one_xyz(filename):
    xyz = []
    with open(filename, 'r') as f:
        content = f.read()
        contact = content.split('\n')
        for line in contact:
            if line == '' or line.isdigit():
                continue
            else:
                atom = line.split()
                xyzitem = [atom[0], atom[1], atom[2]]
                xyz.append(xyzitem)
    return np.array(xyz).astype(float)


def getfiles(dirPath, fileType):
    fileList = []
    files = os.listdir(dirPath)
    files.sort()
    pattern = re.compile('.*/' + fileType)
    for f in files:
        if os.path.isdir(dirPath + '/' + f):
            getfiles(dirPath + '/' + f, fileType)

        elif os.path.isfile(dirPath + '/' + f):
            matches = pattern.match(f)
            if matches is not None:
                fileList.append(dirPath + '/' + matches.group())
        # else:
        #     fileList.append(dirPath + '/invalid')

    return fileList


def points_cloud_processing(translation=True):
    path = '/home/s2020153/cardiac/DATASET/ACDC_ALL_FRAMES_SEG_FINAL'
    input_files = os.listdir(path)
    fType = '.xyz'
    lv = getfiles(path + os.sep + input_files[0], fType)
    myo = getfiles(path + os.sep + input_files[1], fType)
    rv = getfiles(path + os.sep + input_files[2], fType)

    for i in range(len(lv)):
        lv_in = read_one_xyz(lv[i])
        myo_in = read_one_xyz(myo[i])
        rv_in = read_one_xyz(rv[i])

        if translation:
            lv_centre = np.average(lv_in, axis=0)
            myo_centre = np.average(myo_in, axis=0)
            rv_centre = np.average(rv_in, axis=0)

            myo_offset = lv_centre - myo_centre
            myo_in = myo_in + myo_offset
            myo_centre = np.average(myo_in, axis=0)

            mid_point = (rv_centre + lv_centre) / 2
            translation_matrix = -mid_point
            # print(translation_matrix)

            lv_in = lv_in + translation_matrix
            rv_in = rv_in + translation_matrix
            myo_in = myo_in + translation_matrix

            lv_centre2 = np.average(lv_in, axis=0)
            myo_centre2 = np.average(myo_in, axis=0)
            rv_centre2 = np.average(rv_in, axis=0)

            lv_rv_vec = lv_centre2 - rv_centre2
            unit_lv_rv_vec = lv_rv_vec / np.linalg.norm(lv_rv_vec)

            x_vect = np.array([1, 0, 0])
            z_vect = np.array([0, 0, 1])

            dot_product = np.dot(unit_lv_rv_vec, x_vect)
            angle = np.arccos(dot_product)
            # angle = np.rad2deg(angle)
            # print(angle)
            rot_angle = -(np.pi / 4 - angle)
            rot_vect = rot_angle * z_vect
            rotation = f.Rotation.from_rotvec(rot_vect)
            lv_in = rotation.apply(lv_in)
            myo_in = rotation.apply(myo_in)
            rv_in = rotation.apply(rv_in)

        lv_out = np.array([np.append(ls, [int(lv[i][-7:-4]), 1]) for ls in lv_in])
        myo_out = np.array([np.append(ls, [int(myo[i][-7:-4]), 2]) for ls in myo_in])
        rv_out = np.array([np.append(ls, [int(rv[i][-7:-4]), 3]) for ls in rv_in])

        lv_out = random.sample(list(lv_out), 5000)
        myo_out = random.sample(list(myo_out), 10000)
        rv_out = random.sample(list(rv_out), 4000)

        output = np.concatenate((lv_out, myo_out, rv_out))
        out_save_file_tr = '/home/s2020153/cardiac/DATASET/training' + os.sep + lv[i][-18:-4] + '.bin'

        output.tofile(out_save_file_tr)


def points_generator(name):
    root_path = r'/home/s2020153/cardiac/DATASET/' + name
    ls = os.listdir(root_path)
    ls.sort()
    length = len(ls)
    count = 1
    k = 1
    pc = np.fromfile(os.path.join(root_path, ls[0]))
    print(ls[0])
    set_path = pathlib.Path(r'/home/s2020153/cardiac/DATASET/{}_set'.format(name))
    set_path.mkdir(parents=True, exist_ok=True)
    while count < length and k < 10:
        pc2 = np.fromfile(os.path.join(root_path, ls[count]))
        pc = np.concatenate((pc, pc2))
        print(ls[count])
        k += 1
        count += 1
        if k == 10:
            out_save_path = r'/home/s2020153/cardiac/DATASET/{}_set'.format(name) + os.sep + ls[count - 1][:-8] + '.bin'
            pc.tofile(out_save_path)
            if count < length:
                pc = np.fromfile(os.path.join(root_path, ls[count]))
                print('------------------------------------------------------------------')
                print(ls[count])
                count += 1
            k = 1


def points_reader(name, path):
    root_path = r'/home/s2020153/cardiac/DATASET/' + name
    pc = np.fromfile(os.path.join(root_path, path))
    pc = pc.reshape(-1, 5)
    return pc


def get_file_names(name):
    root_path = '/home/s2020153/cardiac/DATASET/' + name
    names = os.listdir(root_path)
    names.sort()
    return names


def get_labels():
    root_path = r'/home/s2020153/cardiac/cardiac_pillars/labels'
    files = os.listdir(root_path)
    files.sort()
    labels = []
    for fi in files:
        contents = []
        with open(root_path + os.sep + fi) as txt:
            for line in txt:
                line = line.strip('\n')
                line = line.rstrip()
                contents.append(line)
            labels.append(contents[2][7:])
    return labels


def get_ED_ES_from_file(name, training=True):
    contents = []
    if training:
        root_path = r'/home/s2020153/cardiac/cardiac_pillars/labels'
    else:
        root_path = r'/home/s2020153/cardiac/cardiac_pillars/test_info'
    with open(root_path + os.sep + name + '.txt') as txt:
        for line in txt:
            line = line.strip('\n')
            line = line.rstrip()
            contents.append(line)
    try:
        return int(contents[0][-1]), int(contents[1][-2:])
    except ValueError:
        return int(contents[0][-1]), int(contents[1][-1])


def set_picked_frames(d, s):
    ls = [d - 1, s - 1]
    if s < 10:
        for i in range(d, s - 1):
            ls.append(i)
        while len(ls) != 10:
            ls.sort()
            ls.append(ls[-1] + 1)
        return [i + 1 for i in ls]
    elif d > 1:
        for i in range(d, s - 1):
            ls.append(i)
        while len(ls) != 10:
            ls.append(ls[0] - 1)
            ls.sort()
        return [i + 1 for i in ls]
    else:
        for i in range(d, s - 1, 2):
            ls.append(i)

        if s % 2 == 0:
            for j in range(s - 2, d, -2):
                ls.append(j)
                if len(ls) == 10:
                    break
        if s % 2 != 0:
            for j in range(s - 3, d, -2):
                ls.append(j)
                if len(ls) == 10:
                    break
        return [i + 1 for i in ls]


def get_label_files_names(training):
    if training:
        root_path = r'/home/s2020153/cardiac/cardiac_pillars/labels'
    else:
        root_path = r'/home/s2020153/cardiac/cardiac_pillars/test_info'
    files = os.listdir(root_path)
    files.sort()
    return [fi[:-4] for fi in files]


def train_val_split(file_name, training=True):
    if training:
        path = r'/home/s2020153/cardiac/DATASET/training'
    else:
        path = r'/home/s2020153/cardiac/DATASET/testing'
    files = get_file_names(file_name)
    names = get_label_files_names(training)

    for i in names:
        ed, es = get_ED_ES_from_file(i, training)
        frames = set_picked_frames(ed, es)
        for j in files:
            if (i in j) and (int(j[-7:-4]) not in frames):
                os.remove(path + os.sep + j)


def get_acdc_features(name):
    path = r'/home/s2020153/cardiac/cardiac_pillars/manual_features/acdc_{}.txt'.format(name)
    contents = []
    with open(path) as file:
        for line in file:
            line = line.strip('\n')
            line = line.rstrip()
            contents.append(line)
    contents = [i.split(' ') for i in contents]
    thick_dict = {}
    for ls in contents:
        thick_dict.update({ls[0]: np.array(ls[1:], dtype=float)})
    return thick_dict


class CardiacDataset(Dataset):
    def __init__(self, train_or_val, transform=None, target_transform=None, loader=None, root_path=None, config=None,
                 training=True):
        super(CardiacDataset, self).__init__()
        if config is None:
            return

        # acdc_thickness = get_acdc_features('thickness')
        # acdc_volume = get_acdc_features('volume')
        class_names = config.CLASS_NAMES
        origin_labels = get_labels()
        points = get_file_names(train_or_val)
        if training:
            labels = [class_names.index(origin_labels[int(i[-7:-4]) - 1]) for i in points
                      if int(i[-7:-4]) <= 100]
            labels.extend([class_names.index(origin_labels[int(j[-7:-4]) - 51]) for j in points
                           if int(j[-7:-4]) >= 151])
            labels = np.array(labels)
        else:
            labels = []
        point_cloud_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
        voxel_size = np.array([0.16, 0.16, 4])
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size

        # self.acdc_thickness = acdc_thickness
        # self.acdc_volume = acdc_volume
        self.train_or_val = train_or_val
        self.config = config
        self.root_path = root_path
        self.class_names = class_names
        self.training = training

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.points = points
        self.labels = labels
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.point_feature_encoder = PointFeatureEncoder(self.config.DATA_CONFIG.POINT_FEATURE_ENCODING,
                                                         point_cloud_range=self.point_cloud_range)

        self.data_augmentor = DataAugmentor(self.root_path, self.config.DATA_CONFIG.DATA_AUGMENTOR,
                                            self.class_names, logger=None) if self.training else None

        self.data_processor = DataProcessor(
            self.config.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

    def get_labels(self):
        return self.labels

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                label: optional, (1), string
                ...

        Returns:
            data_dict:
                points: (N, 3 + C_in)
                gt_names: optional, (1), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                # voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                # voxel_coords: optional (num_voxels, 3)
                # voxel_num_points: optional (num_voxels)
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)  # （B, N, 4)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def __getitem__(self, item):
        points_name = self.points[item]
        if self.training:
            label = np.array(self.labels[item])
        else:
            label = []
        # thick = self.acdc_thickness[points_name[:-4]]
        # volume = self.acdc_volume[points_name[:-4]]
        pc = self.loader(self.train_or_val, points_name)

        if self.transform is not None:
            pc = self.transform(pc)

        input_dict = {
            'name': points_name[:-4],
            'points': pc,
            'label': label,
            # 'thickness': thick,
            # 'volume': volume
        }

        data_dict = self.prepare_data(input_dict)

        return data_dict

    def __len__(self):
        return len(self.points)


if __name__ == '__main__':
    fire.Fire()
