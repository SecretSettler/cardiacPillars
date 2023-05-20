import copy
import numpy as np

from ...utils import common_utils


def random_flip_along_x(points):
    """
    沿着x轴随机翻转
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    # 随机选择是否翻转
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1] # 点云y坐标翻转

    return points


def random_flip_along_y(points):
    """
    沿着y轴随机翻转
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    # 随机旋转是否翻转
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]#  点云x坐标取反

    return points


def global_rotation(points, rot_range):
    # 在均匀分布中随机产生旋转角度
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    # 沿z轴旋转noise_rotation弧度，这里之所以取第0个，是因为rotate_points_along_z对batch进行处理，而这里仅处理单个点云
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]

    return points


def global_scaling(points, scale_range):
    """
    随机缩放
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-3:
        return points
    # 在缩放范围内随机产生缩放尺度
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    # 将点云和box同时乘以缩放尺度
    points[:, :3] *= noise_scale
    return points
