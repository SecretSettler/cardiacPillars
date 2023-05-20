import pickle
import time

import numpy as np
import torch
import tqdm
import torch.nn as nn

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    """
    统计信息
    Args:
        cfg:配置文件
        ret_dict:结果字典
        metric:度量字典
        disp_dict:展示字典
    """
    # [0.3,0.5,0.7]根据不同的阈值进行累加
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0) # 真值框的数量
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0] # 0.3
    # 最小阈值的展示字典统计
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    """
    模型评估
    Args:
        cfg:配置文件
        model:模型
        dataloader: 数据加载器
        epoch_id：epoch的id
        logger:日志记录器
        dist_test:分布式测试
        save_to_file: 保存到文件
        result_dir: 结果文件夹:OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_80/val/default
    Returns:
        ret_dict: 结果字典
    """
    # 构造文件夹
    result_dir.mkdir(parents=True, exist_ok=True)
    # /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_80/val/default/final_result/data
    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    running_correct = 0
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    start_time = time.time()

    result_dict = {}

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, disp_dict = model(batch_dict)
            data = pred_dicts['cls_output']
            names = pred_dicts['name']
            _, preds = torch.max(data, 1)
        for j in range(len(preds)):
            if names[j] not in result_dict.keys():
                result_dict.update({names[j]: [preds[j]]})
            else:
                result_dict[names[j]].append(preds[j])
        # running_correct += torch.sum(preds == target.data).double()
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    # epoch_acc = running_correct / len(dataset)

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    result_str = 'The evaluation of this epoch is over'

    logger.info(result_str)

    with open(result_dir / 'result.txt', 'w') as f:
        for key, vals in result_dict.items():
            f.write(key)
            f.write(': ')
            f.write(str(vals))
            f.write('\n')

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return result_dict


if __name__ == '__main__':
    pass
