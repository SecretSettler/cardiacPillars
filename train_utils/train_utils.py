import copy
import glob
import os

import numpy as np
import torch.optim as optim
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.models import load_data_to_gpu
import torch.nn as nn
from torchvision import models
from pcdet.utils import loss_utils


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    # 确定每个epoch的iter数量，并且将train_loader转化为迭代器，next函数用来从迭代器中一个个获取数据
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    # 表示进程序号，用于进程间通讯，表征进程优先级，rank = 0的主机为master节点
    # world size指进程总数，我们使用的卡数
    # local_rank进程内，GPU编号，非显式参数，由torch.distributed.launch内部指定
    # 比方说，rank = 3，local_rank = 0表示第3个进程内的第1块GPU
    # 初始化tqdm（只在master节点进行）
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    running_correct = 0
    running_loss = 0
    for cur_it in range(total_it_each_epoch):
        try:
            # 获取一个batch数据
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)
        # 获取当前学习率
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
        model.train()

        optimizer.zero_grad()

        loss, batch_dict, disp_dict = model_func(model, batch)

        data = batch_dict['cls_output']

        target = batch_dict['label']

        _, pred = torch.max(data, 1)

        loss.backward()

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)

        optimizer.step()

        accumulated_iter += 1

        tb_dict = {
            'cls_loss': loss.item()
        }

        print(loss.item())

        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        running_correct += torch.sum(pred == target.data).double()
        running_loss += loss.item() * data.size(0)

        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

    epoch_acc = running_correct / len(train_loader.dataset)
    epoch_loss = running_loss / len(train_loader.dataset)
    print('epoch accuracy is {}'.format(epoch_acc))
    print('epoch loss is {}'.format(epoch_loss))

    if rank == 0:
        pbar.close()
    return accumulated_iter, epoch_acc


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    """
    模型训练
    Args:
        model:模型
        optimizer: 优化器
        train_loader: Dataloader
        model_func: 模型函数装饰器，其在model的__init__.py中:主要是将数据放到模型上在返回loss
        lr_scheduler: 学习率调度器
        optim_cfg: 优化器配置
        start_epoch: 起始epoch
        total_epochs: 总共epoch数量
        start_iter: 起始迭代数
        rank:进程号
        tb_log: tensorboad的log
        ckpt_save_dir: checkpoint存储文件夹路径
        train_sampler: 训练数据采样器 DistributedSampler
        lr_warmup_scheduler: 学习率热身调度器
        ckpt_save_interval: checkpoint存储间隔，默认为1
        max_ckpt_save_num: 最大的checkpoint存储数量，默认为50
        merge_all_iters_to_one_epoch: 是否将所有iter合并为一个epoch
    """
    file_name = '/home/s2020153/cardiac/cardiac_pillars/output/densenet/model_save.pth'
    accumulated_iter = start_iter  # 累计已迭代次数
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)  # 完成一个epoch所需的iter次数：928
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)  # 将Dataloder转换为迭代格式
        best_acc = 0
        for cur_epoch in tbar:  # 到这里会显示进度条，cur_epoch会自动获取tbar中的值
            # if train_sampler is not None:
            #     train_sampler.set_epoch(cur_epoch)

            # 1.确定学习率调度器，是否需要进行学习率预热
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            # 2.训练一个epoch
            accumulated_iter, epoch_acc = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # 3.保存训练模型
            # 更新epoch
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                # 查看ckpt文件夹中的ckpt文件
                # glob会包含全部路径，os.listdir只会包含其中的文件名
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                # 按照时间排序
                # os.path.getmtime用于获取指定路径文件最后的修改时间
                ckpt_list.sort(key=os.path.getmtime)

                # 如果权重文件的数量大于最大数量的文件，则删除最前面的文件
                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                # 定义权重文件保存路径
                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                # 保存权重:先获取state_dict,在保存
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': best_model,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, file_name)


def model_state_to_cpu(model_state):
    # 只有分布式训练需要
    model_state_cpu = type(model_state)()  # ordered dict,字典类型
    # 将model_state的值转到cpu上
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    # 1.获取优化器state_dict
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())  # 将model_state放到cpu上
        else:
            # 2.获取模型state_dict
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        # 获取版本信息
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    # 3.返回epoch,it,model_state,optim_state和version信息
    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)
    # 保存ckpt文件
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
