import torch
import dataloader
import model_io
from options import BaseOptions
import numpy as np
import cv2
import os
import time
import torch.nn as nn
from loss import build_loss
import pickle

class SegLoss(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average

    def build_loss(self, mode='focal'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
             return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()


        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')

        loss = criterion(logit, target.long())

        # if self.batch_average:
        #     loss /= n
        return loss


if __name__=='__main__':
    opt = BaseOptions().parse()
    opt.isTrain = True
    # load data
    trainset = dataloader.SegDataSet(opt.train_data_dir, opt.train_batch_size, is_train=opt.isTrain)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opt.train_batch_size,
                                              shuffle=True,
                                              num_workers=opt.train_data_num_thread,
                                              drop_last=True)

    # network init
    gru = torch.nn.GRU(opt.train_batch_size, 27, opt.layers, batch_first=True)
    Linear = torch.nn.Linear(opt.hidden, 1)
    criterion = build_loss(mode='ce')
    optimizer =torch.optim.Adam([{'params': gru.parameters()},
                                     {'params': Linear.parameters()}], opt.lr)
    model = gru()
    model_io.load_model_by_path(opt.model_path, model)
    device = torch.device(opt.device)
    model.to(device)
    if not opt.isTrain: model.eval()

    last_time = time.time()
    loss_sum = 0
    loss_cnt = 0
    loss_x = 0
    print('start training...')
    for epoch in range(100):
        for idx, batch in enumerate(trainloader):
            input_data, target = batch["train_data"], batch["train_label"]
            input_data = torch.cat(input_data, dim=0)

            output, _ = gru(input_data.unsqueeze(0))
            out_last = output[:, -1, :]
            pred = Linear(out_last).squeeze(0)
            # loss = self.criteria(pred.item(), target.tolist()[0])
            loss = criterion(pred, target)
            loss = torch.FloatTensor(loss, 1)
            loss.backward()
            opt.step()
            loss_sum += loss
            if (idx) % 5 == 0:
                time_diff = time.time() - last_time
                use_visdom = False

                train_speed = (4 * 5) / time_diff
                loss_cnt += 1
                print("===>Speed:{:.2f} iters/s Epoch[{}]({}/{}): loss: {:.6f}".format(train_speed, epoch, idx,
                                                                                       len(trainloader), loss))
                last_time = time.time()

            if (idx + 1) % 5 == 0:
                if loss_cnt != 0:
                    if use_visdom:
                        opt.vis.line(X=torch.tensor([loss_x], dtype=torch.float32),
                                 Y=torch.tensor([loss_sum / loss_cnt], dtype=torch.float32),
                                 win='avg_loss_l1_ssim',
                                 update='append' if loss_x > 0 else None, opts=dict(title='avg_loss_l1_ssim'))
                    loss_cnt = 0
                    loss_sum = 0
                    loss_x += 1

                model_io.save_model_by_path(opt.model_path,model.module if isinstance(model, torch.nn.DataParallel) else model)

