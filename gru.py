import torch
# import os
import numpy as np
import torch.nn as nn
from datetime import datetime
import csv_read
import matplotlib
import matplotlib.pyplot as plt
import time
import model_io
import visdom as vis

class GRU(nn.Module):
    def __init__(self, hidden=10, lr=0.001):
        super().__init__()
        self.features = 75
        self.layers = 2
        self.output = 1
        #self.frames = 28

        self.gru = torch.nn.GRU(self.features, hidden, self.layers, batch_first=True)
        self.Linear = torch.nn.Linear(hidden, 1)
        self.criteria = torch.nn.MSELoss()
        self.opt = torch.optim.Adam([{'params': self.gru.parameters()},
                                     {'params': self.Linear.parameters()}], lr)
        # self.train_data = torch.FloatTensor(csv_read.train_data.astype(np.float64))
        # self.train_label = torch.LongTensor(csv_read.train_label)
        # self.test_data = torch.FloatTensor(csv_read.test_data.astype(np.float64))
        # self.test_label = torch.LongTensor(csv_read.test_label)
        self.train_num = 0
        self.test_num = 0

    def train(self, trainloader, vis=None):

        self.gru.train()
        self.Linear.train()
        self.epochs = []
        self.output_list = []
        self.label_list = []
        self.train_loss=0
        self.epoch_loss=0
        last_time = time.time()
        loss_sum = 0
        loss_cnt = 0
        loss_x = 0
        print('start training...')
        for epoch in range(100):
            for idx, batch in enumerate(trainloader):
                input_data, target = batch["train_data"], batch["train_label"]
                input_data = torch.cat(input_data, dim=0)
                output,_ = self.gru(input_data.unsqueeze(0))
                out_last = output[:, -1, :]
                pred = self.Linear(out_last).squeeze(0)

                loss = self.criteria(pred, target)
                loss=torch.FloatTensor(loss,1)
                loss.backward()
                self.opt.step()
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
                            vis.line(X=torch.tensor([loss_x], dtype=torch.float32),
                                         Y=torch.tensor([loss_sum / loss_cnt], dtype=torch.float32),
                                         win='avg_loss_l1_ssim',
                                         update='append' if loss_x > 0 else None, opts=dict(title='avg_loss_l1_ssim'))
                        loss_cnt = 0
                        loss_sum = 0
                        loss_x += 1

                    #model_io.save_model_by_path(opt.model_path,model.module if isinstance(model, torch.nn.DataParallel) else model)

                # self.train_loss += loss.item()
            # self.epoch_loss = self.train_loss / (self.train_num // 4)
            # print('epoch:{},train_loss:{}'.format(epoch, self.train_loss))
