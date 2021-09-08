import argparse
import os
import torch
import visdom

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='gru', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_id', type=str, default=0, help='gpu id: 0 . use -1 for CPU')
        self.parser.add_argument('--model_path', type=str, default='../checkpoints/gru_0908.pth')

        # for encoder
        # self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        # self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')

        self.parser.add_argument('--show_preview', default=True, action='store_true', help='whether to show preivew')
        self.parser.add_argument('--preview_freq', type=int, default=20)
        self.parser.add_argument('--use_visdom', action='store_true', help='whether to show preivew by visdom')
        self.parser.add_argument('--visdom_name', type=str, default=r'msc_gru')
        self.parser.add_argument('--visdom_ip', type=str, default=r'10.92.173.171')
        self.parser.add_argument('--visdom_port', type=int, default=8888)
        self.parser.add_argument('--save_preview', action='store_true', help='whether to save preivew')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--features', type=int, default=27)
        self.parser.add_argument('--hidden', type=int, default=1)
        self.parser.add_argument('--layers', type=int, default=1)
        self.parser.add_argument('--train_data_num_thread', type=int, default=6, help='data process num thread')
        self.parser.add_argument('--train_data_dir', type=str, default=r'/Users/momo/Desktop/KIMORE/', help='data dir')
        self.parser.add_argument('--test_data_dir', type=str, default=r'/Users/momo/Desktop/KIMORE/')
        self.parser.add_argument('--train_batch_size', type=int, default=4 , help='input batch size')
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size')

        self.parser.add_argument('--total_epoch_num', type=int, default=1000, help='total train iter num')
        self.parser.add_argument('--save_epoch_interval', type=int, default=10, help='how much epochs per save')
        self.parser.add_argument('--label_type', type=str, default=r'label', help='train label type')

        self.isTrain = True
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        if self.opt.gpu_id > 0:
            self.opt.device = torch.device("cuda:%d" % self.opt.gpu_id)
        else:
            self.opt.device = torch.device("cpu")

        try:
            debug_flag = os.environ['IPYTHONENABLE']
        except:
            debug_flag = False

        if debug_flag: self.opt.num_threads = 0

        if self.opt.use_visdom:
            if self.opt.visdom_ip == '10.92.173.171':
                # opt.vis = visdom.Visdom('http://' + opt.visdom_ip, port=opt.visdom_port, env=opt.visdom_name, username='username', password='121121')
                self.opt.vis = visdom.Visdom('http://' + self.opt.visdom_ip, port=self.opt.visdom_port, env=self.opt.visdom_name)
                assert self.opt.vis.check_connection()
                self.opt.vis.close()
                # opt.vis = visdom.Visdom('http://' + opt.visdom_ip, port=opt.visdom_port, env=opt.visdom_name, username='username', password='121121')
                self.opt.vis = visdom.Visdom('http://' + self.opt.visdom_ip, port=self.opt.visdom_port, env=self.opt.visdom_name)

        return self.opt
