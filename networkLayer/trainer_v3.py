from collections import OrderedDict

import utils
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import glob
from networkLayer.networks_v3 import get_generator
import copy
import numpy as np
import pdb

# noinspection PyAttributeOutsideInit
"""
v8: use new perspective loss: cosine + L2?; for gradient loss, use mse to
prevent bad pixels
v11: change perceptual dict the same as fd, use cosine for preserve_info
RD: refined data
RD_v9: share low level encoders
RD_v10: use color shading for comparison
RD_v10_test: pad images to eliminate corner artifacts
RD_v11_test: from RD_v10_test
RD_v12: networks_mpi_RD_v9 --> RD_v12

fossil seg
"""


class Trainer_Basic(nn.Module):
    @staticmethod
    def name():
        return 'Segfossil_Trainer'

    def __init__(self, t_opt):
        super(Trainer_Basic, self).__init__()
        self.opt = t_opt
        self.is_train = t_opt.is_train
        self.save_dir = t_opt.output_root

        self.weights = copy.deepcopy(t_opt.optim)
        nb = t_opt.data.batch_size
        size = t_opt.data.new_size

        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
        self.input_i = None
        self.input_s = None

        self.lr = None
        # Init the networks
        print('Constructing Networks ...')
        self.gen_seg = get_generator(t_opt.model.gen, t_opt.train.mode).cuda()

        print('Loading Networks\' Parameters ...')
        if t_opt.continue_train:
            which_epoch = t_opt.which_epoch
            self.resume(self.gen_seg, 'G_seg', which_epoch)

        # define loss functions---need modify
        self.padedge = 30
        self.padimg = nn.ReplicationPad2d(self.padedge)
        self.criterion_CE = torch.nn.CrossEntropyLoss()  # ignore_index=0
        self.criterion_L1 = torch.nn.L1Loss()

        # initialize optimizers
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_seg.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_g, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, t_opt.optim))

        self.gen_seg.train()

        print('---------- Networks initialized -------------')
        # utils.print_network_info(self.gen_seg)
        print('---------------------------------------------')
        utils.print_network_info(self, print_struct=False)
        # pdb.set_trace()

    def forward(self):
        # pred_seg = self.gen_seg(self.padimg(self.input))
        # #self.fake_s = fake_s.repeat(1, 3, 1, 1)
        # self.pred_seg = pred_seg[:,:,self.padedge:-(self.padedge),self.padedge:-(self.padedge)]
        pred_seg = self.gen_seg(self.input)
        self.pred_seg = pred_seg

    def set_input(self, input_data):
        input_i = input_data['I_seq']
        input_s = input_data['S']

        input_i_cur = input_data['I']

        self.img_name = input_data['name']

        # input image
        self.input_i = input_i.cuda()  # input_img
        self.input_i_cur = input_i_cur.cuda()
        if hasattr(self, "real_s"):
            del self.real_s
        self.input_s = input_s.cuda()  # seg_gt

        self.input = Variable(self.input_i)
        # real_s = torch.tensor(self.input_s.clone(), dtype=torch.long)
        self.real_s = Variable(input_s).cuda().detach()
        # print([torch.max(self.real_s), torch.min(self.real_s)])

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def inference(self, input_img=None):
        self.gen_seg.eval()

        with torch.no_grad():
            # reduce memory usage and speed up
            self.forward()
            self.loss_basic_computation()

            self.loss_gen_total = self.loss_gen_basic

        self.gen_seg.train()

    def gen_update(self):
        self.optimizer_gen.zero_grad()

        # compute loss
        self.loss_basic_computation()
        self.loss_gen_total = self.loss_gen_basic
        self.loss_gen_total.backward()
        # optimize
        self.optimizer_gen.step()

    def loss_basic_computation(self):
        """ compute all the loss """
        weight = self.opt.optim  # weight for optim settings

        # print('compute cross entropy')
        if weight.crossentropy_w > 0:
            self.loss_idt_s = self.criterion_CE(self.pred_seg, self.real_s)
            # self.loss_idt_s = self.criterion_L1(self.pred_seg, self.real_s.unsqueeze(1).float())
        else:
            self.loss_idt_s = self.Tensor([0.0])

        # print(self.loss_idt_s)
        self.loss_gen_basic = weight.crossentropy_w * self.loss_idt_s

    def optimize_parameters(self):
        # forward
        for _ in range(1):
            self.forward()
            self.gen_update()

    def get_current_errors(self):
        """plain prediction loss"""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_gen_total
        ret_errors['idt_S'] = self.loss_idt_s

        ret_errors['img_name'] = self.img_name

        return ret_errors

    def get_current_visuals(self):
        mean = self.opt.data.image_mean
        std = self.opt.data.image_std
        use_norm = self.opt.data.use_norm

        # img_real_s = utils.tensor2img(self.input_s.detach().clone().float(), mean, std, use_norm)
        img_real_s = np.uint8(self.input_s[0].cpu().numpy() * 255)
        # img_pred_s = utils.tensor2img(self.pred_seg.detach().clone(), mean, std, use_norm)
        img_pred_s = np.uint8(torch.argmax(self.pred_seg, dim=1)[0].cpu().numpy() * 255)
        img_input = utils.tensor2img(self.input_i_cur.detach().clone(), mean, std, use_norm)

        ret_visuals = OrderedDict([('input_I', img_input),
                                   ('real_Seg', img_real_s),
                                   ('pred_Seg', img_pred_s),
                                  ])

        return ret_visuals

    @staticmethod
    def loss_log(losses):
        log_detail = '\t{}:{:.5f} \n\t{}'.format(
                                                 'loss_seg', losses['idt_S'],
                                                 losses['img_name'])
        return log_detail

    def resume(self, model, net_name, epoch_name):
        """resume or load model"""
        if epoch_name == 'latest' or epoch_name is None:
            model_files = glob.glob(self.save_dir + "/*.pth")
            if len(model_files):
                save_path = max(model_files)
            else:
                save_path = 'NotExist'
        else:
            save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
            save_path = os.path.join(self.save_dir, save_filename)

        # pdb.set_trace()
        if os.path.exists(save_path):
            if len(self.opt.gpu_ids) > 1:
                model = nn.DataParallel(model, devices_ids=self.opt.gpu_ids)
            model.load_state_dict(torch.load(save_path))
            print('Loding model from : %s .' % save_path)
        else:
            print('Begin a new train')
        pass

    def save_network(self, model, net_name, epoch_name):
        save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
        utils.check_dir(self.save_dir)

        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)

        model.cuda()
        if len(self.opt.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.opt.gpu_ids)

    def save(self, label):
        self.save_network(self.gen_seg, 'G_seg', label)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        if lr <= 1.4e-6 and self.opt.use_wave_lr:  # reset lr
            #for group in self.optimizers[0].param_groups:
            #    group['lr'] = lr * 1e2
            self.refresh_optimizers(lr * 1e2)
            lr = self.optimizers[0].param_groups[0]['lr']
            print('new learning rate = %.7f' % lr)
        self.lr = lr

    def get_lr(self):
        return self.lr

    def refresh_optimizers(self, lr):
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_seg.parameters() if p.requires_grad],
                                              lr=lr,
                                              betas=(self.opt.optim.beta1,
                                                     self.opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, self.opt.optim))
