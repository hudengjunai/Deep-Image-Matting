import torch
import numpy as np
from data import get_train_val_dataloader
from models import Encoder_Decoder
import argparse
import torch.optim as optim
import os
from models import ComposeLoss,AlphaPredLoss
from utils import Visulizer
import time
from torch import autograd


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Pytorch learning args')
    parser.add_argument('--stage',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--num_workers',type=int,default=10)
    parser.add_argument('--crop_size',type=int,default=320)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--wd',type=float,default=1e-5)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--ngpus',type=int,default=1)
    parser.add_argument('--pretrain_model',type=str,default=None)
    parser.add_argument('--eps',type=float,default=1e-6)
    parser.add_argument('--lmd',type=float,default=0.5)
    parser.add_argument('--last_epoch',type=int,default=0)
    parser.add_argument('--freq',type=int,default=20)

class Trainer(object):
    """the unify trainer for encoder-decoder refinehead and ovaerall"""
    model_app={0:"encoder_decoder",
               1:"refine_head",
               2:"over_all"}
    #training stage for encoder_decoder or over_all
    def __init__(self,args):
        self.stage = args.stage
        self.model_name = self.model_app[args.stage]
        self.args = args
        self.train_loader,self.valid_loader = get_train_val_dataloader(batch_size=args.batch_size,
                                                            num_workers=args.num_workers)
        self.model = Encoder_Decoder(stage=args.stage)

        if self.stage==0:
            if not self.args.pretrain_model:
                self.model.load_vggbn('./checkpoints/vgg16_bn-6c64b313.pth')
            else:
                self.model.load_state_dict(self.args.pretrain_model)
            self.loss =[ComposeLoss(eps=self.args.eps),AlphaPredLoss(eps=self.args.eps)]
            self.loss_lambda=[self.args.lmd,1-self.args.lmd]

            self.trainer = optim.SGD([
                {'params': self.model.down1.parameters(),'lr':1},
                {'params': self.model.down2.parameters(),'lr':1},
                {'params': self.model.down3.parameters(), 'lr': 1},
                {'params': self.model.down4.parameters(), 'lr': 1},
                {'params': self.model.down5.parameters(), 'lr': 1},
                {'params': self.model.trans.parameters(), 'lr': 1},
                {'params': self.model.deconv5.parameters(), 'lr': 1},
                {'params': self.model.deconv4.parameters(), 'lr': 1},
                {'params': self.model.deconv3.parameters(), 'lr': 1},
                {'params': self.model.deconv2.parameters(), 'lr': 1},
                {'params': self.model.deconv1.parameters(), 'lr': 1},
                {'params': self.model.rawalpha.parameters(),'lr':2}
            ],
            lr=self.args.lr,weight_decay=self.args.wd,momentum=self.args.momentum)
            self.lr_schedular = optim.lr_scheduler.MultiStepLR(self.trainer,
                                                               milestones=[5,10,30],
                                                               gamma=0.5,
                                                               last_epoch=self.args.last_epoch)
            self.metrics = []

        elif self.stage==1:
            self.model.load_state_dict(self.args.pretrain_model)
            self.loss=[AlphaPredLoss(eps=self.args.eps)]
            self.loss_lambda =[1]
            self.trainer = optim.SGD([
                {'params':self.model.refine_head.parameters(),'lr':1}
            ],
            lr=self.args.lr,weight_decay=self.args.wd,momentum=self.args.momentum)
            self.lr_schedular = optim.lr_scheduler.MultiStepLR(self.trainer,
                                                               milestones=[3,10,30],
                                                               gamma=0.2,
                                                               last_epoch=self.args.last_epoch)
        else:
            self.model.load_state_dict(self.args.pretrain_model)
            self.loss = [AlphaPredLoss(eps=self.args.eps)]
            self.loss_lambda=[1]
            self.trainer = optim.Adam(self.model.parameters(),lr=self.args.lr)
            self.lr_schedular = optim.lr_scheduler.CosineAnnealingLR(self.trainer,T_max=2)

        self.vis = Visulizer(env='{0}_{1}_{2}'.format('matting',self.model_name,time.strftime('%m_%d')))


    def training(self,epoch):
        train_loss = 0.0
        total_loss,prev_loss = 0,0
        for i,(data,label) in enumerate(self.train_loader):
            self.lr_schedular.update(i,epoch)
            data,label = data.cuda(),label.cuda()
            al_pred = self.model(data)
            l_loss = self.loss_lambda[0]*self.loss[0](al_pred,label)
            if len(self.loss)==2:
                l_loss += self.loss_lambda[1]*self.loss[1](al_pred,label)
            l_loss.backward()
            self.trainer.step()
            total_loss += l_loss.item()
            if i%self.args.freq==(self.freq-1):
                step_loss = total_loss - prev_loss
                self.vis.plot('fre_loss',step_loss)
                prev_loss = total_loss
        self.vis.plot("total_loss",total_loss)
        self.vis.log("training epoch {0} finished ".format(epoch))




    def validation(self,epoch):
        mse = 0.0
        sad = 0.0
        self.model.train(mode=False)
        mse_total,mse_pre = 0,0
        for i,(data,label) in enumerate(self.valid_loader):
            data,label = data.cuda(),label.cuda()
            a_pred = self.model(data)
            mse = self.metric_mse(a_pred,label)
            sad = self.metric_sad(a_pred,label)
            mse_total += mse
            if i%self.args.freq ==(self.args.freq-1):
                self.vis.plot('mse_alpha',mse_total/i)
                mse_pre = mse_total
        self.vis.log('the validation of epoch {0}'.format(epoch))


    def save_model(self,epoch):
        file_name = '{0}_{1}_{2}.params'.format(self.model_name,time.strftime('%m_%d'),str(epoch))
        self.model.save_parameters(file_name)

    def metric_mse(self,alpha_pred,label):
        """
        compute the mean square error of the aplha predict
        :param alpha_pred: the predicted alpha value (0,1) N,1,H,W
        :param label: the label fg,bg,mask,alpha_gt
        :return: mse_error
        """
        return 0

    def metric_sad(self,alpha_pred,label):
        """
        the sad of two images
        :param alpha_pred:
        :param label:
        :return:
        """
        return 0
