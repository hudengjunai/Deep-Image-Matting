import torch
from torch import autograd
from torch import nn
import numpy


class ComposeLoss(nn.Module):
    """compute the ComposeLoss of gt_merged and pre_merged"""
    def __init__(self,eps=1e-6,cuda=False):
        super(ComposeLoss,self).__init__()
        self.eps = torch.tensor(eps)

    def cuda(self):
        self.eps = self.eps.cuda()



    def forward(self,a_pred,label):
        #a_pred,fg,bg,gt_merge,mask):
        """
        compute the compose loss
        :param a_pred: the encoder_decoder predict alpha
        :param fg:     the foreground image [0,255.0] float32
        :param bg:     the background image[0,255.0] float32
        :param gt_merge:the gt_alpha merged iamge [0,255.0] float32
        :param mask:    the unknown region mask (0,1) binary mask mat
        :return:        the matting region compose loss
        """
        fg = label[:,:3,:,:]
        bg = label[:,3:6,:,:]
        gt_merged = label[:,6:9,:,:]
        mask = label[:,-1:,:,:]
        prd_comp = a_pred*fg+(1-a_pred)*bg
        dis = mask*(gt_merged - prd_comp)/255
        dis = dis.sum()
        loss = torch.sqrt(torch.pow(dis,2)+torch.pow(self.eps,2))
        return loss


class AlphaPredLoss(nn.Module):
    """compute the pred alpha and the gt_alpha loss"""
    def __init__(self,eps=1e-6):
        super(AlphaPredLoss,self).__init__()
        self.eps = torch.tensor(eps)

    def cuda(self):
        self.eps = self.eps.cuda()



    def forward(self, a_pred,label):
        #a_pred,a_gt,mask):
        """
        compute the encoder-decoder or refine head alpha loss
        :param a_pred: the encoder decoder or refine_head output alpha,value in (0,1) float32
        :param a_gt:   the groundtruth alpha value in (0,1) float32
        :param mask:   the unknown region mask
        :return:
        """
        a_gt = label[:,-2:-1,:,:]
        mask = label[:,-1:,:,:]
        dis = mask*(a_pred-a_gt) #dis is every

        loss = torch.sqrt(torch.pow(dis,2).sum() + torch.pow(self.eps,2))
        return loss




if __name__=='__main__':
    print("test the training loss")

    a_pred = torch.rand((4,1,30,30),dtype=torch.float32)
    label = torch.rand((4,11,30,30),dtype=torch.float32)

    alpha_loss = AlphaPredLoss()
    loss = alpha_loss(a_pred,label)
    print(loss.item())

    comp_loss = ComposeLoss()
    loss2 = comp_loss(a_pred,label)
    print(loss2.item())

