import torch
from torch import autograd
from torch import nn
import numpy


class ComposeLoss(nn.Module):
    """compute the ComposeLoss of gt_merged and pre_merged"""
    def __init__(self,eps=1e-5):
        super(ComposeLoss,self).__init__()
        self.eps = eps

    def forward(self,a_pred,fg,bg,gt_merge,mask):
        """
        compute the compose loss
        :param a_pred: the encoder_decoder predict alpha
        :param fg:     the foreground image [0,255.0] float32
        :param bg:     the background image[0,255.0] float32
        :param gt_merge:the gt_alpha merged iamge [0,255.0] float32
        :param mask:    the unknown region mask (0,1) binary mask mat
        :return:        the matting region compose loss
        """
        
        pass

class AlphaPredLoss(nn.Module):
    """compute the pred alpha and the gt_alpha loss"""
    def __init__(self,eps=1e-5):
        super(AlphaPredLoss,self).__init__()
        self.eps=eps

    def forward(self, a_pred,a_gt,mask):
        """
        compute the encoder-decoder or refine head alpha loss
        :param a_pred: the encoder decoder or refine_head output alpha,value in (0,1) float32
        :param a_gt:   the groundtruth alpha value in (0,1) float32
        :param mask:   the unknown region mask
        :return:
        """
        pass




if __name__=='__main__':

