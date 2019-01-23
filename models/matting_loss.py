from mxnet import nd
from mxnet.gluon import nn


class Compose_Loss(nn.Block):
    def __init__(self,eps):
        self.eps = eps

    def forward(self,fg,bg,pred,mask,merg):
        c = fg*pred +(1-pred)*bg
        dis = mask*(c-merg)
        l = nd.sqrt(self.eps + nd.square(dis).sum(0))
        return l

class AlphaPre_Loss(nn.Block):
    def __init__(self,eps):
        self.eps = eps

    def forward(self,pre,aph,mask):
        pass

class AlphaRef_Loss(nn.Block):
    """this is the alpha refinement loss"""
    def __init__(self,eps):
        self.eps = eps

    def forward(self, pred,aph,mask):
        pass


