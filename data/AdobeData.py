import mxnet as mx
from mxnet.gluon.data import Dataset,DataLoader
from mxnet.image import imread
from PIL import Image
import os
import numpy as np
import cv2
import math
from mxnet import nd
import mxnet.gluon.data.vision.transforms as T

default_transform = T.Compose(T.ToTensor(),T.Normalize(mean=(),std=()))

class AdobeDataset(Dataset):
    """the adobe dataset to for load the train and test dataset
    get item will return the bg_img,trimap,alpha,"""


    def __init__(self,usage,size=320,transform = default_transform):
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        self.transform = transform
        self.size = size
        self.usage = usage
        filename = '{}_names.txt'.format(usage)
        with open(filename,'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        if self.usage in ['train','valid']:
            self.a_path = './data/adobe_train/alpha/'
            self.fg_path = './data/adobe_train/fg'
            self.bg_path = './data/adboe_train/bg'
            fg_names = 'training_fg_names.txt'
            bg_names = 'training_bg_names.txt'
            with open(fg_names,'r') as f:
                self.fg_files = f.read().splitlines()
            with open(bg_names,'r') as f:
                self.bg_files = f.read().splitlines()

        elif self.usage=='test':
            self.a_path=" "
            pass

    def __len__(self):
        return len(self.names)


    def process(self,im_name,bg_name):
        im = cv2.imread(os.path.join(self.fg_path,im_name))
        a = cv2.imread(os.path.join(self.a_path,im_name))
        h,w = im.shape[:2]
        bg = cv2.imread(os.path.join(self.bg_path,bg_name))
        bh,bw = bg.shape[:2]
        wratio = w/bw
        hratio = h/bh
        ratio = max(wratio,hratio)
        if ratio>1
            bg = cv.resize(src=bg,
                           dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
                           interpolation=cv.INTER_CUBIC)
        return self.compose(im,bg,a,w,h)

    def compose(self,fg,bg,a,w,h):
        fg = np.array(fg,np.float32)
        bg_h,bg_w = bg.shape[:2]

        x =0
        if bg_w>w:
            x = np.random.randint(0,bg_w-w)
        y=0
        if bg_h>h
            y = np.random.randint(0,bg_h-h)
        bg = np.array(bg[y:y+h,x:x+w],np.float32)

        #gernerate alpah and merged image
        alpha = np.zeros((h,w,1),np.float32)
        alpha[:,:,0] = a/255.0
        im = alpha*fg + (1-alpha)*bg
        im = im.astype(np.uint8)

        #generate trimap
        fg_tr = np.array(np.equal(a,255).astype(np.float32))
        un_tr = np.array(np.not_equal(a,0).astype(np.float32))
        un_tr = cv2.dilate(un_tr,self.kernel,
                           iterations = np.random.randint(1,20))
        trimap = fg_tr*255+(un_tr-fg)*128



        return im,alpha,fg,bg,trimap


    def __getitem__(self, item):
        """get the x and y
        x is the [merged[0:3],trimap[3] ] ,
        y is the [bg[0:3],fg[3:6],mask[6],alpha[7] ]"""
        name = self.names[item]
        fcount,bcount = [int(x) for x in name.split('.')[0].split('_')]
        im_name = self.fg_files[fcount]
        bg_name = self.bg_files[bcount]

        merged,alpha,fg,bg,trimap = self.process(im_name,bg_name)
        x = nd.empty((self.size,self.size,4),dtype=np.float32)
        y = nd.empty((self.size,self.size,7),dtype=np.float32)

        if self.transform:
            merged = self.transform(nd.array(merged))


        x[:,:,0:3] = nd.array(merged)
        x[:,:,-1] = nd.array(trimap)

        y[:,:,0:3] = nd.array(bg)
        y[:,:,3:6] = nd.array(fg)
        y[:,:,-1] = nd.array(alpha)
        return x,y


if __name__=='__main__':
    "this is the test for dataset"
    train_dataset = AdobeDataset(usage='train')
    for i,(x,y) in enumerate(train_dataset):
        print(x.shape)
        print(y.shape)
        if i ==3:
            break
            

