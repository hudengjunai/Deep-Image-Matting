from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import PIL
from PIL import Image
import math
import torchvision.transforms as T
import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
mean=(0.485, 0.456, 0.406)
std=[0.229, 0.224, 0.225]
default_transform = T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean,std=std)])




class AdobeDataset(Dataset):

    def __init__(self,usage,size=320,transform=default_transform):
        super(AdobeDataset,self).__init__()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.transform = transform
        self.size = size
        self.usage = usage
        filename = './data/adobe_data/{}_names.txt'.format(usage) # just store the image index for save
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        if self.usage in ['train', 'valid']:
            self.a_path = './data/adobe_data/trainval/alpha'
            self.fg_path = './data/adobe_data/trainval/fg/'
            self.bg_path = '/data/jh/notebooks/hehao/datasets/coco/train2014/' #the coco path
            fg_names = './data/adobe_data/training_fg_names.txt' # the file name all the foreground files
            bg_names = './data/adobe_data/training_bg_names.txt' # the file name all the background file names
            with open(fg_names, 'r') as f:
                self.fg_files = f.read().splitlines()
            with open(bg_names, 'r') as f:
                self.bg_files = f.read().splitlines()
        self.unknown_code = 128

    def __len__(self):
        return len(self.names)

    def process(self,im_name,bg_name):
        fg = cv2.imread(os.path.join(self.fg_path, im_name))
        a = cv2.imread(os.path.join(self.a_path, im_name),0)
        bg = cv2.imread(os.path.join(self.bg_path, bg_name))
        fg = cv2.cvtColor(fg,cv2.COLOR_BGR2RGB)
        bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)

        h, w = fg.shape[:2]
        bh, bw = bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = max(wratio, hratio)
        if ratio > 1: # need to enlarge the bg image
            bg = cv2.resize(src=bg,
                           dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
                           interpolation=cv2.INTER_CUBIC)
        return self.compose(fg, bg, a, w, h)

    def compose(self, fg, bg, a, w, h):
        fg = np.array(fg, np.float32)
        bg_h, bg_w = bg.shape[:2]

        x = 0
        if bg_w > w:
            x = np.random.randint(0, bg_w - w)
        y = 0
        if bg_h > h:
            y = np.random.randint(0, bg_h - h)
        bg = np.array(bg[y:y + h, x:x + w], np.float32)

        # gernerate alpah and merged image
        alpha = np.zeros((h, w, 1), np.float32)
        alpha[:, :, 0] = a / 255.0
        im = alpha * fg + (1 - alpha) * bg


        # generate trimap
        fg_tr = np.array(np.equal(a, 255).astype(np.float32))
        un_tr = np.array(np.not_equal(a, 0).astype(np.float32))
        un_tr = cv2.dilate(un_tr, self.kernel,
                           iterations=np.random.randint(1, 20))
        trimap = fg_tr * 255 + (un_tr - fg_tr) * 128

        return im, alpha, fg, bg, trimap # in channel BGR


    def __getitem__(self, item):
        """get the x and y
        x is the [merged[0:3],trimap[3] ] ,
        y is the [bg[0:3],fg[3:6],mask[6],alpha[7] ]"""
        name = self.names[item]
        fcount,bcount = [int(x) for x in name.split('.')[0].split('_')]
        im_name = self.fg_files[fcount]
        bg_name = self.bg_files[bcount]

        merged,alpha,fg,bg,trimap = self.process(im_name,bg_name) #all is float32 type and RGB channels last and 255 max


        data = torch.empty(size=(self.size,self.size,4),dtype=torch.float32)
        label = torch.empty(size=(self.size,self.size,11),dtype=torch.float32)

        #safe crop and resize
        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            merged = np.fliplr(merged)
            alpha = np.fliplr(alpha)
            fg = np.fliplr(fg)
            bg = np.fliplr(bg)
            trimap = np.fliplr(trimap)

        #to generate the clip contains the trimap unknow region
        different_sizes= [(320, 320), (480, 480), (640, 640)]
        scale_crop = random.choice(different_sizes)
        x,y = self.random_choice(trimap,scale_crop)

        merged = self.safe_crop(merged,x,y,crop_size=scale_crop,fixed=(self.size,self.size))
        alpha = self.safe_crop(alpha,x,y,crop_size=scale_crop,fixed=(self.size,self.size))
        fg    = self.safe_crop(fg,x,y,crop_size=scale_crop,fixed=(self.size,self.size))
        bg    = self.safe_crop(bg,x,y,crop_size=scale_crop,fixed=(self.size,self.size))
        trimap = self.safe_crop(trimap,x,y,crop_size=scale_crop,fixed=(self.size,self.size))
        mask = np.equal(trimap, 128).astype(np.float32)

        image =torch.tensor(merged).div(255)
        image.sub_(torch.tensor(mean)).div_(torch.tensor(std))

        data[:,:,0:3] = image # firset three rgb channel
        data[:,:,3] = torch.tensor(trimap)  # last channel is trimap

        label[:,:,0:3] = torch.tensor(bg)
        label[:,:,3:6] = torch.tensor(fg)
        label[:,:,6:9] = torch.tensor(merged)
        label[:,:,9:10] = torch.tensor(alpha.reshape(self.size,self.size,1))
        label[:,:,10] = torch.tensor(mask)

        data = data.transpose(0, 1).transpose(0, 2).contiguous()
        label = label.transpose(0, 1).transpose(0, 2).contiguous()
        return data,label

    def random_choice(self,trimap, crop_size):
        crop_height, crop_width = crop_size
        y_indices, x_indices = np.where(trimap == self.unknown_code)
        num_unknowns = len(y_indices)
        tri_h,tri_w = trimap.shape[:2]
        x, y = 0, 0
        if num_unknowns > 0:
            ix = np.random.choice(range(num_unknowns))
            center_x = x_indices[ix]
            center_y = y_indices[ix]                 #sampled center_x and center_y,so the crop region have unknown region
            x = max(0, center_x - int(crop_width / 2))
            y = max(0, center_y - int(crop_height / 2))
            x = min(x,tri_w-crop_width)
            y = min(y,tri_h-crop_height)            #prerequest condition,the tri_w,tri_h must bigger than crop_size
        return x, y

    def safe_crop(self,mat, x, y, crop_size,fixed):
        crop_height, crop_width = crop_size
        if len(mat.shape) == 2 :
            ret = np.zeros((crop_height, crop_width), np.float32)
        else:
            channels = mat.shape[2]
            ret = np.zeros((crop_height, crop_width, channels), np.float32)
        crop = mat[y:y + crop_height, x:x + crop_width]
        h, w = crop.shape[:2]
        ret[0:h, 0:w] = crop
        if crop_size != fixed:
            ret = cv2.resize(ret, dsize=fixed, interpolation=cv2.INTER_NEAREST)
        return ret

def get_train_val_dataloader(batch_size,num_workers):
    train_dataset = AdobeDataset(usage='train')
    val_dataset = AdobeDataset(usage='valid')
    train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers)
    valid_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers)
    return train_loader,valid_loader

if __name__=='__main__':
    """ this is the dataset test
    run in DeepMatting_MXNet root directory path"""
    train_dataset = AdobeDataset(usage='train')
    for i,(x,y) in enumerate(train_dataset):
        print(x.shape)
        print(y.shape)
        if i==3:
            break
    valid_dataset = AdobeDataset(usage='valid')
    for i,(x,y) in enumerate(valid_dataset):
        print(x.shape)
        print(y.shape)
        if i==3:
            break
    print("test valid dataset finished")

    train_loader,valid_loader = get_train_val_dataloader(4,10)
    for i,(x,y) in enumerate(train_loader):
        print(x.shape)
        print(y.shape)
        if i==4:
            break
    print("test dataloader finished")