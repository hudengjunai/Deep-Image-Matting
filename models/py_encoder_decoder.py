import torch
from torch import nn
if __name__=='__main__':
    from pretrain_keys_pair import pairs_keys
from .pretrain_keys_pair import pairs_keys
class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class segnetUp(nn.Module):
    def __init__(self,in_size,out_size):
        super(segnetUp,self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size,out_size,5,1,2) #kernel_size,stride,padding

    def forward(self, inputs,indices,output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        return outputs

class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape

class transMap(nn.Module):
    def __init__(self):
        super(transMap,self).__init__()
        self.conv1 = conv2DBatchNormRelu(512,512,3,1,1)
        self.conv2 = conv2DBatchNormRelu(512,512,1,1,0)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class refineHead(nn.Module):
    def __init__(self):
        super(refineHead,self).__init__()
        self.conv1 = conv2DBatchNormRelu(4,64,3,1,1)
        self.conv2 = conv2DBatchNormRelu(64,64,3,1,1)
        self.conv3 = conv2DBatchNormRelu(64,64,3,1,1)
        self.alpha = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.alpha(x)


class Encoder_Decoder(nn.Module):
    """the deep image matting encoder decoder structure
    stage 0:just train encoder decoder
    stage 1:just train refinehead
    stage 2:overall train the encoder decoder and refine_head"""

    def __init__(self,stage=1):
        super(Encoder_Decoder,self).__init__()
        self.down1 = segnetDown2(4,64)
        self.down2 = segnetDown2(64,128)
        self.down3 = segnetDown3(128,256)
        self.down4 = segnetDown3(256,512)
        self.down5 = segnetDown3(512,512)

        self.trans = transMap()

        self.deconv5 = segnetUp(512,512)
        self.deconv4 = segnetUp(512,256)
        self.deconv3 = segnetUp(256,128)
        self.deconv2 = segnetUp(128,64)
        self.deconv1 = segnetUp(64,3)

        self.rawalpha = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=1,kernel_size=5,padding=2,stride=1),
            nn.Sigmoid())

        self.refine_head = refineHead()
        self.stage = stage

    def forward(self, x):
        down1,indices_1,unpool_shape1 = self.down1(x)
        down2,indices_2,unpool_shape2 = self.down2(down1)
        down3,indices_3,unpool_shape3 = self.down3(down2)
        down4,indices_4,unpool_shape4 = self.down4(down3)
        down5,indices_5,unpool_shape5 = self.down5(down4)

        trans = self.trans(down5)

        up5 = self.deconv5(trans,indices_5,unpool_shape5)
        up4 = self.deconv4(up5,indices_4,unpool_shape4)
        up3 = self.deconv3(up4,indices_3,unpool_shape3)
        up2 = self.deconv2(up3,indices_2,unpool_shape2)
        up1 = self.deconv1(up2,indices_1,unpool_shape1)
        raw_alpha = self.rawalpha(up1)
        if self.stage==0:
            return raw_alpha,0
        else:
            refine_in = torch.cat((x[:,:3,:,:],alpha),1)
            refine_alpha = self.refine_head(refine_in)
            return raw_alpha,refine_alpha

    def load_vggbn(self,file):
        state_dict = torch.load(file)
        origin_keys = state_dict.keys()

        struct_dict = self.state_dict() #torch.load('./checkpoints/struct.pth')
        p0 = pairs_keys[0]
        #conv filters
        origin_conv0 = state_dict[p0[0]]
        addChn_conv0 = struct_dict[p0[1]]
        addChn_conv0.data.zero_() #set all the alpha channel filters zero
        addChn_conv0.data[:,0:3,:,:]=origin_conv0.data

        #all other data insert
        for p in pairs_keys[1:]:
            k1,k2 = p
            struct_dict[k2].data = state_dict[k1].data
        self.load_state_dict(struct_dict)



if __name__=='__main__':
    model = Encoder_Decoder(stage=0)
    #this is a pre stored random params
    torch.save(model.state_dict(),'./checkpoints/struct.pth')
    model.load_vggbn('./checkpoints/vgg16_bn-6c64b313.pth')
    x = torch.rand(2,4,320,320)
    y = model(x)
