from mxnet.gluon import nn
from mxnet import autograd
from mxnet import nd
from mxnet.initializer import Xavier,Zero
from gluoncv.model_zoo.model_store import get_model_file
from mxnet import ndarray
vgg_spec = [(2,2,3,3,3),(64,128,256,512,512)]


class Matting_Encoder(nn.HybridBlock):
    def __init__(self,spec=vgg_spec):
        super(Matting_Encoder,self).__init__()
        layers,channels= spec
        features = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                features.add(nn.Conv2D(channels=channels[i], kernel_size=3, padding=1, strides=1,
                                       weight_initializer=Zero(),
                                       bias_initializer='zeros'))
                features.add(nn.BatchNorm())
                features.add(nn.Activation('relu'))
            features.add(nn.MaxPool2D(strides=2))
        self.features = features
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)

    def load_vgg_encoder_params(self):
        """load from vggbn16 params,initialize the vgg conv1 for alpha zero"""
        vgg_file = get_model_file('vgg%d%s' % (16, '_bn'))
        loaded = ndarray.load(vgg_file)
        params = self._collect_params_with_prefix()
        for name in loaded:
            if name in params:
                params[name]._load_init(loaded[name])



class Encoder_Decoder(nn.HybridBlock):
    """this is the Deep Image matting encoder decoder structure for alpha matting"""
    def __init__(self,spec,stage):
        super(Encoder_Decoder,self).__init__()
        self.stage = stage
        self.encoder = Matting_Encoder(spec=spec)
        self.decoder = Matting_Decoder()

        self.refine = nn.HybridSequential(prefix='refine')
        channels = [64,64,64,1]
        with self.refine.name_scope():
            for i,c in enumerate(channels):
                self.refine.add(nn.Conv2D(channels=64,kernel_size=3,
                                          weight_initializer=Xavier(),
                                          bias_initializer='zeros'))
                self.refine.add(nn.BatchNorm())
                self.refine.add(nn.Activation('relu'))


    def hybrid_forward(self, F,rgbt,fg,bg):
        if self.stage==1:
            feature  = self.encoder(x)
            alpha = self.decoder(feature)
            return [alpha]
        elif self.stage==2:
            feature = self.encoder(x)
            alpha = self.decoder(feature)
            ref_input = nd.concat(rgbt[:,0:3,:,:],alpha,dims=1)
            alpha = self.refine(ref_input)
            return [alpha]
        else
            feature = self.encoder(x)
            alpha = self.decoder(feature)
            ref_input = nd.concat(rgbt[:, 0:3, :, :], alpha, dims=1)
            alpha2 = self.refine(ref_input)
            return [alpha,alpha2]




class Matting_Decoder(nn.HybridBlock):
    def __init__(self):
        super(Matting_Decoder,self).__init__()
        self.trans = nn.HybridSequential(prefix='')
        self.trans.add(nn.Conv2D(channels=512,kernel_size=1))
        self.trans.add(nn.BatchNorm())

        channels = [512,256,128,64,64]
        self.dec_layers = []
        for i,c in enumerate(channels):
            block = nn.HybridSequential(prefix='decove_{0}'.format(6-i))
            block.add(nn.Conv2D(channels=c,kernel_size=5,padding=2,
                                   weight_initializer=Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                                   bias_initializer='zeros'))
            block.add(nn.BatchNorm())
            block.add(nn.Activation('relu'))
            self.dec_layers.append(block)

        self.alpha_block = nn.HybridSequential()
        self.alpha_block.add(nn.Conv2D(channels=1,kernel_size=5,padding=2,
                              weight_initializer=Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                              bias_initializer='zeros'))
        self.alpha_block.add(nn.BatchNorm())
        self.alpha_block.add(nn.Activation('relu'))


    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.trans(x)
        for layer in self.dec_layers:
            out = layer(out)
            _,_,h,w = out.shape
            out = F.contrib.BilinearResize2D(out, height= * 2, width= * 2)
        out = self.alpha_block(out)
        return out




