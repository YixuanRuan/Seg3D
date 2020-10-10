import functools

from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as ta_grad
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import json
from skimage.transform import resize
# from configs.intrinsic_mpi_v11 import opt
import pdb

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Interfaces
##################################################################################
"""
RD_v12: change divergence loss computation, change to CrossGenMS: DecoderMS

fossil seg
"""

def get_generator(gen_opt, train_mode):
    """Get a generator, gen_opt is options of gen, train mode determine which generator model to use"""
    return Segnet(gen_opt)

def get_manual_criterion(name):
    if 'distance' in name.lower():
        return DistanceLoss()
    else:
        raise NotImplementedError(name + 'should in [distance/...]')


##################################################################################
# Generator
##################################################################################

class Segnet(nn.Module):
    def __init__(self, gen_opt):
        super(Segnet, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim = gen_opt.output_dim

        self.encoder_name = gen_opt.encoder_name
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'

        # Feature extractor as Encoder

        self.encoder = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder = DecoderMS_Res(self.input_dim, dim=self.dim,
                                     output_dim=self.output_dim,  # s
                                     n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                     norm=self.norm, decoder_mode=self.decoder_mode)

    def decode(self, x, feats=None):
        return self.decoder(x, feats)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decode(x, feats)
        return out


##################################################################################
# Encoder and Decoders
##################################################################################
class Vgg19EncoderMS(nn.Module):
    def __init__(self, input_dim, pretrained):
        super(Vgg19EncoderMS, self).__init__()
        features = list(vgg19(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'low': result_dict['conv1_2'],
            'mid': result_dict['conv2_2'],
            'mid2': result_dict['conv3_4'],
            'deep': result_dict['conv4_4'],
            'out': result_dict['conv5_4']
        }
        return out_feature


class DecoderMS_Res(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_layers, pad_type, activ, norm, decoder_mode='Basic'):
        """output_shape = [H, W, C]"""
        super(DecoderMS_Res, self).__init__()

        self.select_out = Conv2dBlock(512, 256, kernel_size=1, stride=1, padding=0,
                                      pad_type=pad_type, activation=activ, norm=norm)

        self.select_deep = Conv2dBlock(512, 256, kernel_size=1, stride=1, padding=0,
                                       pad_type=pad_type, activation=activ, norm=norm)

        #self.fuse_out = Conv2dBlock(512, 256, kernel_size=3, stride=1, padding=1,
        #                            pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_deep = ResDilateBlock(512, 128, 224, rate=1,
                                        padding_type=pad_type, norm=norm)
        self.fuse_mid2 = ResDilateBlock(480, 128, 160, rate=1,
                                        padding_type=pad_type, norm=norm)
        self.fuse_mid = ResDilateBlock(288, 96, 128, rate=1,
                                        padding_type=pad_type, norm=norm)
        self.fuse_low = ResDilateBlock(192, 64, 64, rate=1,
                                        padding_type=pad_type, norm=norm)
        self.fuse_input = ResDilateBlock(64 + input_dim, 64, dim, rate=1,
                                        padding_type=pad_type, norm=norm)

        self.contextual_blocks = []
        rates = [1,1,1,1,1]
        if n_layers > 5:
            raise NotImplementedError('contextual layer should less or equal to 5')
        if decoder_mode == 'Basic':
            for i in range(n_layers):
               self.contextual_blocks += [Conv2dBlock(dim, dim, kernel_size=3, dilation=rates[i], padding=rates[i],
                                                      pad_type=pad_type, activation=activ, norm=norm)]
        elif decoder_mode == 'Residual':
            for i in range(n_layers):
               self.contextual_blocks += [ResDilateBlock(input_dim=dim, dim=dim, output_dim=dim, rate=rates[i],
                                          padding_type=pad_type, norm=norm)]
        else:
            raise NotImplementedError

        # use reflection padding in the last conv layer
        self.contextual_blocks += [
            Conv2dBlock(dim, dim, kernel_size=3, padding=1, norm='in',
                        activation=activ, pad_type='reflect')]
        self.contextual_blocks += [
            Conv2dBlock(dim, output_dim, kernel_size=1, norm='none', activation='none', pad_type=pad_type)]
        self.contextual_blocks = nn.Sequential(*self.contextual_blocks)

    @staticmethod
    def _fuse_feature(x, feature):
        _, _, h, w = feature.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, input_x, feat_dict):
        x1 = feat_dict['out']
        x1 = self.select_out(x1)  # 256
        x = feat_dict['deep']
        x = self.select_deep(x)  # 256
        x = self._fuse_feature(x, x1)  # 512
        x = self.fuse_deep(x)  # 224
        x = self._fuse_feature(x, feat_dict['mid2'])  # 224+256=480
        x = self.fuse_mid2(x)  # 160
        x = self._fuse_feature(x, feat_dict['mid'])  # 160+128=288
        x = self.fuse_mid(x)  # 128
        x = self._fuse_feature(x, feat_dict['low'])  # 128+64=192
        x = self.fuse_low(x)  # 64
        x = self._fuse_feature(x, input_x)  #64+3=67
        x = self.fuse_input(x)  # dim

        x = self.contextual_blocks(x)
        return x


##################################################################################
# Modified VGG
##################################################################################
import torch.utils.model_zoo as model_zoo
import math

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(in_channels=3, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg19(in_channels=3, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


##################################################################################
# GAN Blocks: discriminator
##################################################################################


##################################################################################
# Basic Blocks
##################################################################################
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and \
            (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        # if hasattr(m, 'weight') and (classname.find('ConvTranspose') != -1):
        #     init.constant_(m.weight.data, 1.0)
        #     if hasattr(m, 'bias') and m.bias is not None:
        #         init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DeConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, out_padding=0, norm='none', activation='relu', pad_type='zero'):
        super(DeConv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.deconv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride,
                                         output_padding=out_padding, bias=self.use_bias)

    def forward(self, x):
        x = self.deconv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# define a resnet(dilated) block
class ResDilateBlock(nn.Module):
    def __init__(self, input_dim, dim, output_dim, rate,
                 padding_type, norm, use_bias=False):
        super(ResDilateBlock, self).__init__()
        feature_, conv_block = self.build_conv_block(input_dim, dim, output_dim, rate,
                                                     padding_type, norm, use_bias)
        self.feature_ = feature_
        self.conv_block = conv_block
        #self.activation = nn.ReLU(True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def build_conv_block(self, input_dim, dim, output_dim, rate,
                         padding_type, norm, use_bias=False):

        # branch feature_: in case the output_dim is different from input
        feature_ = [self.pad_layer(padding_type, padding=0),
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1,
                              bias=False, dilation=1),
                    self.norm_layer(norm, output_dim),
                    ]
        feature_ = nn.Sequential(*feature_)

        # branch convolution:
        conv_block = []

        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(input_dim, dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, dim),
                       nn.ReLU(True)]
        # dilated conv, padding = dilation_rate, when k=3, s=1, p=d
        # k=5, s=1, p=2d
        conv_block += [self.pad_layer(padding_type='replicate', padding=1*rate),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 bias=False, dilation=rate),
                       self.norm_layer(norm, dim),
                       nn.ReLU(True)]
        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(dim, output_dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, output_dim),
                       ]
        conv_block = nn.Sequential(*conv_block)
        return feature_, conv_block

    @staticmethod
    def pad_layer(padding_type, padding):
        if padding_type == 'reflect':
            pad = nn.ReflectionPad2d(padding)
        elif padding_type == 'replicate':
            pad = nn.ReplicationPad2d(padding)
        elif padding_type == 'zero':
            pad = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return pad

    @staticmethod
    def norm_layer(norm, norm_dim):
        if norm == 'bn':
            norm_layer_ = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            norm_layer_ = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            norm_layer_ = LayerNorm(norm_dim)
        elif norm == 'none':
            norm_layer_ = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        return norm_layer_

    def forward(self, x):
        feature_ = self.feature_(x)
        conv = self.conv_block(x)
        out = feature_ + conv
        out = self.activation(out)
        return out


# Defines the submodule with two-way skip connection.
# X -------------------identity-------------------- X
# |                 /- |submodule1| -- up-sampling1 --|
# |-- down-sampling    |          |                   |
# |                 \- |submodule2| -- up-sampling2 --|
class UnetTwoWaySkipConnectionBlock(nn.Module):
    def __init__(self, outer1_nc, outer2_nc, inner_nc, input_nc=None, submodule1=None, submodule2=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetTwoWaySkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = max(outer1_nc, outer2_nc)
        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                              stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc)
        up_relu = nn.ReLU(True)
        up_norm1 = norm_layer(outer1_nc)
        up_norm2 = norm_layer(outer2_nc)

        if outermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [down_conv]
            up1 = [up_relu, upconv1, nn.Tanh()]
            up2 = [up_relu, upconv2, nn.Tanh()]
            model1 = down + [submodule1] + up1
            model2 = down + [submodule2] + up2
        elif innermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv]
            up1 = [up_relu, upconv1, up_norm1]
            up2 = [up_relu, upconv2, up_norm2]
            model1 = down + up1
            model2 = down + up2
        else:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up1 = [up_relu, upconv1, up_norm1]
            up2 = [up_relu, upconv2, up_norm2]

            if use_dropout:
                model1 = down + [submodule1] + up1 + [nn.Dropout(0.5)]
                model2 = down + [submodule2] + up1 + [nn.Dropout(0.5)]
            else:
                model1 = down + [submodule1] + up1
                model2 = down + [submodule2] + up2

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x):
        if self.outermost:
            return self.model1(x), self.model2(x)
        else:
            return torch.cat([x, self.model1(x)], 1), torch.cat([x, self.model2(x)], 1)
        pass


##################################################################################
# Normalization layers
##################################################################################

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def _get_norm_layer(norm_type='instance'):
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


##################################################################################
# Distribution distance measurements and losses blocks
##################################################################################


class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.eps = 1e-12
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super(JSDivergence, self).__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class DivergenceLoss_bak(nn.Module):
    """assume orthogonal is max divergence"""
    def __init__(self):
        super(DivergenceLoss_bak, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        # pass

    def forward(self, features_x, features_y):
        # normalize
        cos = self.cos_sim(features_x, features_y)  # cosine value between x and y
        d = torch.pow(cos, 2) * 2.0  # symmetry function
        loss = torch.mean(d)
        return loss

class DivergenceLoss(nn.Module):
    """assume orthogonal is max divergence
        Pers_Loss smaller, similarity larger
    """
    def __init__(self, detail_weights, cos_w=0.99, norm_w=0.01, alpha=1.2, scale=1.0):
        super(DivergenceLoss, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.l2_diff = nn.MSELoss()
        self.l1_diff = nn.L1Loss()
        self.detail_weights = detail_weights  # opt.optim.div_detail_dict
        self.scale = scale
        self.alpha = alpha
        self.cos_w = cos_w
        self.norm_w = norm_w
        # pass

    def _compute_dist_loss(self, fea_x, fea_y):
        # x, y shape=[B,C,H,W]
        # pdb.set_trace()
        # for image pool data
        # encourage smaller cosine and larger norm distance
        if fea_x.size(0) < fea_y.size(0):
            n = fea_y.size(0) // fea_x.size(0) + 1
            fea_x = fea_x.repeat(n, 1, 1, 1)[:fea_y.size(0),:,:,:]
        elif fea_x.size(0) > fea_y.size(0):
            n = fea_x.size(0) // fea_y.size(0) + 1
            fea_y = fea_y.repeat(n, 1, 1, 1)[:fea_x.size(0),:,:,:]
        # pdb.set_trace()
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        #d_cos = (1.0 - cos) * 1.0  # cosine distance smaller, d smaller
        d_cos = torch.pow(cos, 2) * 2.0  # symmetry function; the smaller, the better
        d_cos = torch.mean(d_cos)

        d_l2 = self.l1_diff(fea_x, fea_y)  # the smaller, more similar
        d_l2 = self._rescale_distance(d_l2)  # normed into value range (0,1)
        d_l2 = torch.mean(d_l2)

        d = d_cos*self.cos_w + d_l2*self.norm_w
        return d

    def _rescale_distance(self, dist):
        d_ = dist * self.scale
        g_ = -(d_ - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_rescale = (1 / (1 + torch.exp(g_)))
        return 1 - d_rescale  # the larger distance, the better

    def _compute_dist_loss_v1(self, fea_x, fea_y):
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d = (1.0 - torch.pow(cos, 2)) * 2.0
        d = torch.mean(d)
        return d

    def forward(self, features_x, features_y, detail_weights=None):
        fea_weights = self.detail_weights if detail_weights is None else detail_weights
        loss = 0
        n_sum = 0
        for key in features_x.keys():
            if key=='out':
                fea_x = torch.max(features_x[key], dim=2, keepdim=True)[0]
                fea_x = torch.max(fea_x, dim=3, keepdim=True)[0]
                fea_y = torch.max(features_y[key], dim=2, keepdim=True)[0]
                fea_y = torch.max(fea_y, dim=3, keepdim=True)[0]
                loss += self._compute_dist_loss(fea_x, fea_y) * fea_weights[key]
            else:
                loss += self._compute_dist_loss(features_x[key], features_y[key]) * fea_weights[key]
            n_sum += 1

        loss = loss / (n_sum + self.eps)

        return loss


class DistanceLoss_YF(nn.Module):
    """assume a*b==-1 is max divergence"""
    def __init__(self):
        super(DistanceLoss_YF, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()  # why need this?---speed up
        # pass

    def forward(self, features_x, features_y):
        # normalize
        d = self.cos_sim(features_x, features_y)  # cosine value between x and y
        d = self.sigmoid(d)
        d = -torch.log(torch.abs(d))
        # d = 1 - torch.abs(d)
        loss = torch.mean(d)
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, alpha=1.4, scale=1.):
        """
        DistanceLoss
        :param alpha: see eq. 6
        :param scale: scale the L1 distance between two image features
        """
        super(DistanceLoss, self).__init__()
        self.eps = 1e-12
        self.alpha = alpha
        self.scale = scale
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()
        self.l1_diff = nn.L1Loss()
        pass

    def _compute_dist_loss(self, dist):
        # normalize
        d_l1 = dist * self.scale
        g_ab = -(d_l1 - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_psi = 1 / (1 + torch.exp(g_ab))

        return 1 - d_psi

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self._compute_dist_loss(self.l1_diff(features_x[key], features_y[key]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self._compute_dist_loss(self.l1_diff(features_x[idx], features_y[idx]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)

        else:
            loss = self._compute_dist_loss(self.l1_diff(features_x, features_y))

        return loss


class PerspectiveLoss(nn.Module):
    """assume orthogonal is max divergence
        Pers_Loss smaller, similarity larger
    """
    def __init__(self, detail_weights, cos_w=0.01, norm_w=0.99, alpha=1.2, scale=1.0):
        super(PerspectiveLoss, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.l2_diff = nn.MSELoss()
        self.l1_diff = nn.L1Loss()
        self.detail_weights = detail_weights # opt.optim.div_detail_dict_equal
        self.scale = scale
        self.alpha = alpha
        self.cos_w = cos_w
        self.norm_w = norm_w
        # pass

    def _compute_dist_loss(self, fea_x, fea_y):
        # x, y shape=[B,C,H,W]
        # pdb.set_trace()
        # for image pool data
        # encourage larger cosine and smaller norm distance
        if fea_x.size(0) < fea_y.size(0):
            n = fea_y.size(0) // fea_x.size(0) + 1
            fea_x = fea_x.repeat(n, 1, 1, 1)[:fea_y.size(0),:,:,:]
        elif fea_x.size(0) > fea_y.size(0):
            n = fea_x.size(0) // fea_y.size(0) + 1
            fea_y = fea_y.repeat(n, 1, 1, 1)[:fea_x.size(0),:,:,:]
        # pdb.set_trace()
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d_cos = (1.0 - cos) * 1.0  # cosine distance smaller, d smaller
        d_cos = torch.mean(d_cos)

        d_l2 = self.l1_diff(fea_x, fea_y)  # the smaller, more similar
        d_l2 = self._rescale_distance(d_l2)  # normed into value range (0,1)
        d_l2 = torch.mean(d_l2)

        d = d_cos*self.cos_w + d_l2*self.norm_w
        return d

    def _rescale_distance(self, dist):
        d_ = dist * self.scale
        g_ = -(d_ - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_rescale = 1 / (1 + torch.exp(g_))
        return d_rescale

    def _compute_dist_loss_v1(self, fea_x, fea_y):
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d = (1.0 - torch.pow(cos, 2)) * 2.0
        d = torch.mean(d)
        return d

    def forward(self, features_x, features_y, detail_weights=None):
        fea_weights = self.detail_weights if detail_weights is None else detail_weights
        loss = 0
        n_sum = 0
        for key in features_x.keys():
            if key=='out':
                fea_x = torch.max(features_x[key], dim=2, keepdim=True)[0]
                fea_x = torch.max(fea_x, dim=3, keepdim=True)[0]
                fea_y = torch.max(features_y[key], dim=2, keepdim=True)[0]
                fea_y = torch.max(fea_y, dim=3, keepdim=True)[0]
                loss += self._compute_dist_loss(fea_x, fea_y) * fea_weights[key]
            else:
                loss += self._compute_dist_loss(features_x[key], features_y[key]) * fea_weights[key]
            n_sum += 1

        loss = loss / (n_sum + self.eps)

        return loss


class PerspectiveLoss_v0(nn.Module):
    def __init__(self, alpha=1.4, scale=1.):
        """
        DistanceLoss
        :param alpha: see eq. 6
        :param scale: scale the L1 distance between two image features
        """
        super(PerspectiveLoss_v0, self).__init__()
        self.eps = 1e-12
        self.alpha = alpha
        self.scale = scale
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()
        self.l1_diff = nn.L1Loss()
        pass

    def _compute_dist_loss(self, dist):
        # normalize
        d_l1 = dist * self.scale
        g_ab = -(d_l1 - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_psi = 1 / (1 + torch.exp(g_ab))

        return d_psi

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self._compute_dist_loss(self.l1_diff(features_x[key], features_y[key]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self._compute_dist_loss(self.l1_diff(features_x[idx], features_y[idx]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)

        else:
            loss = self._compute_dist_loss(self.l1_diff(features_x, features_y))

        return loss


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        """Patch discriminator, input: (B,C,H,W)"""
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(size=input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(size=input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)


class Grad_Img(nn.Module):

    def __init__(self, Lambda=0.3):
        """ input image has channel 3 (rgb / bgr)"""
        super(Grad_Img, self).__init__()
        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        #conv_x = np.array([[-1.0,0,1], [-2,0,2], [-1,0,1]])
        conv_x = np.array([[0., 0., 0.], [-1, 0, 1], [0., 0., 0.]], dtype=np.float32)
        kernel_x = np.zeros([3,3,3,3], dtype=np.float32)  # [out_ch,in_ch,ks,ks]
        kernel_x[0, 0, :, :] = conv_x
        kernel_x[1, 1, :, :] = conv_x
        kernel_x[2, 2, :, :] = conv_x
        #conv_y = np.array([[-1.0,-2,-1], [0,0,0], [1,2,1]])
        conv_y = np.array([[0.,-1, 0.], [0, 0, 0], [0., 1, 0.]], dtype=np.float32)
        kernel_y = np.zeros([3, 3, 3, 3], dtype=np.float32)
        kernel_y[0, 0, :, :] = conv_y
        kernel_y[1, 1, :, :] = conv_y
        kernel_y[2, 2, :, :] = conv_y
        self.conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float(), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float(), requires_grad=False)

        # pdb.set_trace()
        # self.Lambda = Lambda

    def forward(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        h, w = input.shape[2], input.shape[3]
        grd_x[:, :, [0, h - 1], :] = 0
        grd_x[:, :, :, [0, w - 1]] = 0
        grd_y[:, :, [0, h - 1], :] = 0
        grd_y[:, :, :, [0, w - 1]] = 0

        out = torch.sqrt(grd_x**2 + grd_y**2) / 2

        return out, grd_x, grd_y


class Grad_Img_v1(nn.Module):

    def __init__(self, Lambda=0.3):
        """ input image has channel 3 (rgb / bgr)
            output gradient images of 1 channel (mean along colour channel)
        """
        super(Grad_Img_v1, self).__init__()
        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x = np.array([[-1.0,0,1], [-2,0,2], [-1,0,1]])
        kernel_x = np.zeros([3,3,3,3])
        kernel_x[0, 0, :, :] = conv_x
        kernel_x[1, 1, :, :] = conv_x
        kernel_x[2, 2, :, :] = conv_x
        conv_y = np.array([[-1.0,-2,-1], [0,0,0], [1,2,1]])
        kernel_y = np.zeros([3, 3, 3, 3])
        kernel_y[0, 0, :, :] = conv_y
        kernel_y[1, 1, :, :] = conv_y
        kernel_y[2, 2, :, :] = conv_y
        self.conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float(), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float(), requires_grad=False)

        # pdb.set_trace()
        # self.Lambda = Lambda

    def forward(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        h, w = input.shape[2], input.shape[3]
        grd_x[:, :, [0, h - 1], :] = 0
        grd_x[:, :, :, [0, w - 1]] = 0
        grd_y[:, :, [0, h - 1], :] = 0
        grd_y[:, :, :, [0, w - 1]] = 0

        out = torch.sqrt(grd_x**2 + grd_y**2) / 2

        #grad_x = torch.mean(grd_x, dim=1, keepdim=True)  #[batch,channel,w,h]
        #grad_y = torch.mean(grd_y, dim=1, keepdim=True)
        out = torch.mean(out, dim=1, keepdim=True)
        out = out.repeat(1,3,1,1 )
        return out, grd_x, grd_y


##################################################################################
# Test codes
##################################################################################

if __name__ == '__main__':
    # test_gen()
    # test_gradient_img()
    print('networks definition')
