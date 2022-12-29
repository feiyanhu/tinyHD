import torch.nn as nn
import torch as t
from torchvision.ops import misc as misc_nn_ops

from torchvision.models.mobilenetv2 import ConvBNReLU
from collections import OrderedDict

from .utils import register_layers

class C2D:
    def __init__(self, output_channel, input_channel=None, activation='relu', kernel_size=3, stride=1, padding = None):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

class CNN_constructer(nn.Sequential):
    def __init__(self, networklist, upsample_mode='bilinear', use_3d=False, 
                 use_group=None, encoder_hook_index=None, decoder_hook_index=None,
                 bn_config=None, use_bias=True, reinit=False):
        d = OrderedDict()
        # next_feature = in_channels
        conv_list = []
        for x in networklist:
            if isinstance(x, C2D):
                conv_list.append(x)
        for i in range(1, len(conv_list), 1):
            if conv_list[i].input_channel is None:
                conv_list[i].input_channel = conv_list[i-1].output_channel

        prev_conv = None
        #networklist = networklist[1:]
        for layer_idx, layer_conv in enumerate(networklist):
            print(layer_idx, layer_conv)
            if isinstance(layer_conv, C2D):
                if layer_conv.padding == None:
                    padding_n = layer_conv.kernel_size // 2
                else:
                    padding_n = layer_conv.padding
                stride = layer_conv.stride

                if use_3d:
                    d["conv{}".format(layer_idx)] = nn.Conv3d(
                        layer_conv.input_channel, layer_conv.output_channel, kernel_size=layer_conv.kernel_size,
                        stride=stride, padding=padding_n, bias=use_bias)
                else:
                    d["conv{}".format(layer_idx)] = nn.Conv2d(
                        layer_conv.input_channel, layer_conv.output_channel, kernel_size=layer_conv.kernel_size,
                        stride=stride, padding=padding_n, bias=use_bias)
                if layer_conv.activation == 'relu':
                    d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
                prev_conv = layer_conv
            elif 'M' in str(layer_conv):
                scale = int(layer_conv[1])
                if use_3d:
                    d["maxpool2d{}".format(layer_idx)] = nn.MaxPool3d(
                        kernel_size=scale, stride=scale,
                        padding=0, dilation=1, ceil_mode=False)
                else:
                    d["maxpool3d{}".format(layer_idx)] = nn.MaxPool2d(
                        kernel_size=scale, stride=scale,
                        padding=0, dilation=1, ceil_mode=False)
            elif 'U' in str(layer_conv):
                scale = int(layer_conv[1])
                d["upsample2d{}".format(layer_idx)] = nn.Upsample(
                    scale_factor=scale, mode=upsample_mode, align_corners=True)
            elif 'S' in str(layer_conv):
                scale = int(layer_conv[1])
                d["shuffle2d{}".format(layer_idx)] = nn.PixelShuffle(scale)
                #try:
                #    next_conv.input_channel = int(prev_conv.output_channel/(scale*scale))
                #except:
                #    print('I am the first!')
            elif str(layer_conv) == 'sigmoid':
                d['sigmoid'] = nn.Sigmoid()
            elif str(layer_conv) == 'tanh':
                d['tanh'] = nn.Tanh()
            elif str(layer_conv) == 'batchnorm2d':
                d['bn'] = nn.BatchNorm2d(prev_conv.output_channel)
            elif str(layer_conv) == 'batchnorm3d':
                if bn_config is None:
                    d['bn3d{}'.format(layer_idx)] = nn.BatchNorm3d(prev_conv.output_channel)
                else:
                    eps, momentum = bn_config
                    d['bn3d{}'.format(layer_idx)] = nn.BatchNorm3d(prev_conv.output_channel, eps=eps, momentum=momentum)
            elif str(layer_conv) == 'logsoftmax':
                d['logsoftmax'] = nn.LogSoftmax(dim=1)
            elif str(layer_conv) == 'relu':
                d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
        d = [l for k,l in d.items()]
        if encoder_hook_index is not None:
            register_layers(d, encoder_hook_index, 'student_encoder')
        if decoder_hook_index is not None:
            register_layers(d, decoder_hook_index, 'student_decoder')

        #if reinit:
        #    init_(d)
        super(CNN_constructer, self).__init__(*d)
        '''for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, std=0.01)
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)'''

class adaptation_layers(nn.Module):
    def __init__(self, layers_config, use_3d=False, use_group=False, detach=False, 
                 bn_config=None, use_bias=True, reinit=False):
        super(adaptation_layers, self).__init__()
        self.layers = nn.ModuleList()
        for i, ls in enumerate(layers_config):
            print(i, ls)
            block = CNN_constructer(ls, use_3d=use_3d, use_group=use_group, bn_config=bn_config, reinit=reinit)
            self.layers.append(
                block
            )
        self.detach = detach
        if reinit:
            #self.__init_weight()
            init_(self)

    def forward(self, inputs):
        out = []
        for input_x, layer in zip(inputs, self.layers):
            if self.detach: input_x = input_x.detach()
            tmp = layer(input_x)
            out.append(tmp)
        return out
    
    def __init_weight(self, std=0.01):
        import math
        for name, param in self.named_parameters():
            #print(name)
            if "weight" in name and 'conv' in name:
                #print('conv_weights', name)
                n = param.shape[2] * param.shape[3] * param.shape[1]
                nn.init.normal_(param, mean=0, std=math.sqrt(2. / n))
                #nn.init.normal_(param, std=std)
                # nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif 'weight' in name and 'conv' not in name:
                nn.init.normal_(param, std=std)
            elif "bias" in name:
                nn.init.constant_(param, 0)

def init_(net):
    for m in net.modules():
        #print(m,' ??')
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    