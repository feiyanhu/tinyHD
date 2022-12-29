import torch as t
import torch.nn as nn
from torchvision.models.mobilenetv2 import InvertedResidual, ConvBNReLU, ConvNormActivation

def inflate_weights_conv2d(layer_3d, layer_2d, reduce_channel):
    if layer_3d.weight is not None:
        tmp = t.stack([layer_2d.weight.data for _ in range(layer_3d.weight.data.shape[2])])
        tmp = tmp.permute(1, 2, 0, 3, 4)
        #print(layer_3d.weight.data.shape, tmp.shape)
        if reduce_channel > 1:
            #print(layer_3d, layer_2d)
            tmpn, tmpm, _, _, _ = tmp.shape
            tmpn_new, tmpm_new, tmp1, tmp2, tmp2 = layer_3d.weight.data.shape
            tmp = tmp.view(int((tmpn/tmpn_new) * (tmpm/tmpm_new)), tmpn_new, tmpm_new, tmp1, tmp2, tmp2)
            tmp = tmp.mean(0)
        elif reduce_channel < 1:
            tmpn, tmpm, _, _, _ = tmp.shape
            tmpn_new, tmpm_new, tmp1, tmp2, tmp2 = layer_3d.weight.data.shape
            tmp = t.cat([tmp for _ in range(int(tmpn_new/tmpn))], dim=0)
            tmp = t.cat([tmp for _ in range(int(tmpm_new/tmpm))], dim=1)
            #tmp = tmp.view(int((tmpn/tmpn_new) * (tmpm/tmpm_new)), tmpn_new, tmpm_new, tmp1, tmp2, tmp2)
            #print(layer_3d.weight.data.shape, tmp.shape)
            #exit()
        assert layer_3d.weight.data.shape == tmp.shape
        layer_3d.weight.data = tmp
        print(layer_3d.weight.data.shape, layer_2d.weight.data.shape, '????')
    if layer_3d.bias is not None:
        #print(layer_3d.bias.data.shape, layer_2d.bias.data.shape, '????')
        #exit()
        assert layer_3d.bias.data.shape == layer_2d.bias.data.shape
        layer_3d.bias.data = layer_2d.bias.data
        print(layer_3d.bias.data.shape, layer_2d.bias.data.shape, '????')
    return layer_3d
            

def inflate_weights_batchnorm2d(layer_3d, layer_2d, reduce_channel):
    if layer_3d.weight is not None:
        tmp = layer_2d.weight.data
        if reduce_channel > 1:
            tmp = tmp.view(reduce_channel, -1)
            tmp = tmp.mean(0)
        elif reduce_channel < 1:
            tmp = t.cat([tmp for _ in range(int(1/reduce_channel))], dim=0)
        layer_3d.weight.data = tmp
        print(layer_3d.weight.data.shape, tmp.shape, '????')
    if layer_3d.bias is not None:
        tmp = layer_2d.bias.data
        if reduce_channel > 1:
            tmp = tmp.view(reduce_channel, -1)
            tmp = tmp.mean(0)
        elif reduce_channel < 1:
            tmp = t.cat([tmp for _ in range(int(1/reduce_channel))], dim=0)
        print(layer_3d.bias.data.shape, tmp.shape, '????')
        layer_3d.bias.data = tmp
        #exit()
    return layer_3d

def default_2d_3d(x, new_x=None):
    if new_x is not None:
        return (new_x, ) + x
    return (x[0],) + x

def inflate_conv2d(layer, isfirst, config, reduce_channel, loadweights):
    print('inflate conv2d')
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    stride = layer.stride
    kernel_size = layer.kernel_size
    padding = layer.padding
    groups = layer.groups
    if layer.bias is None: bias=None
    else: bias = True
    
    if len(config) == 0:
        k, s, p = None, None, None
    else:
        k, s, p = config
    kernel_size = default_2d_3d(kernel_size, k)
    padding = default_2d_3d(padding, p) #kernel_size[0]//2
    stride = default_2d_3d(stride, s)
    #print(in_channels, reduce_channel)
    in_channels = in_channels if isfirst else int(in_channels/reduce_channel)
    out_channels = int(out_channels/reduce_channel)
    groups = groups if groups==1 else int(groups/reduce_channel)
    print(in_channels, reduce_channel, out_channels, groups)
    #exit()
    layer_3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=groups, bias=bias)
    if loadweights:
        layer_3d = inflate_weights_conv2d(layer_3d, layer, reduce_channel)
    #print(layer_3d)
    #exit()
    return layer_3d

def inflate_batchnorm2d(layer, bn_config, reduce_channel, loadweights):
    print('inflate batchnorm2d')
    if len(bn_config) == 0:
        num_features = int(layer.num_features/reduce_channel) if layer.num_features>reduce_channel else layer.num_features
        layer_3d = nn.BatchNorm3d(num_features)
    else:
        num_features = int(layer.num_features/reduce_channel) if layer.num_features>reduce_channel else layer.num_features
        eps, momentum = bn_config
        layer_3d = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum)
    if loadweights:
        layer_3d = inflate_weights_batchnorm2d(layer_3d, layer, reduce_channel)
    return layer_3d

def inflate_ConvBNReLU(is_first, block, config, reduce_channel, loadweights):
    print('inflate convbnrelu')
    #print(block)
    for i, x in enumerate(block):
        if isinstance(x, nn.Conv2d):
            is_first = is_first and i == 0
            local_config = config[i]
            block[i] = inflate_conv2d(x, is_first, local_config, reduce_channel, loadweights)
        elif isinstance(x, nn.BatchNorm2d):
            is_first = is_first and i == 0
            local_config = config[i]
            block[i] = inflate_batchnorm2d(x, local_config, reduce_channel, loadweights)
    #print(block)
    #exit()
    return block

def inflate_InvertedResidual(is_first, block, config, reduce_channel, loadweights):
    for i, x in enumerate(block.conv):
        if isinstance(x, ConvNormActivation):
            is_first = is_first and i == 0
            local_config = config[i]
            block.conv[i] = inflate_ConvBNReLU(is_first, x, local_config, reduce_channel, loadweights)
        elif isinstance(x, nn.Conv2d):
            is_first = is_first and i == 0
            local_config = config[i]
            block.conv[i] = inflate_conv2d(x, is_first, local_config, reduce_channel, loadweights)
        elif isinstance(x, nn.BatchNorm2d):
            is_first = is_first and i == 0
            local_config = config[i]
            block.conv[i] = inflate_batchnorm2d(x, local_config, reduce_channel, loadweights)
    return block

def inflate_mobilenetv2Block(is_first, block, config, reduce_channel=1, loadweights=True):
    print(is_first, block, config)

    if isinstance(block, ConvNormActivation):
        return inflate_ConvBNReLU(is_first, block, config, reduce_channel, loadweights)
    elif isinstance(block, InvertedResidual):
        return inflate_InvertedResidual(is_first, block, config, reduce_channel, loadweights)
        