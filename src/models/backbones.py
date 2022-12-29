import torch.nn as nn
from torchvision.models import vgg16, squeezenet1_1, mobilenet_v2
#from efficientnet_pytorch import EfficientNet

def vgg_16_pretrained():
    vgg_16 = vgg16(pretrained=True)
    features = list(vgg_16.features)[:30]
    features = nn.Sequential(*features)
    return features

def squeezenet11_pretrained():
    sn11 = squeezenet1_1(pretrained=True)
    features = list(sn11.features)
    features = nn.Sequential(*features)
    return features

def efficientnet_pretrain():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    return model

def mobilenetv2_pretrain():
    model = mobilenet_v2(pretrained=True)
    features = list(model.features)
    features = nn.Sequential(*features)
    return features