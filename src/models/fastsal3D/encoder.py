import torch.nn as nn
import numpy as np

from ..backbones import mobilenetv2_pretrain
from ..utils import register_layers, get_student_features
from ..inflate import inflate_mobilenetv2Block
from .config_multi import mbv2_inflate_config_3dd
from .config_single import mbv2_inflate_config

def inflate_encoder(model, reduce_channel, decoder_config, single_mode, force_multi):
    if len(decoder_config) == np.sum(single_mode) and not force_multi:
        mbv2_inflate_config_ = mbv2_inflate_config
    else:
        mbv2_inflate_config_ = mbv2_inflate_config_3dd
        
        
    assert len(mbv2_inflate_config_) == len(model)
    for i, block in enumerate(model):
        tmp_config = mbv2_inflate_config_[i]
        is_first = i==0
        model[i] = inflate_mobilenetv2Block(is_first, block, tmp_config, reduce_channel)
    return model

class Encoder_wrapper(nn.Module):
    def __init__(self, n_reduced, decoder_config, single_mode, force_multi):
        super(Encoder_wrapper, self).__init__()
        self.hook_index = None
        self.encoder = inflate_encoder(mobilenetv2_pretrain(), n_reduced, decoder_config, single_mode, force_multi)
        self.register_encoder_forward_hook('student_encoder', range(1, 19, 1))

    def register_encoder_forward_hook(self, hook_key, hook_index):
        self.hook_key = hook_key
        self.hook_index = hook_index
        register_layers(self.encoder, hook_index, hook_key)

    def forward(self, x):
        y = self.encoder(x)
        y = get_student_features(range(len(self.hook_index)), self.hook_key)
        return y

    def get_encoder(self):
        return self.encoder, self.hook_key, range(len(self.hook_index))