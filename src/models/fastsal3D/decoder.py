from genericpath import exists
from json import decoder
import torch as t
import torch.nn as nn
import numpy as np
from ..sequential_model import adaptation_layers, C2D

def create_decoder(decoder_config, single_mode, force_multi, use_sigmoid=True):
    len_dict = {'d1':4, 'd2':1, 'd3':1}
    n_maps = sum([len_dict[d] for d in decoder_config])
    use_3d = True
    if len(decoder_config) == np.sum(single_mode) and not force_multi:
        use_3d = False
    if use_sigmoid:
        decoder = adaptation_layers([[C2D(1, n_maps, activation=None, kernel_size=1), 'sigmoid']], use_3d=use_3d, reinit=True)
    else:
        decoder = adaptation_layers([[C2D(1, n_maps, activation=None, kernel_size=1), 'sigmoid']], use_3d=use_3d, reinit=True)
    return decoder.layers[0]

class Decoder_wrapper(nn.Module):
    def __init__(self, decoder_config, single_mode, d1_last, force_multi, n_output):
        super(Decoder_wrapper, self).__init__()
        self.decoder = create_decoder(decoder_config, single_mode, force_multi)
        self.n_output = n_output
        self.decoder_config = decoder_config
        self.single_mode = single_mode
        self.d1_last = d1_last
        self.force_multi = force_multi
        if self.d1_last:
            norm_idx, d1_idx = [], []
            for i, d in enumerate(self.decoder_config):
                if d=='d1':d1_idx.append(i)
                else: norm_idx.append(i)
            self.d1_last_idx = norm_idx + d1_idx
            self.decoder_config = [self.decoder_config[i] for i in self.d1_last_idx]
            self.single_mode = [self.single_mode[i] for i in self.d1_last_idx]

    def forward(self, x_maps_):
        if self.d1_last:
            x_maps = [x_maps_[i] for i in self.d1_last_idx]
        else:
            x_maps = x_maps_
        
        if len(self.decoder_config) == np.sum(self.single_mode) and not self.force_multi:
            y = t.cat(x_maps, dim=1).squeeze(2)
            y = self.decoder(y)
        elif len(self.decoder_config) == np.sum(self.single_mode) and self.force_multi:
            y = t.cat(x_maps, dim=1)
            y = t.cat([y for _ in range(self.n_output)], dim=2)
            y = self.decoder(y)
            #print(y.shape)
            #exit()
        else:
            new_maps = []
            for x_map, d, s in zip(x_maps, self.decoder_config, self.single_mode):
                #print(x_map.shape, d, s)
                if s:
                    x_map = t.cat([x_map for _ in range(self.n_output)], dim=2)
                new_maps.append(x_map)
                #print(x_map.shape, '!')
            y = t.cat(new_maps, dim=1)
            y = self.decoder(y)
        return y