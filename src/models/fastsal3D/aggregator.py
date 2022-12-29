import numpy as np
import torch as t
import torch.nn as nn
from .config_multi import inv_config_1, inv_config_2, inv_config_4, inv_config_8, \
                          decoder_inflate_config_3dd, decoder_inflate_config_3dd_d1m, \
                          decoder_inflate_config_3dd_d1m_8
from .config_single import decoder_inflate_config

from .utils import process_config, process_config2
from ..inflate import inflate_mobilenetv2Block
from torchvision.models.mobilenetv2 import InvertedResidual

def create_decoder(reduce_channel, decoder_config, single_mode, force_multi, n_output):
    if reduce_channel == 1:
        inv_config = inv_config_1
    elif reduce_channel == 2:
        inv_config = inv_config_2
    elif reduce_channel == 4:
        inv_config = inv_config_4
    elif reduce_channel == 8:
        inv_config = inv_config_8

    #print(decoder_config)
    if len(decoder_config) == np.sum(single_mode) and not force_multi:
        inv_config, decoder_inflate_config_, idx_list = process_config2(inv_config, decoder_config, decoder_inflate_config, decoder_inflate_config_3dd_d1m, single_mode)
    else:
        if n_output == 16:
            inv_config, decoder_inflate_config_, idx_list = process_config2(inv_config, decoder_config, decoder_inflate_config_3dd, decoder_inflate_config_3dd_d1m, single_mode)
        elif n_output == 8:
            inv_config, decoder_inflate_config_, idx_list = process_config2(inv_config, decoder_config, decoder_inflate_config_3dd, decoder_inflate_config_3dd_d1m_8, single_mode)
    #forward_config, inv_config_, inv_config_return = process_config(inv_config, d1, d2, d3)
    module_list = [InvertedResidual(tmp[0], tmp[1], 1, 2) for tmp in inv_config]

    for b in module_list:
        b.use_res_connect = False
    
    return module_list, decoder_inflate_config_, idx_list

def inflate_aggregator(model, decoder_inflate_config_):
    assert len(model) == len(decoder_inflate_config_)
    for i, block in enumerate(model):
        is_first = True
        model[i] = inflate_mobilenetv2Block(is_first, block, decoder_inflate_config_[i][0], 1, loadweights=False)
    return model

class Aggregator_wrapper(nn.Module):
    def __init__(self, n_reduced, decoder_config, single_mode, force_multi, n_output):
        super(Aggregator_wrapper, self).__init__()
        modules, decoder_inflate_config, idx_list = create_decoder(n_reduced, decoder_config, single_mode, force_multi, n_output)
        modules = inflate_aggregator(modules, decoder_inflate_config)
        self.aggregate_adapter = nn.ModuleList(modules)
        self.set_forward(decoder_config, single_mode,rcmode=n_reduced, d_idx=idx_list, n_output=n_output)
        
        self.shuffle_up2 = nn.PixelShuffle(2)
        self.trilinear_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=True)
        self.trilinear_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def upsample_features(self, features, temporal_upsample=False):
        N, C, T, H, W = features.shape
        features = features.permute(0, 2, 1, 3, 4).reshape(N*T, C, H, W)
        features = self.shuffle_up2(features)
        features = features.view(N, T, int(C/4), 2*H, 2*W).permute(0, 2, 1, 3, 4)
        if temporal_upsample:
            features = self.bilinear_up2(features)
        return features
    
    def set_forward(self, decoder_config, single_mode, rcmode=1, d_idx=None, n_output=16):
        self.decoder_config = decoder_config
        self.single_mode = single_mode
        self.rcmode = rcmode
        self.d_idx = d_idx
        self.n_output = n_output

    @staticmethod
    def cat_upsample_layers(student_e, up_func, h, dim=1):
        student_e = [up_func(t.cat(student_e[slice(*x)], dim=dim)) for x in h]
        return student_e
    
    def forward_decoder1(self, x, aggregate_adapter, rcmode, use3d=False):
        out = []
        tmp = x[0]
        #print(tmp.shape)
        for i, m in enumerate(aggregate_adapter[0:4]):
            tmp = m(tmp)
            #print(tmp.shape,'??')
            if i == 1:
                if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                tmp = self.upsample_features(tmp)
            #print(tmp.shape)
        #exit()
        out.append(tmp)
        tmp = x[1]
        for i, m in enumerate(aggregate_adapter[4:8]):
            tmp = m(tmp)
            #print(tmp.shape,'??')
            if i == 0 or i ==1:
                if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                #if use3d and i == 0:
                #    if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                tmp = self.upsample_features(tmp)
            #print(tmp.shape)
        out.append(tmp)
        #exit()
        #print(tmp.shape)
        tmp = x[2]
        for i, m in enumerate(aggregate_adapter[8:12]):
            tmp = m(tmp)
            #print(tmp.shape,'??', m)
            if i == 0 or i ==1 or i ==2:
                if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                #if (i ==0) and use3d:
                #    tmp = self.trilinear_up(tmp)
                tmp = self.upsample_features(tmp)
            #print(tmp.shape,'!!')
        out.append(tmp)
        #exit()
        #print(tmp.shape)
        #exit()
        tmp = x[3]
        #print(tmp.shape)
        for i, m in enumerate(aggregate_adapter[12:16]):
            #print(m)
            #continue
            if i == 0 or i ==1 or i ==2 or i == 3:
                if (rcmode == 4 or rcmode=='len') and i==0:
                    if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                    tmp = self.trilinear_up2(tmp)
                else:
                    if use3d: tmp = self.trilinear_up(tmp) #unmark for 16
                    #if (i == 1) and use3d: tmp = self.trilinear_up(tmp)
                    tmp = self.upsample_features(tmp)
            #print(x[3].shape)
            #print(tmp.shape,'??')
            tmp = m(tmp)
            #print(tmp.shape,'!!')
        #exit()
        out.append(tmp)
        #for y in out:print(y.shape)
        #exit()
        #out = t.cat(out, dim=1)
        if use3d and False:
            new_out = []
            for tmp, ns in zip(out, [1, 2, 3, 4]):
                for _ in range(ns):
                    tmp = self.trilinear_up(tmp)
                new_out.append(tmp)
            out = t.cat(new_out, dim=1)
        else:
            out = t.cat(out, dim=1)
        #print(out.shape)
        #exit()
        return out
        
    def forward_decoder2(self, x, aggregate_adapter, rcmode, use3d=False):
        #16:24
        if not use3d:
            x2 = self.upsample_features(aggregate_adapter[0](x[-1])) + x[-2]
            x2 = self.upsample_features(aggregate_adapter[1](x2)) + aggregate_adapter[2](x[-3])
            x2 = self.upsample_features(aggregate_adapter[3](x2)) + aggregate_adapter[5](aggregate_adapter[4](x[-4]))
        else:
            x2 = self.trilinear_up(self.upsample_features(aggregate_adapter[0](x[-1]))) + x[-2]
            #print(aggregate_adapter[5], aggregate_adapter[4])
            #exit()
            x2 = self.trilinear_up(self.upsample_features(aggregate_adapter[1](x2))) + aggregate_adapter[2](x[-3])
            x2 = self.trilinear_up(self.upsample_features(aggregate_adapter[3](x2))) + aggregate_adapter[5](aggregate_adapter[4](x[-4]))

            #x2 = self.trilinear_up(self.upsample_features(aggregate_adapter[1](x2))) + x[-3]
            #x2 = self.trilinear_up(self.upsample_features(aggregate_adapter[3](x2))) + x[-4]

        if rcmode in [1, 'e2', 'len']:
            #print(aggregate_adapter[6](x2).shape, x2.shape)
            #exit()
            x2 = self.upsample_features(aggregate_adapter[6](x2)) #default
            if use3d and self.n_output==16: x2  = self.trilinear_up(x2)
        elif self.rcmode in [2, 4, 8]:
            x2 = self.trilinear_up2(aggregate_adapter[6](x2)) #half size
            if use3d and self.n_output==16: x2  = self.trilinear_up(x2)
            

        x2 = aggregate_adapter[7](x2)
        return x2
    def forward_decoder3(self, x, aggregate_adapter, rcmode, use3d=False):
        #24:37
        #for y in x:print(y.shape)
        #print('-'*20)
        if rcmode in [2, 4, 8]:
            up_funcs = [self.trilinear_up2, self.trilinear_up2, self.trilinear_up2, self.trilinear_up2] #half
        elif rcmode == 1:
            up_funcs = [self.upsample_features, self.upsample_features, self.upsample_features, self.upsample_features] #default
        elif rcmode == 'e2':
            up_funcs = [self.trilinear_up2, self.upsample_features, self.upsample_features, self.upsample_features]
        x_1 = [uf(y) for uf, y in zip(up_funcs, x)] #default
        if use3d and self.n_output==16: x_1  = [self.trilinear_up(y) for y in x_1]
        #for y in x_1:print(y.shape)
        #for y in x:print(y.shape)
        x_1_adp = [m(y) for m, y in zip(aggregate_adapter[0:3],x[:-1])]
        x_1[1:] = [y+ya for y, ya in zip(x_1[1:], x_1_adp)]
        #for y in x_1:print(y.shape)
        #for y in x_1_adp:print(y.shape)
        #print('-'*20)
        #exit()

        x_2 = [self.upsample_features(m(y)) for m, y in zip(aggregate_adapter[3:6], x_1[1:])]
        if use3d: x_2  = [self.trilinear_up(y) for y in x_2]
        #for y in x_2:print(y.shape)
        #for y in x_1:print(y.shape, '?')
        x_2_adp = [m(y) for m, y in zip(aggregate_adapter[6:9],x_1[:-1])]
        #for y in x_2_adp:print(y.shape)
        #exit()
        x_2 = [y+ya for y, ya in zip(x_2, x_2_adp)]
        
        #for y in x_2:print(y.shape)
        #print('-'*20, 2)
        x_3 = [self.upsample_features(y) for y in x_2[1:]]
        if use3d: x_3  = [self.trilinear_up(y) for y in x_3]
        #for y in x_3:print(y.shape)
        x_3_adp = [m(y) for m, y in zip(aggregate_adapter[9:11],x_2[:-1])]
        #for y in x_3_adp:print(y.shape)
        #print(aggregate_adapter[9])
        #print(aggregate_adapter[10])
        #exit()
        x_3 = [y+ya for y, ya in zip(x_3, x_3_adp)]
        
        #for y in x_3:print(y.shape)
        #print('-'*20)

        x_4 = [self.upsample_features(y) for y in x_3[1:]]
        if use3d: x_4  = [self.trilinear_up(y) for y in x_4]
        x_4_adp = [m(y) for m, y in zip(aggregate_adapter[11:12],x_3[:-1])]
        x_4 = [y+ya for y, ya in zip(x_4, x_4_adp)]

        x_out = aggregate_adapter[12](x_4[0])
        return x_out
    
    def forward(self, x):
        x = self.cat_upsample_layers(x, lambda x:x, [(0, 1), (1, 3), (3, 6), (6, 13), (13, 18)], dim=1)
        if self.rcmode == 8:
            up_func = self.trilinear_up2
        else:
            up_func = self.upsample_features
        x[1:] = self.cat_upsample_layers(x[1:], up_func, [(0,1), (1,2), (2,3), (3, 4)], dim=1)
        x = self.cat_upsample_layers(x, lambda x:x, [(0,2), (2,3), (3,4), (4, 5)], dim=1)
        #x = [self.forward_ema(y, i) for i, y in enumerate(x)]
        #for y in x:print('-'*10, y.shape)
        #print(self.decoder_config, self.single_mode)
        #print(self.rcmode)
        #print(self.d_idx)
        #exit()
        out = []
        for decoder_type, s_mode, (start_i, end_i) in zip(self.decoder_config, self.single_mode, self.d_idx):
            use3d = not s_mode
            #print(decoder_type, s_mode, use3d, start_i, end_i)
            if decoder_type == 'd1':
                tmp = self.forward_decoder1(x, self.aggregate_adapter[start_i:end_i], self.rcmode, use3d=use3d)
            elif decoder_type == 'd2':
                tmp = self.forward_decoder2(x, self.aggregate_adapter[start_i:end_i], self.rcmode, use3d=use3d)
            elif decoder_type == 'd3':
                tmp = self.forward_decoder3(x, self.aggregate_adapter[start_i:end_i], self.rcmode, use3d=use3d)
            tmp = self.sigmoid(tmp)
            out.append(tmp)

        return out
    
    def cat_layers_features(self, x):
        x = self.cat_upsample_layers(x, lambda x:x, [(0, 1), (1, 3), (3, 6), (6, 13), (13, 18)], dim=1)
        if self.rcmode == 8:
            up_func = self.trilinear_up2
        else:
            up_func = self.upsample_features
        x[1:] = self.cat_upsample_layers(x[1:], up_func, [(0,1), (1,2), (2,3), (3, 4)], dim=1)
        x = self.cat_upsample_layers(x, lambda x:x, [(0,2), (2,3), (3,4), (4, 5)], dim=1)
        return x



if __name__ == '__main__':
    a = 0
    #3, 64, 'M2', 128, 'M2', 256, 'M2', 512, 'M2', 512