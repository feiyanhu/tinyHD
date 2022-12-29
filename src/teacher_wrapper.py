import torch as t
import torch.nn as nn
from models.S3D.SalGradNet_original import SalGradNet as S3D_SGN
from models.fastsal3D.model import FastSalA
from models.other_models.ViNet import VideoSaliencyModel
from models.other_models.TASED_net import TASED_v2
from models.S3D.SalGradNet_DLA import SalGradNet as S3D_SGN_DLA

class S3D_wrapper:
    def __init__(self, model_path):
        self.model = S3D_SGN()
        self.model.load_state_dict(t.load(model_path, map_location='cuda:0'))
        self.model.eval()
        self.stack_func = lambda x: t.stack(x, dim=1)
    
    def get_supervision(self, x, teacher_x_idx):
        h1, h2, h3, h4, y_t = list(), list(), list(), list(), list()
        with t.no_grad():
            for x_idx in teacher_x_idx:
                t_h1, t_h2, t_h3, t_h4, t_y_t = self.model(x[:, :, x_idx, ...])
                h1.append(t_h1)
                h2.append(t_h2)
                h3.append(t_h3)
                h4.append(t_h4)
                y_t.append(t_y_t)
        [h1, h2, h3, h4, y_t] = [self.stack_func(tmp) for tmp in [h1, h2, h3, h4, y_t]]
        inter_maps = t.stack([h1, h2, h3, h4])
        return y_t, inter_maps

class SO_wrapper:
    def __init__(self, model_path):
        reduced_channel = 1
        decoder_config = ['d1', 'd2', 'd3']
        single_mode = [True, True, True]
        force_multi = False
        d1_last = False

        self.model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi)
        state_dict = t.load(model_path, map_location='cuda:0')['student_model']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.stack_func = lambda x: t.stack(x, dim=1)
    
    def get_supervision(self, x, teacher_x_idx):
        h1, h2, h3, h4, h5, h6, y_t = list(), list(), list(), list(), list(), list(), list()
        with t.no_grad():
            for x_idx in teacher_x_idx:
                t_y_t, t_h = self.model(x[:, :, x_idx, ...])
                h1.append(t_h[0][:, 0, :, :, :])
                h2.append(t_h[0][:, 1, :, :, :])
                h3.append(t_h[0][:, 2, :, :, :])
                h4.append(t_h[0][:, 3, :, :, :])
                h5.append(t_h[1])
                h6.append(t_h[2])
                y_t.append(t_y_t)
        [h1, h2, h3, h4, h5, h6, y_t] = [self.stack_func(tmp) for tmp in [h1, h2, h3, h4, h5, h6, y_t]]
        inter_maps = [t.stack([h1, h2, h3, h4]), h5, h6]
        #print(y_t.shape, inter_maps.shape)
        return y_t, inter_maps

class MO_wrapper:
    def __init__(self, model_path):
        reduced_channel = 1
        decoder_config = ['d1', 'd2', 'd3']
        single_mode = [True, False, False]
        force_multi = False
        d1_last = True
        self.model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi)
        state_dict = t.load(model_path, map_location='cuda:0')['student_model']
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def get_supervision(self, x):
        with t.no_grad():
            y_t, t_h = self.model(x)
            h1, h2, h3, h4 = t_h[0][:, 0:1, :, :, :], t_h[0][:, 1:2, :, :, :], t_h[0][:, 2:3, :, :, :], t_h[0][:, 3:4, :, :, :]

        y_t = y_t.permute(0, 2, 1, 3, 4)
        inter_maps = [t.stack([h1, h2, h3, h4]), t_h[1].permute(0, 2, 1, 3, 4).unsqueeze(2), t_h[2].permute(0, 2, 1, 3, 4).unsqueeze(2)]
        #print(inter_maps[0].shape, inter_maps[1].shape, inter_maps[2].shape)
        return y_t, inter_maps

class TASED_wrapper:
    def __init__(self, model_path):
        self.model = TASED_v2()
        self.model.eval()
        self.load_weight(model_path)
    
    def get_supervision(self, x, teacher_x_idx):
        y_all = []
        with t.no_grad():
            for x_idx in teacher_x_idx:
                y = self.model(x[:, :, x_idx, ...])
                y_all.append(y.unsqueeze(1))
        y_all = t.cat(y_all, dim=1)
        return y_all
    
    def load_weight(self, file_weight):
        print ('loading weight file')
        weight_dict = t.load(file_weight)
        model_dict = self.model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')

class ViNet_wrapper:
    def __init__(self, model_path):
        self.model = VideoSaliencyModel(
            transformer_in_channel=32, 
            nhead=4,
            use_upsample=bool(1),
            num_hier=3,
     	    #num_clips=32
        )
        state_dict = t.load(model_path, map_location='cuda:0')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def get_supervision(self, x, teacher_x_idx):
        y_all = []
        with t.no_grad():
            for x_idx in teacher_x_idx:
                y = self.model(x[:, :, x_idx, ...])
                y_all.append(y.unsqueeze(1))
        y_all = t.cat(y_all, dim=1)
        return y_all

class S3D_DLA_wrapper:
    def __init__(self, model_path):
        self.model = S3D_SGN_DLA()
        self.model.add_layer()
        self.model.load_state_dict(t.load(model_path, map_location='cuda:0')['student_model'])
        self.model.eval()
        self.stack_func = lambda x: t.stack(x, dim=1)
    
    def get_supervision(self, x, teacher_x_idx):
        h1, h2, h3, h4, y_t = list(), list(), list(), list(), list()
        with t.no_grad():
            for x_idx in teacher_x_idx:
                t_y_t = self.model(x[:, :, x_idx, ...])
                y_t.append(t_y_t)
        y_t = self.stack_func(y_t)
        return y_t
