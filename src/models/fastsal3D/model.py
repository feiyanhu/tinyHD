#from .student_abstract import Student
from numpy import single
import torch
from torch.autograd.grad_mode import F
import torch.nn as nn
from .encoder import Encoder_wrapper
from .decoder import Decoder_wrapper
from .aggregator import Aggregator_wrapper

class FastSalA(nn.Module):
    def __init__(self, n_reduced, decoder_config, single_mode, d1_last=False, force_multi=False, n_output=16):
        super(FastSalA, self).__init__()
        #d1, d2, d3 = [True, True, True]
        #n_reduced = 4
        #single_mode = True
        self.single_mode = single_mode

        self.encoder = Encoder_wrapper(n_reduced, decoder_config, single_mode=single_mode, force_multi=force_multi)
        self.aggregator = Aggregator_wrapper(n_reduced, decoder_config, single_mode=single_mode, force_multi=force_multi, n_output=n_output)
        self.decoder = Decoder_wrapper(decoder_config, single_mode=single_mode, d1_last=d1_last, force_multi=force_multi, n_output=n_output)

    def forward(self, x):
        x = self.encoder(x)
        #return x
        x_inter = self.aggregator(x)
        x = self.decoder(x_inter)
        return x, x_inter
    
    def get_optimizer(self, lr, use_adam=False, exclude_list=[None]):
        weight_decay_default = 0.0005
        lr_dict = {'encoder': 0.5*lr, 'alpha': 100*lr}
        regularization_dict = {'encoder': 0.0005, 'alpha': 0}
        params = []
        parem_dict = dict(self.named_parameters())
        #parem_dict.update(dict(self.student_audio.named_parameters()))
        #print(len(parem_dict))
        for key, value in parem_dict.items():
            module_name = key.split('.')[0]
            if value.requires_grad and module_name not in exclude_list:
                if module_name in lr_dict.keys():
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr_dict[module_name] * 2, 'weight_decay': 0}]
                        print(key, lr_dict[module_name] * 2)
                    else:
                        params += [{'params': [value], 'lr': lr_dict[module_name],
                                    'weight_decay': regularization_dict[module_name]}]
                        print(key, lr_dict[module_name])
                else:
                    if 'bias' in key:
                        params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                        print(key, lr * 2)
                    else:
                        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay_default}]
                        print(key, lr)
        if use_adam:
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        return optimizer