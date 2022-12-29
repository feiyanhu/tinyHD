from cgi import test
import torch as t
from models.fastsal3D.model import FastSalA

def test_single():
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, True, True]
    d1_last = False
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last)
    
    state_dict= t.load('../pretrained/d1d2d3_S_lt.pth', map_location='cuda:0')['student_model']
    model.load_state_dict(state_dict)
    x = t.zeros(7, 3, 16, 192, 256).cuda()
    model.cuda()
    y, _ = model(x)
    print('start', '-'*30)
    print(y.shape)

def test_multi():
    #d1, d2, d3 = True, False, False #when single_mode is false, config d1, d2, d3 = True, False, False won't work
    #single_mode = False
    
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, False, False]
    d1_last = True
    n_output = 8
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, n_output=n_output)

    state_dict= t.load('../pretrained/d1d2d3_M_lt.pth', map_location='cuda:0')['student_model']
    model.load_state_dict(state_dict)
    x = t.zeros(2, 3, 16, 192, 256).cuda()
    model.cuda()
    with t.cuda.amp.autocast():
        y, y_inters = model(x)
        for yy in y_inters: print(yy.shape)
    print('start', '-'*30)
    print(y.shape)

def test_force_multi():
    #d1, d2, d3 = True, False, False #when single_mode is false, config d1, d2, d3 = True, False, False won't work
    #single_mode = False
    
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, True, True]
    d1_last = False
    force_multi = True
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi)

    #state_dict= t.load('../pretrained/d1d2d3_M_lt.pth', map_location='cuda:0')['student_model']
    #model.load_state_dict(state_dict)
    x = t.zeros(2, 3, 16, 192, 256).cuda()
    model.cuda()
    with t.cuda.amp.autocast():
        y, y_inters = model(x)
        for yy in y_inters: print(yy.shape)
    print('start', '-'*30)
    print(y.shape)


if __name__ == '__main__':
    #test_single()
    #test_force_multi()
    test_multi()
