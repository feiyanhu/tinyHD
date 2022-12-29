import numpy as np
def process_config(inv_config, d1, d2, d3):
    forward_config, inv_config_, inv_config_return = [], [], []
    for i, d_config in enumerate([d1, d2, d3]):
        if i==0:
            c_len = 0; idx_=(0, 0)
            if d_config: c_len = 16; idx_=(0, 16)
        elif i==1:
            c_len = 0; idx_=(0, 0)
            if d_config: c_len =  8; idx_=(16, 24)
        elif i==2:
            c_len = 0; idx_=(0, 0)
            if d_config: c_len = 13; idx_=(24, 37)
        if idx_[1] - idx_[0] > 0:
            inv_config_return.append(idx_)
        inv_config_.extend(inv_config[idx_[0]:idx_[1]])
        forward_config.append(c_len)
    #print(forward_config)
    idx_start = np.cumsum([0]+forward_config[:-1])
    forward_config = [(i, i+j) for i, j in zip(idx_start, forward_config)]
    return forward_config, inv_config_, inv_config_return

def process_config2(inv_config, decoder_config, decoder_inflate_config_S, decoder_inflate_config_M, single_mode):
    print(len(inv_config))
    print(decoder_config, single_mode)
    print(len(decoder_inflate_config_S), len(decoder_inflate_config_M))
    new_inv_config = []
    len_list = []
    new_decoder_inflate_config = []
    idx_dict = {'d1':(0, 16), 'd2':(16, 24), 'd3':(24, 37)}
    for d, s in zip(decoder_config, single_mode):
        start_i ,end_i = idx_dict[d]
        new_inv_config.extend(inv_config[start_i:end_i])
        len_list.append(end_i-start_i)
        if s:
            new_decoder_inflate_config.extend(decoder_inflate_config_S[start_i:end_i])
        else:
            new_decoder_inflate_config.extend(decoder_inflate_config_M[start_i:end_i])
    idx_list = np.cumsum([0]+ len_list)
    idx_list = [(idx_list[i], idx_list[i+1]) for i in range(len(idx_list) - 1)]
    return new_inv_config, new_decoder_inflate_config, idx_list