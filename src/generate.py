import numpy as np
import torch as t
from torch.utils.data import DataLoader
import cv2
from os.path import exists
from os import mkdir

import dataset.DHF1K_vid as dhf1k
import dataset.UCFsport_vid as ucf
from metrics.utils import normalize_map

import os

from models.fastsal3D.model import FastSalA
from my_sampler import EvalClipSampler


def post_process_png(prediction, original_shape):
    prediction = normalize_map(prediction)
    prediction = (prediction * 255).astype(np.uint8)
    prediction = cv2.resize(prediction, (original_shape[0], original_shape[1]),
                            interpolation=cv2.INTER_CUBIC)
    prediction = cv2.GaussianBlur(prediction, (5, 5), 0)
    #prediction = normalize_map(prediction)
    prediction = np.clip(prediction, 0, 255)
    return prediction

def vid_length(sal_indx):
    from os import listdir
    from os.path import isfile, join
    len_dict = {}
    for vid in range(600, 700):
        direc = '/data/DHF1K/annotation/{:04d}/maps/'.format(vid+1)
        onlyfiles = [f for f in listdir(direc) if isfile(join(direc, f)) and '.png' in f]
        len_dict[vid+1] = len(onlyfiles) - (16 - sal_indx[-1]) + 1
    return len_dict

def padd_same(window, cid, vid_dir, img):
    if cid == window:
        for i in range(1, 16):
            file_name = '{}/{:04d}.png'.format(vid_dir, i)
            #print(file_name)
            cv2.imwrite(file_name, img)

def padd_append(window_size, cid, start_idx, newsize, vid_dir, model, vgg_in, sal_indx, data_type, tmp_name):
    if cid <= window_size:
        for i in range(1, cid, sal_indx[-1] - sal_indx[0]+1):
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            img = y.detach().cpu().numpy()[0][0]
            img = post_process_png(img, newsize[[1, 0]])
            #for i in range(len_dict[int(vid)] - 14, len_dict[int(vid)]+1):
            #file_name = '{}/{:04d}.png'.format(vid_dir, i)
            if data_type == 'dhf1k':
                file_name = '{}/{:04d}.png'.format(vid_dir, i)
            elif data_type == 'ucf':
                file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, i)
            print(file_name)
            cv2.imwrite(file_name, img)

def padd_append_multi(window_size, cid, start_idx, newsize, vid_dir, model, vgg_in, sal_indx):
    if cid == sal_indx[0] + 1:
        for i in range(1, cid, sal_indx[-1] - sal_indx[0]+1):
            #print(i)
            #continue
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            imgs = y.detach().cpu().numpy()[0][0]#[sal_indx]
            #print(imgs.shape, range(i, i+sal_indx[-1] - sal_indx[0]+1), len(range(i, i+sal_indx[-1] - sal_indx[0]+1)))
            #exit()
            for img, j in zip(imgs, range(i, i+sal_indx[-1] - sal_indx[0]+1)):
                img = post_process_png(img, newsize[[1, 0]])
                #for i in range(len_dict[int(vid)] - 14, len_dict[int(vid)]+1):
                file_name = '{}/{:04d}.png'.format(vid_dir, j)
                print('padding append multi !!!', file_name)
                cv2.imwrite(file_name, img)
        #exit()
def padd_same_end(window, cid, vid_dir, img, sal_indx):
    for i in range(cid+1, cid+(16 - sal_indx[-1])):
        #print(i)
        #continue
        file_name = '{}/{:04d}.png'.format(vid_dir, i)
        print('padding same !!!', file_name)
        cv2.imwrite(file_name, img)
    #exit()


def test_one_single(model, dataloader, save_path, data_type, sal_indx):
    if data_type == 0: data_type = 'dhf1k' 
    elif data_type == 1: data_type = 'ucf'
    elif data_type == 2: data_type = 'hollywood' #'dhf1k'
    #len_dict = vid_length()
    for i, (X, original_size, video_ids, clip_ids, has_label, rand_sig) in enumerate(dataloader):
        vgg_in = X[0].float().cuda(0)
        #with t.cuda.amp.autocast():
        y = model(vgg_in)[0]#.unsqueeze(1)
        y = y.detach().cpu().numpy()
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #continue
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
            if data_type == 'dhf1k':
                vid_dir = '{}{:04d}'.format(save_path, int(vid))
            elif data_type == 'ucf':
                vid_dir = '{}{}'.format(save_path, vid)
                tmp_name = vid.split('/')[-1]
                tmp_name = tmp_name[:-4] + '_' + tmp_name[-3:]
            if not exists(vid_dir):
                mkdir(vid_dir)
                print(vid_dir)
            #print(imgs.shape, cids)
            for img, cid in zip(imgs, cids):
                #print(img.shape, cid)
                img = post_process_png(img, newsize[[1, 0]])
                if data_type == 'dhf1k':
                    file_name = '{}/{:04d}.png'.format(vid_dir, cid)
                    tmp_name = None
                elif data_type == 'ucf':
                    file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, cid)
                #print(file_name)
                #exit()
                #print(img.shape)
                #exit()
                cv2.imwrite(file_name, img)
                #print(cid, file_name, img.shape)
                #exit()
                #padd_same(16, cid, vid_dir, img) #paper number?? need to check
                padd_append(16, cid, start_idx, newsize, vid_dir, model, vgg_in, sal_indx, data_type, tmp_name)

def test_one_multi(model, dataloader, save_path, data_type, sal_indx):
    if data_type == 0: data_type = 'dhf1k' 
    elif data_type == 1: data_type = 'ucf'
    elif data_type == 2: data_type = 'hollywood' #'dhf1k'
    #exit()
    for i, (X, original_size, video_ids, clip_ids, has_label, rand_sig) in enumerate(dataloader):
        vgg_in = X[0].float().cuda(0)
        #with t.cuda.amp.autocast():
        y = model(vgg_in)[0].squeeze(1)
        y = y.detach().cpu().numpy()
        #print(y.shape)
        #exit()
        #y = y[:, sal_indx, :, :]
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
            #vid_dir = '{}{:04d}'.format(save_path, int(vid))
            if data_type == 'dhf1k':
                vid_dir = '{}{:04d}'.format(save_path, int(vid))
            elif data_type == 'ucf':
                vid_dir = '{}{}'.format(save_path, vid)
                tmp_name = vid.split('/')[-1]
                tmp_name = tmp_name[:-4] + '_' + tmp_name[-3:]
            if not exists(vid_dir):
                mkdir(vid_dir)
                print(vid_dir)
            #print(imgs.shape, cids)
            for img, cid in zip(imgs, cids):
                #print(img.shape, cid)
                img = post_process_png(img, newsize[[1, 0]])
                #file_name = '{}/{:04d}.png'.format(vid_dir, cid)
                #print(img.shape)
                #exit()
                if data_type == 'dhf1k':
                    file_name = '{}/{:04d}.png'.format(vid_dir, cid)
                    tmp_name = None
                elif data_type == 'ucf':
                    file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, cid)
                cv2.imwrite(file_name, img)
                
                #padd_append_multi(16, cid, start_idx, newsize, vid_dir, model, vgg_in, sal_indx)
                #print(len_dict[int(vid)], cid); exit()
                #if cid == len_dict[int(vid)]:
                #    padd_same_end(16, cid, vid_dir, img, sal_indx)
                #exit()

def main(model, dataloader, sal_idx, pretrain_path, save_path, data_type):
    if pretrain_path:
        state_dict = t.load(pretrain_path, map_location='cuda:0')['student_model']
    else:
        print('please specify trained models.')
        exit()

    model.load_state_dict(state_dict)
    model.cuda(0)

    model.eval()
    with t.no_grad():
        if len(sal_idx) == 1:
            test_one_single(model, dataloader['val'], save_path, data_type, sal_indx)
        else:
            test_one_multi(model, dataloader['val'], save_path, data_type, sal_indx)
    print('--------------------------------------------->>>>>>')
    #print('loss val {}'.format(loss_val))

def config_dataset(data_dir, model_input_size, model_output_size):
    data_select_idx = [dt is not None for dt in data_dir]
    assert sum(data_select_idx) == 1, print('Please choose only 1 target dataset!! Mark unwanted dataset path to NULL in config file')
    #print('model input size ', model_input_size, 'model ouput index ', model_output_index, 'teacher input size ', teacher_input_size, 'teacher output index ', teacher_output_index)
    print('model input size ', model_input_size)
    if model_output_size == 1:
        print('single output')
        frame_num, sal_indx = 16, list(range(15, 16))
        #x_indx, teacher_indx = range(0, 16), [range(0, 16)]
        #eval_steps = 1
    elif model_output_size == 16:
        print('16 outputs')
        frame_num, sal_indx = 16, list(range(0, 16))
        #eval_steps = 16
        #x_indx, teacher_indx = range(16, 32), [range(i, i+16) for i in range(16)]
    elif model_output_size == 8:
        print('8 output not yet implemented'); exit()
        #frame_num, sal_indx = 24, list(range(16, 24))
        #x_indx, teacher_indx = range(8, 24), [range(i, i+16) for i in range(8)]
    else:
        print('{} number of output is not supported yet!'.format(model_output_size))
        exit()

    if data_select_idx.index(True) == 0:
        print('generating DFH1K validation set saliency maps!')
        val_snpit = dhf1k.DHF1K('val', frame_num, 1, out_type=['vgg_in'], size=[(192, 256), (192, 256)],
                                sal_indx=sal_indx, inference_mode=True, frame_rate=None,
                                data_dir=data_dir[0])
    elif data_select_idx.index(True) == 1:
        print('generating UCF validation set saliency maps!')
        val_snpit = ucf.UCF('test', frame_num, 1, out_type=['vgg_in'], size=[(192, 256), (192, 256)],
                            sal_indx=sal_indx, inference_mode=True, frame_rate=None,
                            data_dir=data_dir[1])
    elif data_select_idx.index(True) == 2:
        print('Hollywood selected! Aux data selected! Kinetic400 is used! Create dataset now')
    
    eval_steps = sal_indx[-1]-sal_indx[0] + 1
    val_sampler = EvalClipSampler(val_snpit.clips, frame_num, step=eval_steps)
    dataloader = {'val': DataLoader(val_snpit, batch_size=batch_size, shuffle=False, num_workers=4, 
                                    pin_memory=True, sampler=val_sampler)
                 }
    
    return dataloader, frame_num, sal_indx, data_select_idx.index(True)


if __name__ == '__main__':
    import argparse
    import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to config file")
    args = parser.parse_args()
    config = cfg.get_config(args.config_path)
    #config = cfg.get_config('config/eval_config_single.yaml')
    #config = cfg.get_config('config/eval_config_multi.yaml')
    #config = cfg.get_config('config/eval_config_multi_ucf.yaml')
    #config = cfg.get_config('config/eval_config_multi_rc.yaml')

    batch_size = config.LEARNING_SETUP.BATCH_SIZE#15#9#15#18 #11
    save_path = config.LEARNING_SETUP.OUTPUT_PATH
    data_dir=[config.DATASET_SETUP.DHF1K_PATH, config.DATASET_SETUP.UCF_PATH, config.DATASET_SETUP.HOLLYWOOD_PATH]

    model_input_size = config.MODEL_SETUP.INPUT_SIZE
    model_output_size = config.MODEL_SETUP.OUTPUT_SIZE

    reduced_channel = config.MODEL_SETUP.CHANNEL_REDUCTION
    decoder_config = config.MODEL_SETUP.DECODER
    single_mode = config.MODEL_SETUP.SINGLE
    force_multi = config.MODEL_SETUP.FORCE_MULTI
    d1_last = config.MODEL_SETUP.D1_LAST
    model_path = config.MODEL_SETUP.MODEL_WEIGHTS

    data_loader, frame_num, sal_indx, data_type = config_dataset(data_dir, model_input_size, model_output_size)
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi, n_output=model_output_size)
    main(model, data_loader, sal_indx, model_path, save_path, data_type)