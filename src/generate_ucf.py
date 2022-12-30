import numpy as np
import torch as t
from torch.utils.data import DataLoader
import cv2
from os.path import exists
from os import mkdir

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
        direc = '/home/feiyan/data/DHF1K/annotation/{:04d}/maps/'.format(vid+1)
        onlyfiles = [f for f in listdir(direc) if isfile(join(direc, f)) and '.png' in f]
        len_dict[vid+1] = len(onlyfiles) - (16 - sal_indx[-1]) + 1
    return len_dict

def padd_same(window, cid, vid_dir, tmp_name, img):
    if cid == window:
        for i in range(1, 16):
            #file_name = '{}/{:04d}.png'.format(vid_dir, i)
            file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, i)
            #print(file_name)
            cv2.imwrite(file_name, img)

def padd_append(window_size, cid, start_idx, newsize, vid_dir, tmp_name, model, vgg_in):
    if cid <= window_size:
        for i in range(1, cid):
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            img = y.detach().cpu().numpy()[0][0]
            img = post_process_png(img, newsize[[1, 0]])
            #for i in range(len_dict[int(vid)] - 14, len_dict[int(vid)]+1):
            #file_name = '{}/{:04d}.png'.format(vid_dir, i)
            file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, i)
            print(file_name)
            cv2.imwrite(file_name, img)

def padd_append_multi(window_size, cid, start_idx, newsize, vid_dir, tmp_name, model, vgg_in, sal_indx):
    if cid == sal_indx[0] + 1:
        for i in range(1, cid, sal_indx[-1] - sal_indx[0]+1):
            #print(i)
            #continue
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            imgs = y.detach().cpu().numpy()[0][0][sal_indx]
            #print(imgs.shape, range(i, i+sal_indx[-1] - sal_indx[0]+1), len(range(i, i+sal_indx[-1] - sal_indx[0]+1)))
            #exit()
            for img, j in zip(imgs, range(i, i+sal_indx[-1] - sal_indx[0]+1)):
                img = post_process_png(img, newsize[[1, 0]])
                #for i in range(len_dict[int(vid)] - 14, len_dict[int(vid)]+1):
                file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, j)
                #file_name = '{}/{:04d}.png'.format(vid_dir, j)
                print(file_name)
                cv2.imwrite(file_name, img)
        #exit()
def padd_same_end(window, cid, vid_dir, img, sal_indx):
    for i in range(cid+1, cid+(16 - sal_indx[-1])):
        #print(i)
        #continue
        file_name = '{}/{:04d}.png'.format(vid_dir, i)
        print(file_name)
        cv2.imwrite(file_name, img)
    #exit()


def test_one_single(model, dataloader, save_path):
    #len_dict = vid_length()
    for i, (X, original_size, video_ids, clip_ids, has_label, rand_sig) in enumerate(dataloader):
        vgg_in = X[0].float().cuda(0)
        #with t.cuda.amp.autocast():
        #print(vgg_in.shape)
        #exit()
        y = model(vgg_in)[0]#.unsqueeze(1)
        y = y.detach().cpu().numpy()
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #continue
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
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
                file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, cid)
                #print(img.shape)
                #exit()
                cv2.imwrite(file_name, img)
                #padd_same(16, cid, vid_dir, img)
                padd_append(16, cid, start_idx, newsize, vid_dir, tmp_name, model, vgg_in)

def test_one_multi(model, dataloader, save_path, sal_indx):
    #len_dict = vid_length(sal_indx)
    #print(len_dict)
    #exit()
    for i, (X, original_size, video_ids, clip_ids, has_label, rand_sig) in enumerate(dataloader):
        vgg_in = X[0].float().cuda(0)
        #with t.cuda.amp.autocast():
        y = model(vgg_in)[0].squeeze(1)
        y = y.detach().cpu().numpy()
        #print(y.shape)
        y = y[:, sal_indx, :, :]
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
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
                file_name = '{}/{}_{:03d}.png'.format(vid_dir, tmp_name, cid)
                #print(img.shape)
                #exit()
                cv2.imwrite(file_name, img)
                padd_append_multi(16, cid, start_idx, newsize, vid_dir, tmp_name, model, vgg_in, sal_indx)
                #print(len_dict[int(vid)], cid)
                #if cid == len_dict[int(vid)]:
                #    padd_same_end(16, cid, vid_dir, img, sal_indx)
                #exit()

def main(model, single_mode, batch_size, frame_num, sal_indx, pretrain_path, save_path, data_dir, force_multi):
    val_snpit = ucf.UCF('test', frame_num, 1, out_type=['vgg_in'], size=[(192, 256), (192, 256)],
                            sal_indx=sal_indx, inference_mode=True, frame_rate=None,
                            data_dir=data_dir)

    if pretrain_path:
        state_dict = t.load(pretrain_path, map_location='cuda:0')['student_model']
    else:
        print('please specify trained models.')
        exit()

    model.load_state_dict(state_dict)
    model.cuda()

    if len(single_mode) == np.sum(single_mode) and not force_multi:
        dataloader = {
            'val': DataLoader(val_snpit, batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
        }
    else:
        #print(sal_indx[-1]-sal_indx[0] + 1)
        #exit()
        val_sampler = EvalClipSampler(val_snpit.clips, frame_num, step=sal_indx[-1]-sal_indx[0] + 1)
        dataloader = {
            'val': DataLoader(val_snpit, batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)
        }
    print('--------------------------------------------->>>>>>')

    model.eval()
    with t.no_grad():
        if len(single_mode) == np.sum(single_mode) and not force_multi:
            test_one_single(model, dataloader['val'], save_path)
        else:
            test_one_multi(model, dataloader['val'], save_path, sal_indx)
    print('--------------------------------------------->>>>>>')
    #print('loss val {}'.format(loss_val))


if __name__ == '__main__':
    model_path = '/home/feiyan/runs/ucf_kinetic_lt_myschedule_e1_d1s_d2s_d3s/ft_197_0.40632_1.03739.pth'
    #model_path = '/home/runs/ucf_kinetic_lt_myschedule_e1_d1s_d2m_d3m/ft_193_0.43095_1.14993.pth'

    #model_path = '/home/feiyan/runs/ucf_kinetic_lt_myschedule_e1_d1s_d2s_d3_update/ft_163_0.45265_0.74858.pth'

    #save_path = '/home/test_generate_ucf/'
    save_path = '/home/feiyan/runs/test_generate_ucf_tmp/'
    data_dir='/home/feiyan/data/ucf_sport/'
    batch_size = 20
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, True, True]
    force_multi = False
    d1_last = False
    frame_num, sal_indx = 16, list(range(15, 16))
    #frame_num, sal_indx = 16, list(range(8, 16))
    #frame_num, sal_indx = 16, list(range(9, 12))
    #frame_num, sal_indx = 16, list(range(0, 16))
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi)

    main(model, single_mode, batch_size, frame_num, sal_indx, model_path, save_path, data_dir, force_multi)
