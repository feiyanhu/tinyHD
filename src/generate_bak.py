import numpy as np
import torch as t
from torch.utils.data import DataLoader
import cv2
from os.path import exists
from os import mkdir

import dataset.DHF1K_vid as dhf1k
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

def padd_same(window, cid, vid_dir, img):
    if cid == window:
        for i in range(1, 16):
            file_name = '{}/{:04d}.png'.format(vid_dir, i)
            #print(file_name)
            cv2.imwrite(file_name, img)

def padd_append(window_size, cid, start_idx, newsize, vid_dir, model, vgg_in):
    if cid <= window_size:
        for i in range(1, cid):
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            img = y.detach().cpu().numpy()[0][0]
            img = post_process_png(img, newsize[[1, 0]])
            #for i in range(len_dict[int(vid)] - 14, len_dict[int(vid)]+1):
            file_name = '{}/{:04d}.png'.format(vid_dir, i)
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
        y = model(vgg_in)[0]#.unsqueeze(1)
        y = y.detach().cpu().numpy()
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #continue
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
            vid_dir = '{}{:04d}'.format(save_path, int(vid))
            if not exists(vid_dir):
                mkdir(vid_dir)
                print(vid_dir)
            #print(imgs.shape, cids)
            for img, cid in zip(imgs, cids):
                #print(img.shape, cid)
                img = post_process_png(img, newsize[[1, 0]])
                file_name = '{}/{:04d}.png'.format(vid_dir, cid)
                #print(img.shape)
                #exit()
                cv2.imwrite(file_name, img)
                #padd_same(16, cid, vid_dir, img)
                padd_append(16, cid, start_idx, newsize, vid_dir, model, vgg_in)

def test_one_multi(model, dataloader, save_path, output_idx):
    len_dict = vid_length(sal_indx)
    #print(len_dict)
    #exit()
    for i, (X, original_size, video_ids, clip_ids, has_label, rand_sig) in enumerate(dataloader):
        vgg_in = X[0].float().cuda(0)
        #with t.cuda.amp.autocast():
        y = model(vgg_in)[0].squeeze(1)
        y = y.detach().cpu().numpy()
        #print(y.shape)
        #exit()
        y = y[:, sal_indx, :, :]
        #print(vgg_in.shape, y.shape, video_ids, clip_ids)
        #exit()
        for imgs, vid, cids, newsize, start_idx in zip(y, video_ids, clip_ids, original_size, range(len(y))):
            newsize, cids = newsize.numpy(), cids.numpy()
            vid_dir = '{}{:04d}'.format(save_path, int(vid))
            if not exists(vid_dir):
                mkdir(vid_dir)
                print(vid_dir)
            #print(imgs.shape, cids)
            for img, cid in zip(imgs, cids):
                #print(img.shape, cid)
                img = post_process_png(img, newsize[[1, 0]])
                file_name = '{}/{:04d}.png'.format(vid_dir, cid)
                #print(img.shape)
                #exit()
                cv2.imwrite(file_name, img)
                padd_append_multi(16, cid, start_idx, newsize, vid_dir, model, vgg_in, sal_indx)
                #print(len_dict[int(vid)], cid)
                if cid == len_dict[int(vid)]:
                    padd_same_end(16, cid, vid_dir, img, sal_indx)
                #exit()

def main(model, single_mode, batch_size, frame_num, sal_indx, output_idx, pretrain_path, save_path, data_dir, force_multi):
    val_snpit = dhf1k.DHF1K('val', frame_num, 1, out_type=['vgg_in'], size=[(192, 256), (192, 256)],
                            sal_indx=sal_indx, inference_mode=True, frame_rate=None,
                            data_dir=data_dir)

    if pretrain_path:
        state_dict = t.load(pretrain_path, map_location='cuda:0')['student_model']
    else:
        print('please specify trained models.')
        exit()

    model.load_state_dict(state_dict)
    model.cuda(0)

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
            test_one_multi(model, dataloader['val'], save_path, output_idx)
    print('--------------------------------------------->>>>>>')
    #print('loss val {}'.format(loss_val))


if __name__ == '__main__':
    #dhf1k_path = '/data/DHF1K/'

    #channel reduce
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc2_e1d3_noaug/ft_199_0.40973_1.49512.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc4_e1d3_noaug/ft_198_0.44849_1.56443.pth'

    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc2_e1d3_3dd/ft_192_0.44286_1.54738.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc4_e1d3_3dd/ft_196_0.49398_1.58137.pth'

    #MISO
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_noaug/ft_199_0.35054_1.46873.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_unet/ft_157_0.33932_1.47576.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_dla/ft_167_0.33983_1.47756.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_unetdla/ft_162_0.38286_1.48793.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_d1/ft_163_0.31948_1.54798.pth'

    #MIMO
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd/ft_142_0.39578_1.49902.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d1d2/ft_180_0.40766_1.50889.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d1d3/ft_154_0.39163_1.53808.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d2d3/ft_143_0.45504_1.53670.pth'

    #label only on insight-server
    #model_path = '.../../SalGradNet_distillation_server2/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d1d2d3/ft_196_0.38547_1.50119.pth'
    #teacher only on insight-server
    #model_path = '.../../SalGradNet_distillation_server2/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d1d2d3/ft_196_0.38547_1.50119.pth'

    #generating multi outputs
    
    '''
    model_path = '../pretrained/d1d2d3_M_lt.pth'
    data_dir='/home/feiyan/data/DHF1K/'
    batch_size = 30
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, False, False]
    d1_last = True
    frame_num, sal_indx = 16, list(range(0, 16))
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last)

    main(model, single_mode, batch_size, frame_num, sal_indx, model_path, save_path, data_dir)
    '''
    
    #generating single outputs
    '''
    model_path = '../pretrained/d1d2d3_S_lt.pth'
    data_dir='/home/feiyan/data/DHF1K/'
    batch_size = 30
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, True, True]
    d1_last = False
    frame_num, sal_indx = 16, list(range(15, 16))
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last)

    main(model, single_mode, batch_size, frame_num, sal_indx, model_path, save_path, data_dir)
    '''

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s/ft_199_0.33540_1.51855.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s/ft_181_0.32628_1.52257.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s/ft_191_0.38075_1.49977.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s/ft_188_0.37607_1.5001.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s/ft_190_0.38281_1.51521.pth'

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d1s/ft_198_0.33885_1.51840.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d1s/ft_132_0.33642_1.50926.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d2s/ft_197_0.38650_1.49708.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d2s/ft_176_0.38126_1.49932.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s_d3s/ft_197_0.37937_1.49973.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s_d3s/ft_180_0.37685_1.49964.pth' #new

    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_unet/ft_157_0.33932_1.47576.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_dla/ft_167_0.33983_1.47756.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_unetdla/ft_162_0.38286_1.48793.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d3s/ft_197_0.38767_1.50771.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d3s/ft_170_0.38592_1.50621.pth' #new

    #model_path = '../pretrained/d1d2d3_S_lt.pth'

    #-----------------------------------------------------------------------

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf/ft_186_0.39853_1.55912.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf/ft_123_0.39166_1.56105.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1m/ft_178_0.37466_1.56646.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1m/ft_164_0.37195_1.55847.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m/ft_193_0.44210_1.51686.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m/ft_168_0.43620_1.51993.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m/ft_199_0.42919_1.52106.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m/ft_161_0.41879_1.52710.pth' #new

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf_d1sf/ft_197_0.39860_1.54491.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1sf_d1sf/ft_158_0.39358_1.54628.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m_d2m/ft_136_0.43204_1.52385.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m_d2m/ft_146_0.42338_1.52497.pth' #new
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m_d3m/ft_199_0.43429_1.53756.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3m_d3m/ft_174_0.42808_1.53989.pth' #new

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d2m_nev/ft_187_0.39735_1.48858.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d3m_nev/ft_188_0.38703_1.50238.pth'
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d2d3/ft_143_0.45504_1.53670.pth'

    #model_path = '../pretrained/d1d2d3_M_lt.pth'
    #model_path = '../../SalGradNet_distillation_server2/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd/ft_142_0.39578_1.49902.pth' #new
    
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2m_d2m_d2m/ft_145_0.46046_1.52945.pth' #0.90195161 0.82801144 0.47785937 2.73243305 0.37226832
    model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d2_noaug_dla/ft_167_0.33983_1.47756.pth'
    
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2m_d2m_d2m/ft_186_0.43762_1.51910.pth' #0.90522346 0.82586126 0.4841526  2.77675994 0.37733681
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3m_d3m_d3m/ft_164_0.43230_1.52388.pth' #0.90280562 0.82208363 0.47685767 2.7344885  0.37562613
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d1s_d1s/ft_150_0.34033_1.49460.pth' #0.90131676 0.82526991 0.49224418 2.84195852 0.392363
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2s_d2s_d2s/ft_158_0.39658_1.50825.pth' #0.90473893 0.82440241 0.4846678  2.80297054 0.37988547
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2s_d2s_d2s/ft_169_0.38436_1.51231.pth' #0.90491552 0.82656973 0.48474312 2.80423363 0.37741301
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3s_d3s_d3s/ft_160_0.38480_1.49456.pth' #0.90473397 0.82424317 0.4845227  2.79665658 0.37992114
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3s_d3s_d3s/ft_161_0.37882_1.49829.pth'
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1sf_d1sf_d1sf/ft_189_0.38811_1.53651.pth'
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1T/ft_154_0.18415_1.52582.pth' #0.90207978 0.83072765 0.47175678 2.67264573 0.36297555
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1T/ft_164_0.18235_1.52544.pth' #0.90246495 0.83173854 0.47187858 2.67250979 0.36194539
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1Tonly/ft_179_0.17934_1.51203.pth' #0.90346518 0.82980955 0.4719581  2.67842854 0.36326053
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1Tonly/ft_185_0.18277_1.51083.pth' #0.90331696 0.82936963 0.47182399 2.67668925 0.36362888
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1Tonly/ft_194_0.17980_1.51147.pth' #
    model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_2fuse2/ft_170_0.14539_1.52672.pth' #
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc4_2fuse2/ft_189_0.18563_1.56735.pth' #0.89871402 0.83410443 0.45961535 2.58729093 0.34855441
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc4_rc1T/ft_128_0.22907_1.55651.pth' #0.89991784 0.83325807 0.45640186 2.55812889 0.34780966
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc4_rc1T/ft_178_0.22940_1.55596.pth'
    
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d2m_d3m_8out/ft_154_0.35193_1.48697.pth' #0.90740289 0.82895125 0.49324472 2.85568919 0.38611484
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d2m_d3m_8out/ft_123_0.36822_1.48259.pth' #0.9073162  0.82587103 0.49614876 2.86858675 0.39135761
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d2m_8out/ft_191_0.35439_1.48720.pth'
    
    #model_path = '/home/feiyan/unict_weights/dhf1k_kinetic_lt_myschedule_e1_d2m_d2m_d2m/ft_199_0.43450_1.53482.pth' #0.90358207 0.82383532 0.48167721 2.76537248 0.37818701
    #model_path = '/home/feiyan/unict_weights/dhf1k_kinetic_lt_myschedule_e1_d3m_d3m_d3m/ft_197_0.43928_1.52666.pth' #0.90289371 0.82303851 0.4789098  2.75171076 0.37395589

    #model_path = '../pretrained/d1d2d3_M_lt.pth'
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_3fuse/ft_154_0.25509_1.43703.pth' #0.90740819 0.84147396 0.51121244 2.92316497 0.40349776
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_3fuse/ft_184_0.25555_1.43641.pth'
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_3fuse/ft_179_0.27651_1.50404.pth'
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_3fuse_dla/ft_192_0.24020_1.46483.pth' #0.90725835 0.83887497 0.50482167 2.85414801 0.40470345
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_3fuse_dla/ft_151_0.24402_1.46247.pth'
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_3fuse/ft_181_0.28447_1.48403.pth' #[0.90707235 0.84165899 0.49976776 2.81868774 0.39809402]
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_vinet/ft_196_0.31956_1.42004.pth' #0.90728119 0.84998196 0.49364658 2.76337746 0.36668876
    #model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_tased/ft_185_0.38896_1.42080.pth' #0.90809791 0.85194502 0.49742036 2.77508049 0.37052805
    #model_path = '/home/feiyan/runs/new_dhf1k_kinetic_lt_myschedule_e1_d123s/ft_193_0.36545_1.45456.pth' #0.85755682 0.74450694 0.3584435  2.03097509 0.3063572
    model_path = '/home/feiyan/runs/new_dhf1k_kinetic_lt_myschedule_e1_d123s_2hd2split//ft_196_0.37962_1.37775.pth' #
    #model_path = '/home/feiyan/runs/new_dhf1k_kinetic_lt_myschedule_e1_d333s/ft_175_0.41769_1.52025.pth' #0.85667393 0.74303493 0.35356639 1.99231992 0.30232279

    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d1md2md3m/ft_150_0.38638_1.57923.pth' #0.90533659 0.824527   0.48547307 2.80676332 0.38184807
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d1md2md3m_3/ft_183_0.36912_1.49892.pth' #[0.90681264 0.82882621 0.49038684 2.8349491  0.3832103 ]
    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d1md2md3m_3/ft_197_0.37210_1.49874.pth'
    #########################################################################
    #model_path = '../dhf1k_l_myschedule_e1_d123s/ft_150_0.00000_1.47729.pth' #??? [0.90290433 0.70071454 0.37866744]
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_noaug_lonly/ft_185_0.00000_1.46667.pth' #[0.90104332 0.68931009 2.73302048 0.38187308]
    #model_path = '../dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_noaug_lonly/ft_180_0.00000_1.48809.pth' #[0.90316664 0.6968031  2.76730137 0.376393  ] mixed [0.90316194 0.69666393 2.76719876 0.3763757 ]

    #model_path = '../pretrained/d1d2d3_M_lt.pth'

    model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s/ft_199_0.33540_1.51855.pth' #[0.89930011 0.71382875 0.39390699]
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s/ft_191_0.38075_1.49977.pth' #[0.90396785 0.70832649 0.38203093]
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s/ft_190_0.38281_1.51521.pth'

    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d1s_d1s/ft_198_0.33885_1.51840.pth' #here
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d2s_d2s/ft_197_0.38650_1.49708.pth'
    #model_path = '../dhf1k_kinetic_lt_myschedule_e1_d3s_d3s/ft_197_0.37937_1.49973.pth'

    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d1s_d1s_d1s/ft_150_0.34033_1.49460.pth' #0.90131676 0.82526991 0.49224418 2.84195852 0.392363
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d2s_d2s_d2s/ft_169_0.38436_1.51231.pth' #0.90491552 0.82656973 0.48474312 2.80423363 0.37741301
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d3s_d3s_d3s/ft_160_0.38480_1.49456.pth' #0.90473397 0.82424317 0.4845227  2.79665658 0.37992114
    
    #model_path = '../../SalGradNet_distillation/dhf1k_kinetic_v4_s3d_label_oldschedule_rc2_e1d3_noaug/ft_199_0.40973_1.49512.pth' #[0.90381259 0.70626649 0.36407852]
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc2_rc1T/ft_154_0.18415_1.52582.pth' #0.90207978 0.83072765 0.47175678 2.67264573 0.36297555
    #model_path = '/home/runs/dhf1k_kinetic_lt_myschedule_e1_d123m_rc4_rc1T/ft_128_0.22907_1.55651.pth' #0.89991784 0.83325807 0.45640186 2.55812889 0.34780966

    model_path = '/home/feiyan/runs/dhf1k_kinetic_lt_myschedule_e1_d123s_tased/ft_185_0.38896_1.42080.pth' #0.90809791 0.85194502 0.49742036 2.77508049 0.37052805
    save_path = '/home/test_generate_tmp/'
    #save_path = '/home/feiyan/test_generate/'
    #save_path = '/home/feiyan/runs/test_generate_d123m/'
    #save_path = '/home/feiyan/runs/test_generate_d1s/'
    save_path = '/home/feiyan/runs/test_generate_tased_teacher/'
    
    data_dir='/home/feiyan/data/DHF1K/'
    batch_size = 50
    reduced_channel = 1 #can only be 1, 2, 4
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, True, True]
    #decoder_config = ['d1', 'd2', 'd3']
    #single_mode = [True, True, True]
    #single_mode = [False, False, False]
    #single_mode = [True, False, False]
    force_multi = False
    d1_last = False
    n_output = 16
    frame_num, sal_indx = 16, list(range(15, 16))
    #frame_num, sal_indx = 16, list(range(8, 16)) #need to update main func
    #frame_num, sal_indx = 16, list(range(9, 12)) #need to update main func
    #frame_num, sal_indx = 16, list(range(0, 16)) #16 out
    
    output_idx = range(0, 16)

    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi, n_output=n_output)
    #from utils import eval
    #eval(model)
    #exit()
    main(model, single_mode, batch_size, frame_num, sal_indx, output_idx, model_path, save_path, data_dir, force_multi)
