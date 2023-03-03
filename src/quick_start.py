import numpy as np
import torch as t
from torch.utils.data import Dataset, DataLoader
import cv2
from os.path import exists
from os import mkdir

import os

from models.fastsal3D.model import FastSalA
from my_sampler import EvalClipSampler
from dataset.utils.video_clips import VideoClips
from torchvision import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo
from dataset.my_transform import resize_random_hflip_crop
from generate import post_process_png

def get_config_path(w_path):
    if 'd1d2d3_S_lt.pth' in w_path:
        return 'config/eval_config_single.yaml', 1, 'single'
    elif 'd1d2d3_M_lt.pth' in w_path:
        return 'config/eval_config_multi.yaml', 1, 'multi'
    elif 'ucf_d123s.pth' in w_path:
        return 'config/eval_config_single.yaml', 1, 'single'
    elif 'ucf_d123m.pth' in w_path:
        return 'config/eval_config_multi_ucf.yaml', 1, 'multi'
    elif 'd123s_rc2_rc1T.pth' in w_path:
        return 'config/eval_config_single.yaml', 2, 'single'
    elif 'd123s_rc4_rc1T.pth' in w_path:
        return 'config/eval_config_single.yaml', 4, 'single'
    elif 'd123m_rc2_rc1T.pth' in w_path:
        return 'config/eval_config_multi_rc.yaml', 2, 'multi'
    elif 'd123m_rc4_rc1T.pth' in w_path:
        return 'config/eval_config_multi_rc.yaml', 4, 'multi'
    else:
        exit('not implemented')

def padd_append(window_size, cid, start_idx, model, vgg_in):
    sal_indx = list(range(15, 16))
    #print(list(range(1, cid, sal_indx[-1] - sal_indx[0]+1)))
    if cid <= window_size:
        for i in range(1, cid, sal_indx[-1] - sal_indx[0]+1):
            tmp = window_size - i
            tmp = t.stack([vgg_in[start_idx:start_idx+1, :, 0, :, :] for _ in range(tmp)], dim=2)
            tmp = t.cat([tmp, vgg_in[start_idx:start_idx+1, :, :i, :, :]], dim=2)
            y = model(tmp)[0]
            img = y.detach().cpu().numpy()[0][0]
            yield img, i-1

class Video_reader(Dataset):
    def __init__(self, v_path, w, s):
        v_id = v_path.split('/')[-1]
        v_dir = '/'.join(v_path.split('/')[:-1])
        self.vr = VideoClips([v_path], clip_length_in_frames=w, frames_between_clips=s)
        self.vr.video_dir = v_dir
        #print(self.vr.video_paths)
        self.vr.video_paths = [v_id]
        self.to_vgg = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.aug = resize_random_hflip_crop((192, 256), (192, 256), random_hflip=0, random_crop=False, centre_crop=False)
    
    def __getitem__(self, item):
        v_dt, a_dt, info, v_idx, clip_pts = self.vr.get_clip(item)
        o_size = np.asarray(list(v_dt.shape[1:-1]))
        v_dt = self.to_vgg(v_dt)
        [data_v] = self.aug([v_dt])
        #print(clip_pts)
        #exit()
        return data_v, clip_pts, o_size
    
    def __len__(self):
        return len(self.vr)

def predict_single(dataloader, model, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for x, idx, o_sizes in dataloader:
        pred_maps = model(x.cuda())[0]
        pred_maps = pred_maps.detach().cpu().numpy()
        idx = idx.numpy()
        o_sizes = o_sizes.numpy()
        for imgs, f_ids, o_size, start_idx in zip(pred_maps, idx[:, -1], o_sizes, range(len(pred_maps))):
            print(imgs.shape, f_ids, o_size)
            img = post_process_png(imgs[0], o_size[[1, 0]])
            file_name = os.path.join(save_dir, '{}.png'.format(f_ids))
            cv2.imwrite(file_name, img)
            for img_ext, i_ext in padd_append(16, f_ids+1, start_idx, model, x.cuda()):
                print(img_ext.shape, i_ext)
                img_ext = post_process_png(img_ext, o_size[[1, 0]])
                file_name = os.path.join(save_dir, '{}.png'.format(i_ext))
                cv2.imwrite(file_name, img)

def predict_multi(dataloader, model, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for x, idx, o_sizes in dataloader:
        pred_maps = model(x.cuda())[0].squeeze(1)
        pred_maps = pred_maps.detach().cpu().numpy()
        #o_sizes = o_sizes.numpy()
        idx = idx.numpy()
        print(pred_maps.shape, idx.shape, o_sizes.shape)
        for imgs, f_ids, o_size in zip(pred_maps, idx, o_sizes):
            o_size = o_size.numpy()
            for img, f_id in zip(imgs, f_ids):
                img = post_process_png(img, o_size[[1, 0]])
                file_name = os.path.join(save_dir, '{}.png'.format(f_id))
                cv2.imwrite(file_name, img)

def get_video_reader(v_path, mode):
    v = Video_reader(v_path, 16, 1)

    if mode == 'single':
        val_sampler = EvalClipSampler(v.vr, 16, step=1)
    elif mode == 'multi':
        val_sampler = EvalClipSampler(v.vr, 16, step=16)
    dataloader = DataLoader(v, batch_size=20, shuffle=False, num_workers=4,pin_memory=True, sampler=val_sampler)
    
    v_id = v_path.split('/')[-1]
    v_id = v_id.replace('.', '_')
    return dataloader, v_id
def main(model, pretrain_path, dataloader, mode, save_dir):
    if pretrain_path:
        state_dict = t.load(pretrain_path, map_location='cuda:0')['student_model']
    else:
        print('please specify trained models.')
        exit()

    model.load_state_dict(state_dict)
    model.cuda(0)

    model.eval()
    if mode == 'single':
        with t.no_grad():
            predict_single(dataloader, model, save_dir)
    elif mode == 'multi':
        with t.no_grad():
            predict_multi(dataloader, model, save_dir)
    exit('loaded')
    
if __name__ == '__main__':
    import argparse
    import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("weight_path", help="path to weights")
    parser.add_argument("video_path", help="path to a video")
    parser.add_argument("save_path", help="path to save salieny maps")

    args = parser.parse_args()
    w_path = args.weight_path
    v_path = args.video_path
    save_path = args.save_path

    config_path, reduced_channel, mode = get_config_path(w_path)
    config = cfg.get_config(config_path)

    batch_size = config.LEARNING_SETUP.BATCH_SIZE#15#9#15#18 #11
    #save_path = config.LEARNING_SETUP.OUTPUT_PATH
    data_dir=[config.DATASET_SETUP.DHF1K_PATH, config.DATASET_SETUP.UCF_PATH, config.DATASET_SETUP.HOLLYWOOD_PATH]

    model_input_size = config.MODEL_SETUP.INPUT_SIZE
    model_output_size = config.MODEL_SETUP.OUTPUT_SIZE

    decoder_config = config.MODEL_SETUP.DECODER
    single_mode = config.MODEL_SETUP.SINGLE
    force_multi = config.MODEL_SETUP.FORCE_MULTI
    d1_last = config.MODEL_SETUP.D1_LAST

    print(reduced_channel)
    #data_loader, frame_num, sal_indx, data_type = config_dataset(data_dir, model_input_size, model_output_size)
    model = FastSalA(reduced_channel, decoder_config, single_mode, d1_last=d1_last, force_multi=force_multi, n_output=model_output_size)
    dl, v_id = get_video_reader(v_path, mode)
    save_dir = os.path.join(save_path, v_id)
    main(model, w_path, dl, mode, save_dir)