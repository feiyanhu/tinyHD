import os
from cv2 import imread

from numpy.core.fromnumeric import clip
from torchvision.transforms.functional import center_crop
#from .utils import turbo_read_bgr, turbo_read_sal
from .read_utils import read_cv_img, read_saliency
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision.transforms.transforms import Normalize

from .my_transform import resize_random_hflip_crop, to_s3d_tensor, video_S3D
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo
from torchvision import transforms as T
from .image_as_videoclips import VideoClips
#import librosa

PATH_ucf = '/data/ucf_sport/'
frame_direc = 'frames'

def _read_sal(path, id_list, size=None):
    all_img = []
    for i in id_list:
        img_path = path + '{0:03d}.png'.format(i)
        img, o_size = read_saliency(img_path, size)
        all_img.append(img)
    return torch.from_numpy(np.asarray(all_img)/255.0), o_size

class UCF(Dataset):
    def __init__(self, mode, window, step, out_type, size=[(192, 256),(192, 256)], sal_indx=[-1], inference_mode=False, frame_rate=None, data_dir=''):
        self.data_dir = data_dir
        vid_list = self.get_video_list(mode)
        meta_data = self.get_video_meta(vid_list)
        print(vid_list)
        print(meta_data)
        self.clips = VideoClips(vid_list, window, step, meta_data, size=None, every_n_skip=step)

        self.to_s3d = T.Compose([video_S3D()])#to_s3d_tensor
        self.to_vgg = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        #size = (224, 384)
        self.aug = None
        if mode == 'training' or mode == 'train_' or mode == 'train':
            self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0.5, random_crop=True, centre_crop=False, spatial_jitter=None)
        elif mode == 'testing' or mode == 'val_' or mode == 'val' or mode == 'test':
            self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=True)
            if inference_mode:
                self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=False)
                #self.aug = resize_random_hflip_crop((224, 384), (224, 384), random_hflip=False, random_crop=False, centre_crop=False)
        #import torchvision.transforms as tv_transforms
        #unisal_transform = [tv_transforms.ToPILImage(), tv_transforms.Resize((288, 384)), 
        #                    tv_transforms.ToTensor(), tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        #self.to_unisal = T.Compose(unisal_transform)
        self.out_type = out_type
        self.inference_mode = inference_mode
        self.size = size
        self.window = window
        self.read_video = True if 'vgg_in' in out_type or 's3d' in out_type or 'img' in out_type else False
        self.read_audio = True if 'audio' in out_type else False
        self.read_sal = True if 'sal' in out_type else False
        self.sal_indx = sal_indx
        if mode == 'train' or mode == 'train_' or mode == 'val_':
            self.mode_path = 'training'
        else:
            self.mode_path = 'testing'
        #print(self.read_video, self.read_audio, self.read_sal)
        #self.audio_sr = 44100

    def __len__(self):
        return len(self.clips)

    def __read_video_audio(self, item):
        if isinstance(item, list):
            video, audio, clip_idx = list(), list(), list()
            for itm in item:
                v, a, info, video_idx, c_idx = self.clips.get_clip(itm)
                video.append(v); audio.append(a); clip_idx.extend(c_idx.tolist())
            video = torch.cat(video, dim=0)
            audio = torch.cat(audio, dim=1)
        else:
            video, audio, info, video_idx, clip_idxs = self.clips.get_clip(item)
        video_id = self.clips.video_paths[video_idx]
        video_id = video_id.split('/')[-1]
        video_id = video_id.split('.')[0]
        clip_ids = clip_idxs + 1
        return video, audio, video_id, clip_ids
    
    def __getitem__(self, item):
        vgg_data, audio_data, sal_data, o_size, s3d_data = (None, None, None, None, None)
        if self.read_video or self.read_audio:
            video, audio, video_id, clip_ids = self.__read_video_audio(item)
        sal_clip_ids = clip_ids[self.sal_indx]
        o_size = np.asarray(list(video.shape[1:3]))
        #print(video.shape, video_id, clip_ids, sal_clip_ids, o_size)
        #exit()

        if self.read_sal:
            tmp_name = video_id[:-4] + '_' + video_id[-3:]
            sal_data, _ = _read_sal(os.path.join(self.data_dir, self.mode_path, video_id, 'maps', tmp_name+'_'), 
                                    sal_clip_ids, None)
        #if self.read_audio:
        #    audio_data = torch.stack([compute_log_mel(a) for a in audio])

        data_list = []
        for out_type in self.out_type:
            if out_type == 'vgg_in':
                x = self.to_vgg(video)
                #for y in video:print(y.shape)
                #x = torch.stack([self.to_unisal(y) for y in video])
                #print(x.shape)
            elif out_type == 's3d':
                x = self.to_s3d(video)
            elif out_type == 'sal':
                x = sal_data.unsqueeze(0)
            elif out_type == 'img':
                x = video
            data_list.append(x)
        
        if self.aug is not None:
            data_list = self.aug(data_list)
        if o_size is None:
            o_size = np.asarray((404, 720))
        
        has_label = True
        if self.inference_mode:
            return data_list, o_size, video_id, sal_clip_ids, has_label, 0
        return data_list, -1, True, False

    #@staticmethod
    def get_video_list(self, mode):
        if mode == 'train' or mode == 'test':
            if mode == 'train':path = '{}{}'.format(self.data_dir, 'training')
            else: path = '{}{}'.format(self.data_dir, 'testing')
            video_list = os.listdir(path)
            video_list = ['{}/{}'.format(path, v) for v in video_list]
        elif mode == 'train_' or mode == 'val_':
            with open('dataset/metadata/ucf_{}.txt'.format(mode)) as file: lines = [line.rstrip() for line in file]
            video_list = ['{}{}/{}'.format(self.data_dir, 'training', v) for v in lines]
        #print(video_list)
        #exit()
        return video_list
            
    @staticmethod
    def get_video_meta(video_list):
        video_pts = []
        for video_id in video_list:
            frame_list = [x for x in os.listdir('{}/images'.format(video_id)) if '.png' in x]
            new_frame_list = range(len(frame_list))
            video_pts.append(new_frame_list)
        metadata = {
            "video_paths": video_list,
            "video_pts": video_pts,
        }
        return metadata
