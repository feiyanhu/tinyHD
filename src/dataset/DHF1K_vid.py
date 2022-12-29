from hashlib import new
import os

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
from .utils.video_clips import VideoClips
import random
#import librosa

#PATH_DHF1K = '/home/feiyan/data/DHF1K/'
#frame_direc = 'frames'

def _read_img(path, start_indx, end_indx, size=None):
	all_img = []
	for i in range(start_indx, end_indx):
		img_path = os.path.join(path, '{0:04d}.png'.format(i+1))
		img, o_size = read_cv_img(img_path, size)
		all_img.append(img)
	return np.asarray(all_img), o_size

def _read_sal(path, id_list, size=None):
	all_img = []
	for i in id_list:
		img_path = os.path.join(path, '{0:04d}.png'.format(i))
		img, o_size = read_saliency(img_path, size)
		all_img.append(img)
	return torch.from_numpy(np.asarray(all_img)/255.0), o_size

class DHF1K(Dataset):
	def __init__(self, mode, window, step, out_type, size=[(192, 256), (192, 256)], 
				 sal_indx=[-1], inference_mode=False, frame_rate=None,
				 resize_p=0, data_dir=''):
		self.data_dir = data_dir
		meta_path = 'dataset/metadata/dhf1k_{}.pkl'.format(mode)
		self.mode = mode
		if mode == 'train_' or mode == 'val_':
			meta_path = 'dataset/metadata/dhf1k_{}.pkl'.format('train')
		#len_dict = self.generate_video_meta(meta_path)
		vid_list = self.get_video_list(mode)
		meta_data = self.load_video_meta(meta_path, vid_list)
		meta_data['video_dir'] = data_dir
		#print(vid_list)
		#print(meta_data)
		#exit()
		#len_list = self.get_video_meta(vid_list, len_dict=len_dict)
		#self.clips = VideoClips(vid_list, len_list, window, step, size, every_n_skip, drop_last)
		self.clips = VideoClips(vid_list, clip_length_in_frames=window, frames_between_clips=step, num_workers=4, _precomputed_metadata=meta_data,
								frame_rate=frame_rate, _pts_unit="pts")
		if meta_data is None:
			self.save_video_meta(meta_path, self.clips.video_paths, self.clips.video_pts, self.clips.video_fps)

		self.to_s3d = T.Compose([video_S3D()])#to_s3d_tensor
		self.to_vgg = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		
		if mode == 'train' or mode == 'train_':
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0.5, random_crop=True, centre_crop=False, spatial_jitter=None, resize_p=resize_p)
		elif mode == 'val' or mode == 'val_':
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=True)
			if inference_mode:
				self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=False)
		elif mode == 'test':
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=False)

		self.out_type = out_type
		self.inference_mode = inference_mode
		self.size = size
		self.window = window
		self.read_video = True if 'vgg_in' in out_type or 's3d' in out_type or 'img' in out_type else False
		self.read_audio = True if 'audio' in out_type else False
		self.read_sal = True if 'sal' in out_type else False
		self.sal_indx = sal_indx
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
		#print(video.shape, audio.shape, video_id, clip_ids)
		#exit()
		sal_clip_ids = clip_ids[self.sal_indx]

		if self.read_sal:
			sal_data, _ = _read_sal(os.path.join(self.data_dir, 'annotation', '{0:04d}'.format(int(video_id)), 'maps'), 
									sal_clip_ids, None)
		#if self.read_audio:
		#	audio_data = torch.stack([compute_log_mel(a) for a in audio])

		data_list = []
		for out_type in self.out_type:
			if out_type == 'vgg_in':
				x = self.to_vgg(video)
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
			o_size = np.asarray((360, 640))
		
		has_label = True
		if self.inference_mode:
			return data_list, o_size, video_id, sal_clip_ids, has_label, 0
		return data_list, -1, True, False

	@staticmethod
	def get_video_list(mode):
		np.random.seed(seed=10)
		arr = np.arange(600)
		np.random.shuffle(arr)
		if mode == 'train':
			vid_list = range(600)
		#if mode == 'train_':
		#	vid_list = arr[:540]
		#	#print('dataset/metadata/dhf1k_train_.txt')
		#	with open('dataset/metadata/dhf1k_train_.txt') as file: lines = [line.rstrip() for line in file]
		#	vid_list = [int(tmp)for tmp in lines]
		elif mode == 'val':
			vid_list = range(600, 700)
		#elif mode == 'val_':
		#	vid_list = arr[540:]
		#	with open('dataset/metadata/dhf1k_val_.txt') as file: lines = [line.rstrip() for line in file]
		#	vid_list = [int(tmp)for tmp in lines]
		elif mode == 'test':
			vid_list = range(700, 1000)
			#vid_list = [991, 992, 993]
		return ['/home/feiyan/data/DHF1K/video/{0:03d}.AVI'.format(video_ID+1) for video_ID in vid_list]
			
	@staticmethod
	def generate_video_meta(meta_path):
		import pickle
		meta_path = os.path.join(meta_path, 'dhf1k.pkl')
		if os.path.exists(meta_path):
			len_dict = pickle.load(open(meta_path, 'rb'))
		else:
			all_name = []
			for x in ['train', 'val', 'test']:
				all_name.extend(DHF1K.get_video_list(x))
			len_dict = DHF1K.get_video_meta(all_name)
			pickle.dump(len_dict, open(meta_path, 'wb'))
		return len_dict
	
	@staticmethod
	def load_video_meta(meta_path, video_list):
		import pickle
		if os.path.exists(meta_path):
			meta_data = pickle.load(open(meta_path, 'rb'))
			new_path, new_pts, new_fps = list(), list(), list()
			video_list = [x.split('/')[-1] for x in video_list]
			#print(video_list)
			#exit()
			for x, y, z in zip(meta_data['video_paths'], meta_data["video_pts"], meta_data["video_fps"]):
				#print(x)
				if x.split('/')[-1] in video_list:
					new_path.append(x)
					new_pts.append(y)
					new_fps.append(z)
			#print(new_path)
			#exit()
			meta_data = {
            	"video_paths": new_path,
            	"video_pts": new_pts,
            	"video_fps": new_fps,
        	}
		else:
			meta_data = None
			#meta_data = None
		return meta_data
	
	@staticmethod
	def save_video_meta(meta_path, video_paths, video_pts, video_fps):
		import pickle
		metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
		pickle.dump(metadata, open(meta_path, 'wb'))

			


if __name__ == '__main__':
	from torch.utils.data import DataLoader
