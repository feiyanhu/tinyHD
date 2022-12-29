import os
from .read_utils import read_saliency
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from .DHF1K_vid import DHF1K
from .kinetics400 import kinetics400

from .my_transform import resize_random_hflip_crop, to_s3d_tensor, video_S3D
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo
from torchvision import transforms as T


#PATH_DHF1K = '/data/DHF1K/'
#PATH_kinetics400 = '/data/kinetics400/kinetics400'
#frame_direc = 'frames'

def _read_sal(path, start_indx, end_indx, size=None):
	all_img = []
	for i in range(start_indx, end_indx):
		img_path = os.path.join(path, '{0:04d}.png'.format(i+1))
		img, o_size = read_saliency(img_path, size)
		all_img.append(img)
	return np.asarray(all_img), o_size

class DHF1K_kinetics400(Dataset):
	def __init__(self, mode, window, step, out_type, size=[(192, 256), (192, 256)], sal_indx=[-1], inference_mode=False, frame_rate=None, data_dir=['', '']):
		self.dhf1k_dataset = DHF1K(mode, window, step, ['img', 'sal'], sal_indx=sal_indx, frame_rate=frame_rate, data_dir=data_dir[0])
		#print(len(self.dhf1k_dataset))
		#exit()
		self.kinetics400_dataset = kinetics400(mode, window, step, ['img'], frame_rate=frame_rate, data_dir=data_dir[1])	
		self.dhf1k_dataset.aug = None
		self.kinetics400_dataset.aug = None
		
		self.to_s3d = T.Compose([video_S3D()])#to_s3d_tensor
		self.to_vgg = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		
		#(200, 355)
		self.aug2 = None
		if mode == 'train' or mode == 'train_':
			#if mode == 'train' or 'train_':
			#print('here train')
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0.5, random_crop=True, centre_crop=False, spatial_jitter=None)
			#self.aug2 = resize_random_hflip_crop((200, 355), (192, 256), random_hflip=True, random_crop=True, centre_crop=False, spatial_jitter=None)
		elif mode == 'val' or mode == 'val_':
			#elif mode == 'val' or 'val_':
			#print('here val')
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=True)
			if inference_mode:
				self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=False)
		#exit()
		self.out_type = out_type
		self.inference_mode = inference_mode
		self.size = size
		self.window = window
		self.read_video = True if 'vgg_in' in out_type or 's3d' in out_type or 'img' in out_type else False
		self.read_audio = True if 'audio' in out_type else False
		self.read_sal = True if 'sal' in out_type else False
		self.sal_indx = sal_indx

	def __len__(self):
		return len(self.dhf1k_dataset) + len(self.kinetics400_dataset)
	
	def __getitem__(self, item):		
		if item < len(self.dhf1k_dataset):
			#data_list, o_size, video_id, sal_clip_ids, has_sal, _ = self.dhf1k_dataset[item]
			data_list, _, has_sal, _ = self.dhf1k_dataset[item]
			video_class = -1
			has_cls = False
		elif item >= len(self.dhf1k_dataset):
			item_k = item - len(self.dhf1k_dataset)
			#data_list, video_id, clip_ids, video_class, has_sal = self.kinetics400_dataset[item_k]
			data_list, video_class, has_sal, _ = self.kinetics400_dataset[item_k]
			has_cls = True
			data_list.append(torch.zeros(1 , len(self.sal_indx), data_list[0].shape[1], data_list[0].shape[2]))
		
		data_list_new = []
		for out_type in self.out_type:
			if out_type == 'vgg_in':
				x = self.to_vgg(data_list[0])
			elif out_type == 's3d':
				x = self.to_s3d(data_list[0])
			elif out_type == 'sal':
				x = data_list[1]
			data_list_new.append(x)
		if self.aug is not None:
			data_list = self.aug(data_list_new)
			#print(data_list[0].shape, data_list[1].shape, video_class, has_sal, has_cls, item)
		if self.aug2 is not None:
			data_list.extend(self.aug2(data_list_new[0:1]))
		return data_list, video_class, has_sal, has_cls

			


if __name__ == '__main__':
	from torch.utils.data import DataLoader