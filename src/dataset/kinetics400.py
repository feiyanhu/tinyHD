import os
import numpy as np
from torch.utils.data import Dataset
import torch

from .my_transform import resize_random_hflip_crop, video_S3D
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo
from torchvision import transforms as T
from .utils.video_clips import VideoClips

from PIL import Image

#PATH_kinetics400 = '/data/kinetics400/kinetics400'

class kinetics400(Dataset):
	def __init__(self, mode, window, step, out_type, size=[(192, 256), (192, 256)], inference_mode=False, frame_rate=None, data_dir=''):
		if 'train' in mode:mode = 'train'
		if 'val' in mode:mode = 'val'
		self.data_dir = data_dir + mode+'_256/'
		meta_path = 'dataset/metadata/kinetics400_{}.pkl'.format(mode)
		meta_data = self.load_video_meta(meta_path)
		meta_data['video_dir'] = data_dir + mode+'_256/'
		vid_list = self.get_video_list(mode)
		print(len(vid_list), len(meta_data['video_paths']))
		meta_data, vid_list = self.check_meta(meta_data, vid_list, window)
		print('after removing', len(vid_list))
		self.all_label = list(set([x.split('/')[-2] for x in vid_list]))
		self.all_label.sort()

		self.clips = VideoClips(vid_list, clip_length_in_frames=window, frames_between_clips=step, num_workers=4, 
								_precomputed_metadata=meta_data, frame_rate=frame_rate, _pts_unit="pts")
		if meta_data is None:
			self.save_video_meta(meta_path, self.clips.video_paths, self.clips.video_pts, self.clips.video_fps)

		self.to_s3d = T.Compose([video_S3D()])#to_s3d_tensor
		self.to_vgg = T.Compose([ToTensorVideo(), NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		
		self.aug = None
		if mode == 'train':
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0.5, random_crop=True, centre_crop=False, spatial_jitter=None)
		elif mode == 'val':
			self.aug = resize_random_hflip_crop(size[0], size[1], random_hflip=0, random_crop=False, centre_crop=True)
		
		self.out_type = out_type
		self.inference_mode = inference_mode
		self.size = size
		self.window = window
		self.read_video = True if 'vgg_in' in out_type or 's3d' in out_type or 'img' in out_type else False
		self.read_audio = True if 'audio' in out_type else False
		self.read_sal = True if 'sal' in out_type else False

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
		video_id = video_id.split('/')
		video_class = video_id[-2]
		video_id = video_id[-1]
		video_id = video_id.split('.')[0]
		clip_ids = clip_idxs + 1
		return video, audio, video_id, clip_ids, self.all_label.index(video_class)
	
	def __getitem__(self, item):
		vgg_data, audio_data, sal_data, o_size, s3d_data = (None, None, None, None, None)
		#sal_data = Image.fromarray(np.zeros((3 ,1 ,1)).astype(dtype=np.uint8))
		if self.read_video or self.read_audio:
			video, audio, video_id, clip_ids, video_class = self.__read_video_audio(item)
		
		data_list = []
		for out_type in self.out_type:
			if out_type == 'vgg_in':
				x = self.to_vgg(video)
			elif out_type == 's3d':
				x = self.to_s3d(video)
			elif out_type == 'sal':
				#x = sal_data#.unsqueeze(0)
				x = torch.zeros(1 , 1, data_list[0].shape[1], data_list[0].shape[2])
			elif out_type == 'img':
				x = video
			data_list.append(x)
		if self.aug is not None:
			data_list = self.aug(data_list)
		#return data_list, video_id, clip_ids, video_class, False
		return data_list, video_class, False, True

	#@staticmethod
	def get_video_list(self, mode):
		vid_list = []
		if mode == 'train':
			class_list = os.listdir(self.data_dir)
			for cls_id in class_list:
				tmp_vid = os.listdir(os.path.join(self.data_dir, cls_id))
				for tmp in tmp_vid:
					vid_list.append(os.path.join(self.data_dir, cls_id, tmp))
		elif mode == 'val':
			class_list = os.listdir(self.data_dir)
			for cls_id in class_list:
				tmp_vid = os.listdir(os.path.join(self.data_dir, cls_id))
				for tmp in tmp_vid:
					vid_list.append(os.path.join(self.data_dir, cls_id, tmp))
		return vid_list
			
	
	@staticmethod
	def load_video_meta(meta_path):
		import pickle
		if os.path.exists(meta_path):
			meta_data = pickle.load(open(meta_path, 'rb'))
		else:
			meta_data = None
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

	def check_meta(self, meta_data, video_list, window):
		def remove(idx):
			del meta_data['video_fps'][idx]
			del meta_data['video_pts'][idx]
			vid_id = meta_data['video_paths'].pop(idx)
			#print(meta_data['video_dir'], vid_id, video_list)
			video_list.remove(meta_data['video_dir']+vid_id)

		for i, x in enumerate(meta_data['video_fps']):
			if x is None:
				#print(i)
				remove(i)
			if len(meta_data['video_pts'][i])<window:
				#print(i)
				remove(i)
		return meta_data, video_list