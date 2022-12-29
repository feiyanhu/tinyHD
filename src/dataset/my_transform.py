from ctypes import resize
from torchvision import transforms
import torch
import random
from torchvision.transforms import functional as F
import torchvision.transforms.transforms as tv_tram


def to_s3d_tensor(clip):
	clip = clip.float().mul_(2.).sub_(255).div_(255).permute(3,0,1,2)
	return clip

class video_S3D(object):
	def __init__(self):
		pass
	def __call__(self, clip):
		clip = clip.float()
		clip = clip.mul_(2.).sub_(255).div(255)
		clip = clip.permute(3,0,1,2)
		return clip


class resize_random_hflip_crop(torch.nn.Module):
	def __init__(self, size, new_size, random_hflip=0, random_crop=None, centre_crop=None, spatial_jitter=None, resize_p=0):
		super().__init__()
		self.size = size
		self.output_size = new_size

		#random.random() generate [0, 1) p=0 no flip p=1 always flip
		self.resize_p = resize_p #p=1 always resize to outputsize
		self.random_hflip = random_hflip
		if random_crop:
			self.random_crop_params = tv_tram.RandomCrop
		else:
			self.random_crop_params = random_crop
		
		self.spatial_jitter = spatial_jitter
		self.centre_crop = centre_crop

	def forward(self, x):
		if self.spatial_jitter is not None:
			#print(x[0].shape)
			h, w = x[0].shape[2], x[0].shape[3]
			new_h = random.randint(self.spatial_jitter[0], self.spatial_jitter[1])
			new_w = int(new_h * (w/h))
			if new_h < 192 or new_w < 192 or new_h < 256 or new_w < 256:
				(new_h, new_w) = (192, 256)
			x = [F.resize(y, (new_h, new_w)) for y in x]
		else:
			if random.random() < self.resize_p:
				#print(self.output_size, 'herer1')
				x = [F.resize(y, self.output_size) for y in x]
			else:
				#print(self.size, 'herer2')
				x = [F.resize(y, self.size) for y in x]
		
		if random.random() < self.random_hflip:
			x = [F.hflip(y) for y in x]

		if self.random_crop_params:
			i, j, h, w = self.random_crop_params.get_params(x[0], self.output_size)
			x = [F.crop(y, i, j, h, w) for y in x]

		if self.centre_crop:
			x = [F.center_crop(y, self.output_size) for y in x]
		
		return x