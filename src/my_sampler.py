from torch.utils.data import Sampler
import torch
import random

class TemporalSubsampler(Sampler):
	def __init__(self, video_clips, every_n_skip):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.every_n_skip = every_n_skip
	
	def __iter__(self):
		idxs = []
		s = 0
		for c in self.video_clips.nframe_idx_list:
			length = len(c)
			if length == 0:
				continue
			tmp_idx = range(s, s+length-1, self.every_n_skip)
			s += length
			idxs.extend(tmp_idx)
		return iter(idxs)
	
	def __len__(self):
		return sum([int(len(c)/self.every_n_skip) for c in self.video_clips.nframe_idx_list])

class UniformClipSampler(Sampler):
	def __init__(self, video_clips, num_clips_per_video):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.num_clips_per_video = num_clips_per_video

	def __iter__(self):
		idxs = []
		s = 0
		# select num_clips_per_video for each video, uniformly spaced
		for c in self.video_clips.clips:
			length = len(c)
			if length == 0:
				# corner case where video decoding fails
				continue

			sampled = (
				torch.linspace(s, s + length - 1, steps=self.num_clips_per_video)
				.floor()
				.to(torch.int64)
			)
			s += length
			idxs.append(sampled)
		idxs_ = torch.cat(idxs)
		# shuffle all clips randomly
		perm = torch.randperm(len(idxs_))
		idxs = idxs_[perm]
		return iter(idxs.tolist())

	def __len__(self) -> int:
		return sum(
			self.num_clips_per_video for c in self.video_clips.clips if len(c) > 0
		)

class RandomClipSampler(Sampler):
	def __init__(self, video_clips, max_clips_per_video):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.max_clips_per_video = max_clips_per_video

	def __iter__(self):
		idxs = []
		s = 0
		# select at most max_clips_per_video for each video, randomly
		for c in self.video_clips.clips:
			length = len(c)
			size = min(length, self.max_clips_per_video)
			sampled = torch.randperm(length)[:size] + s
			s += length
			idxs.append(sampled)
		idxs_ = torch.cat(idxs)
		# shuffle all clips randomly
		perm = torch.randperm(len(idxs_))
		return iter(idxs_[perm].tolist())

	def __len__(self) -> int:
		return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)

class RandomFrameSampler(Sampler):
	def __init__(self, video_clips, clips_per_video):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.clips_per_video = clips_per_video

	def __iter__(self):
		idxs = []
		s = 0
		# select at most max_clips_per_video for each video, randomly
		for c in self.video_clips.nframe_idx_list:
			length = len(c)
			sampled = torch.randperm(length)[:self.clips_per_video] + s
			sampled, _ = sampled.sort()
			s += length
			print(sampled)
			idxs.append(sampled)
		idxs_ = torch.stack(idxs)
		perm = torch.randperm(len(idxs_))
		idxs = idxs_[perm]
		idxs = idxs.flatten()
		return iter(idxs.tolist())

	def __len__(self) -> int:
		return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.nframe_idx_list)

class TSNSampler(Sampler):
	def __init__(self, video_clips, n_segment, clips_per_segment):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.n_segment = n_segment
		self.clips_per_segment = clips_per_segment
		self.is_sort = True

	def __iter__(self):
		idxs = []
		s = 0
		# select at most max_clips_per_video for each video, randomly
		for c in self.video_clips.nframe_idx_list:
			length = len(c)
			seg_size = float(length/self.n_segment)
			sampled = []
			for n in range(self.n_segment):
				tmp = torch.randperm(int(length/self.n_segment))[:self.clips_per_segment] + s + int((n)*seg_size)
				sampled.append(tmp)
			sampled = torch.stack(sampled).T
			s += length
			for tmp in sampled: idxs.append(tmp)
		idxs_ = torch.stack(idxs)
		perm = torch.randperm(len(idxs_))
		idxs = idxs_[perm]
		idxs = idxs.flatten()
		return iter(idxs.tolist())

	def __len__(self) -> int:
		return sum(self.clips_per_segment*self.n_segment for _ in self.video_clips.nframe_idx_list)


class BatchSampler(Sampler):
	def __init__(self, sampler, batch_size, drop_last):
		self.sampler = sampler
		self.batch_size = batch_size
		if isinstance(sampler, TSNSampler):
			self.batch_size = batch_size * sampler.n_segment
		elif isinstance(sampler, RandomFrameSampler):
			self.batch_size = batch_size * sampler.clips_per_video
		self.drop_last = drop_last

	def __iter__(self):
		batch = []
		for idx in self.sampler:
			batch.append(idx)
			if len(batch) == self.batch_size:
				yield batch
				batch = []
		if len(batch) > 0 and not self.drop_last:
			yield batch

	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size  # type: ignore
		else:
			return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore


class RandomClipSampler_mix(Sampler):
	def __init__(self, dhf1k_clips, kinetics_clips, max_clips_per_video, n_times_kinetics):
		#video_clips.sampler = self
		self.dhf1k_clips = dhf1k_clips
		self.kinetics_clips = kinetics_clips
		self.max_clips_per_video = max_clips_per_video
		self.n_times_kinetics = n_times_kinetics
		self.len_kinetic = len(self.kinetics_clips)
		self.len_dhf1k = len(self.dhf1k_clips)

	def __iter__(self):
		idxs = []
		s = 0
		# select at most max_clips_per_video for each video, randomly
		for c in self.dhf1k_clips.clips:
			length = len(c)
			size = min(length, self.max_clips_per_video)
			sampled = torch.randperm(length)[:size] + s
			s += length
			idxs.append(sampled)
		
		#print(len(self.dhf1k_clips), len(self.dhf1k_clips.clips)); exit()
		if self.max_clips_per_video == 0:
			sampled_kinetic = torch.randperm(self.len_kinetic)[:self.n_times_kinetics*len(self.dhf1k_clips.clips)]
		else:
			sampled_kinetic = torch.randperm(self.len_kinetic)[:self.n_times_kinetics*len(idxs)]
		sampled_kinetic = self.len_dhf1k + sampled_kinetic
		#print(sampled_kinetic)
		idxs_ = torch.cat(idxs)
		idxs_ = torch.cat([idxs_, sampled_kinetic])
		#print(idxs_.shape)
		#print(torch.sum(idxs_<len(self.dhf1k_clips)))
		#exit()
		# shuffle all clips randomly
		perm = torch.randperm(len(idxs_))
		return iter(idxs_[perm].tolist())
	
	def __len__(self):
		if self.max_clips_per_video == 0:
			return self.n_times_kinetics * len(self.dhf1k_clips.clips)
		return sum(min(len(c), self.max_clips_per_video) for c in self.dhf1k_clips.clips) * (self.n_times_kinetics + 1)


class EvalClipSampler(Sampler):
	def __init__(self, video_clips, window, step):
		video_clips.sampler = self
		self.video_clips = video_clips
		self.window = window
		self.step = step

	def __iter__(self):
		idxs = []
		s = 0
		# select num_clips_per_video for each video, uniformly spaced
		for c in self.video_clips.clips:
			length = len(c)
			if length == 0:
				# corner case where video decoding fails
				continue
			sampled = list(range(s, s + length, self.step))
			if sampled[-1] < s + length - 1: sampled.append(s + length - 1)
			s += length
			idxs.extend(sampled)
		return iter(idxs)

	def __len__(self) -> int:
		return sum(
			len(c)+1 for c in self.video_clips.clips if len(c) > 0
		)