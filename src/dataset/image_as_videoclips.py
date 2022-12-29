import numpy as np
import bisect
import torch
#from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from PIL import Image
import cv2


def read_image_stack_idx(path, idx, size=None):
    img_stack = []
    tmp_name = path.split('/')[-1]
    tmp_name = tmp_name[:-4] + '_' + tmp_name[-3:]
    for i in idx:
        img = Image.open(path+'/images/{}_{:03d}.png'.format(tmp_name, i+1))
        if size is not None:
            img = img.resize((size[1], size[0]))
        img = np.array(img)
        img_stack.append(img)
    img_stack = np.asarray(img_stack)
    #print(img_stack.shape)
    return img_stack

def read_image_stack_idx_unisal(path, idx, size=None):
    img_stack = []
    tmp_name = path.split('/')[-1]
    tmp_name = tmp_name[:-4] + '_' + tmp_name[-3:]
    for i in idx:
        #print(path+'/images/{}_{:03d}.png'.format(tmp_name, i+1))
        img = cv2.imread(path+'/images/{}_{:03d}.png'.format(tmp_name, i+1))
        #if size is not None:
        #    img = img.resize((size[1], size[0]))
        img = np.ascontiguousarray(img[:, :, ::-1])
        img_stack.append(img)
    img_stack = np.asarray(img_stack)
    #print(img_stack.shape)
    #exit()
    return img_stack

class VideoClips(object):
    def __init__(self, vid_list, window, step, metadata, size=None, every_n_skip=None):
        self.video_paths = vid_list

        if every_n_skip is None:
            every_n_skip = 1
        
        self.clips = self.get_resampled_frames(vid_list, metadata, window, step, every_n_skip)
        self.cumulative_sizes = np.asarray([len(n) for n in self.clips]).cumsum(0).tolist()
        self.every_n_skip = every_n_skip
        self.window = window
        self.size = size

    def get_clip(self, item):
        video_idx, clip_idx = self.get_clip_indx(item)
        v_id = self.video_paths[video_idx]
        f_id = self.clips[video_idx][clip_idx]

        img_stack = read_image_stack_idx(v_id, f_id, self.size)
        is_last_clip = item + 1 in self.cumulative_sizes
        #print(img_stack.shape)
        img_stack = torch.from_numpy(img_stack)

        return img_stack, None, None, video_idx, np.asarray(f_id)
    
    def get_clip_info(self, item):
        video_idx, clip_idx = self.get_clip_indx(item)
        v_id = self.video_paths[video_idx]
        f_id = self.clips[video_idx][clip_idx]
        last_clip_pts = self.clips[video_idx][-1]
        is_last_clip = last_clip_pts == f_id
        return v_id, f_id, is_last_clip

    def __len__(self):
        return sum([len(n) for n in self.clips])
    
    def get_clip_indx(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def get_resampled_frames(vid_list, metadata, window, step, every_n_skip):
        sampled_frames_vid = []
        assert len(metadata["video_paths"]) == len(metadata["video_pts"])
        #for v_id, v_pts in zip(metadata["video_paths"], metadata["video_pts"]):
        #    v_dict[v_id] = v_pts
        #print(v_dict)

        for v_id, frame_list in zip(metadata["video_paths"], metadata["video_pts"]):
            #print(v_id, frame_list, every_n_skip)
            frame_start_idx = range(0, len(frame_list)-window+1, step)
            sampled_frame = [range(frame_list[idx], frame_list[idx]+window, every_n_skip) for idx in frame_start_idx]
            if len(sampled_frame) == 0:
                sampled_frame = [[0 for _ in range(window-len(frame_list))] + list(range(0, len(frame_list)))]
                #sampled_frame = [np.around(np.linspace(0, len(frame_list)-1, int(window/every_n_skip) )).astype(np.int32)]
            sampled_frames_vid.append(sampled_frame)
        return sampled_frames_vid
    
    def reduce_with_indx(self, indx):
        def select(d, j):
            return [x for i, x in enumerate(d) if i in j]
        self.clips = [select(x, y) for x,y in zip(self.clips, indx)]
        self.cumulative_sizes = np.asarray([len(n) for n in self.clips]).cumsum(0).tolist()
