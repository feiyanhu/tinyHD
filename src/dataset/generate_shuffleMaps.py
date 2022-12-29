import os
import numpy as np
import cv2
np.random.seed(seed=10)

def read_fixations_dhf1k(vid_dir, ids):
    tmp = np.zeros((360, 640))
    for id in ids:
        im = cv2.imread('{}{:04d}.png'.format(vid_dir, id), 0)
        im = im.astype(np.float32)
        im = im/255.0
        tmp += im
        #print('{}{:04d}.png'.format(vid_dir, id))
    return tmp

def get_vid_list(path):
    vid_dict = {}
    for vid in range(1, 601):
        vid_dir = '{}/{:04d}/fixation/'.format(path, vid)
        frame_list = [x for x in os.listdir(vid_dir) if '.png' in x]
        vid_dict[vid_dir] = len(frame_list)
    return vid_dict



def sample_DHF1K(vid_dict):
    vid_keys = list(vid_dict.keys())
    arr = np.arange(len(vid_keys))
    np.random.shuffle(arr)
    selected_keys = [vid_keys[i] for i in arr[:10]]
    #print(selected_keys)

    shuffledMap = []
    for vid_dir in selected_keys:
        #print(vid_dir, vid_dict[vid_dir])
        arr = np.arange(vid_dict[vid_dir])
        np.random.shuffle(arr)
        arr = arr[0] + 1
        vid_shufflemap = read_fixations_dhf1k(vid_dir, [arr])
        #print(vid_shufflemap.shape)
        shuffledMap.append(vid_shufflemap)
    shuffledMap = np.mean(shuffledMap, axis=0)
    shuffledMap[shuffledMap>0]=1
    return shuffledMap

def read_DHF1K_shuffledMap(path='dataset/metadata/dhf1k_shuffledmap.npy'):
    return np.load(path)

def read_fixations_ucf(vid_dir, ids, frame_prefix):
    tmp = np.zeros((480, 720))
    for id in ids:
        img_path = '{}{}_{:03d}.png'.format(vid_dir, frame_prefix, id)
        im = cv2.imread(img_path, 0)
        im = cv2.resize(im, (720, 480))
        im = im.astype(np.float32)
        im = im/255.0
        tmp += im
        #print(img_path)
    #exit()
    return tmp

def sample_UCF(path):
    vids = os.listdir(path)
    np.random.seed(seed=10)
    shuffledMap = np.zeros((480, 720))
    for vid in vids:
        print(vid)
        frame_prefix = list(vid)
        frame_prefix[-4] = '_'
        frame_prefix = "".join(frame_prefix)
        vid_dir = '{}{}/fixation/'.format(path, vid)
        frame_list = [x for x in os.listdir(vid_dir) if '.png' in x]

        arr = np.arange(len(frame_list))
        np.random.shuffle(arr)
        arr = arr[:10] + 1
        vid_shufflemap = read_fixations_ucf(vid_dir, arr, frame_prefix)
        shuffledMap += vid_shufflemap
        print(vid_shufflemap.shape, vid)
    shuffledMap[shuffledMap>0]=1
    print(shuffledMap.shape)
    np.save('dataset/metadata/ucf_shuffledmap.npy', shuffledMap)
    
def read_UCF_shuffledMap(path='dataset/metadata/ucf_shuffledmap.npy'):
    return np.load(path)