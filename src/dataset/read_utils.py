import cv2
import numpy as np

def read_cv_img(path, target_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

    original_size = img.shape[0:2]
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[1] or target_size[1] != original_size[0]):
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        elif isinstance(target_size, int):
            img = cv2.resize(img, (int(original_size[0]/target_size), int(original_size[2]/target_size))
                                     , interpolation=cv2.INTER_AREA)
    #vgg_img = np.asarray(vgg_img, dtype=np.float32)
    #vgg_img = pytorch_normalze(t.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return img, np.asarray([original_size[1], original_size[0]])


def read_saliency(path, target_size):
    saliency = cv2.imread(path, 0)
    original_size = saliency.shape[0:2]
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[0] or target_size[1] != original_size[1]):
            saliency = cv2.resize(saliency, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    saliency = saliency.astype(np.float32)
    #saliency = t.FloatTensor(saliency)
    return saliency, original_size

def turbo_read_bgr(path, target_size):
    in_file = open(path, 'rb')
    bgr_array = jpeg.decode(in_file.read())
    in_file.close()
    original_size = bgr_array.shape[0:2]
    rgb_array = cv2.cvtColor(bgr_array,cv2.COLOR_BGR2RGB)
    rgb_array = cv2.resize(rgb_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return rgb_array, original_size

def turbo_read_sal(path, target_size):
    in_file = open(path, 'rb')
    bgr_array = jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)
    in_file.close()
    rgb_array = bgr_array[: ,: ,0]
    rgb_array = cv2.resize(rgb_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return rgb_array