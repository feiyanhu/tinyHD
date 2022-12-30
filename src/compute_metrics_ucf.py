import os
from metrics.utils import normalize_map
import matplotlib.pylab as plt
import numpy as np
from metrics.metrics import (AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC, SIM)
from multiprocessing import Pool
import cv2

def calc_metric(paths):
    weights = [0.6, 0.4]
    pred_sal_list = paths[0]
    gt_sal = paths[1]
    fixation = plt.imread(gt_sal.replace('maps', 'fixation'))
    gt_sal = plt.imread(gt_sal)
    pred_list = [plt.imread(pred_sal) for pred_sal in pred_sal_list]
    pred_list = [cv2.resize(x, dsize=(fixation.shape[1], fixation.shape[0]), interpolation=cv2.INTER_CUBIC) for x in pred_list]
    if len(pred_list) == 1: pred_sal = pred_list[0]
    else:
        pred_sal = 0
        for x, w in zip(pred_list, weights):
            pred_sal = pred_sal + w * x
    
    #print(fixation.shape, pred_sal.shape, gt_sal.shape)
    #exit()
    auc_judd_score = AUC_Judd(pred_sal, fixation)
    #auc_borji = AUC_Borji(pred_sal, fixation)
    sim = SIM(pred_sal, gt_sal)
    cc = CC(pred_sal, gt_sal)
    nss = NSS(pred_sal, fixation)
    return auc_judd_score, sim, cc, nss

def main():
    pool = Pool(6)
    gt_path = '/home/feiyan/data/ucf_sport/testing/'
    val_vid_list = os.listdir(gt_path)
    sub_dir = '/home/feiyan/runs/'
    #sub_path = [sub_dir+'generated_dhf1k_model10_last1', sub_dir+'generated_dhf1k_model10_last3']
    #sub_path = [sub_dir+'generated_dhf1k_model10_last3', sub_dir+'generated_dhf1k_model10_last4']
    #sub_path = [sub_dir+'test_generate_ucf']
    sub_path = [sub_dir+'test_generate_ucf_tmp']
    all_metrics = []
    for vid in val_vid_list:
        vid_path = '{}/{}/maps/'.format(gt_path, vid)
        frame_list = [n.split('.')[0] for n in os.listdir(vid_path) if '.png' in n]
        frame_list.sort()
        #print(vid_path, frame_list)
        #exit()
        #gt_list = [os.path.join(vid_path, frame_id) for frame_id in frame_list]
        frame_list = [([os.path.join(sub_pred_dir, vid, frame_id+'.png') for sub_pred_dir in sub_path], os.path.join(vid_path, frame_id+'.png')) \
                   for frame_id in frame_list]
        result_matrix = pool.map(calc_metric, frame_list)
        result_matrix = np.asarray(result_matrix)
        all_metrics.append(np.mean(result_matrix, axis=0))
        print(vid, np.mean(result_matrix, axis=0), 'accumulated mean so far', np.mean(all_metrics, axis=0))
    print('----------------------------------->')
    print(np.mean(all_metrics, axis=0))
    pool.close()
    pool.join()

if __name__ == '__main__':
    #dhf1k_path = '/data/DHF1K/'

    #save_path = 'generated/'
    #save_path = '/home/feiyan/data/generated_dhf1k_model10_last3/'

    main()

    #0601 [0.94260393 0.90405098 0.52923381 3.62900686] mean so far [0.94260393 0.90405098 0.52923381 3.62900686]
    #0602 [0.91216144 0.87534007 0.50829668 2.66743018] mean so far [0.92738268 0.88969553 0.51876525 3.14821852]
