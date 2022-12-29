import os
import matplotlib.pylab as plt
import numpy as np
from metrics.metrics import (AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC, SIM)
from multiprocessing import Pool
from dataset import generate_shuffleMaps as shuffleMaps
#dhf_shuffledmap = shuffleMaps.read_DHF1K_shuffledMap()
vid_dict = shuffleMaps.get_vid_list('/data/DHF1K/annotation')

def calc_metric(paths):
    #weights = [0.3, 0.4, 0.3]
    weights = [0.5, 0.5]
    pred_sal_list = paths[0]
    gt_sal = paths[1]
    fixation = plt.imread(gt_sal.replace('maps', 'fixation'))
    gt_sal = plt.imread(gt_sal)
    pred_list = [plt.imread(pred_sal) for pred_sal in pred_sal_list]
    if len(pred_list) == 1: pred_sal = pred_list[0]
    else:
        pred_sal = 0
        for x, w in zip(pred_list, weights):
            pred_sal = pred_sal + w * x
    
    #print(fixation.shape, pred_sal.shape)
    #exit()
    auc_judd_score = AUC_Judd(pred_sal, fixation)
    auc_borji = AUC_Borji(pred_sal, fixation)
    #dhf_shuffledmap = shuffleMaps.sample_DHF1K(vid_dict)
    #sauc = AUC_shuffled(pred_sal, fixation, dhf_shuffledmap)
    cc = CC(pred_sal, gt_sal)
    nss = NSS(pred_sal, fixation)
    sim = SIM(pred_sal, gt_sal)
    #return auc_judd_score, sauc, nss, sim
    #return auc_judd_score, sauc, cc, nss, sim
    return auc_judd_score, auc_borji, cc, nss, sim
    #return auc_judd_score, auc_borji, sauc, cc, nss, sim

def main():
    pool = Pool(6)
    val_vid_list = range(600, 700)
    gt_path = '/data/DHF1K/annotation/'

    sub_dir = '/home/feiyan/'
    #sub_path = [sub_dir+'test_generate']
    #sub_path = [sub_dir+'test_generate_rc4']
    sub_path = [sub_dir+'test_generate_rc2']
    #sub_path = [sub_dir+'test_generate_samepad']
    #sub_path = [sub_dir+'test_generate_d123m']
    #sub_path = [sub_dir+'test_generate_d123m_rc2_ta']

    #sub_dir = '/home/'
    #sub_path = [sub_dir+'test_generate_tmp']

    #sub_dir = '/home/feiyan/'
    #sub_path = [sub_dir+'test_generate']

    #sub_path = [sub_dir+'test_generate_d123m']
    #sub_path = [sub_dir+'test_generate_vinet']
    #sub_path = [sub_dir+'test_generate_tased']
    #sub_path = [sub_dir+'test_generate_s3ddla']
    #sub_path = [sub_dir+'test_generate_tmp', sub_dir+'test_generate_d1d2d3s']
    all_metrics = []
    for vid in val_vid_list:
        vid = vid + 1
        vid_path = '{}/{:04d}/maps/'.format(gt_path, vid)
        frame_list = [n.split('.')[0] for n in os.listdir(vid_path) if '.png' in n]
        frame_list.sort()
        
        #frame_list = frame_list[:int(int(len(frame_list)/16)*16)]
        #gt_list = [os.path.join(vid_path, frame_id) for frame_id in frame_list]
        
        frame_list = [([os.path.join(sub_pred_dir, '{:04d}'.format(vid), frame_id+'.png') for sub_pred_dir in sub_path], os.path.join(vid_path, frame_id+'.png')) \
                   for frame_id in frame_list]
        result_matrix = pool.map(calc_metric, frame_list)
        result_matrix = np.asarray(result_matrix)
        #np.save('../eval_results/d123s_lonly/{}.npy'.format(vid), result_matrix)
        #np.save('../eval_results/d1s/{}.npy'.format(vid), result_matrix)
        #np.save('../eval_results/d123s_tasedteacher/{}.npy'.format(vid), result_matrix)
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
    #shuffleMaps.sample_DHF1K('/home/feiyan/data/DHF1K/annotation')
    #shuffleMaps.sample_UCF('/home/feiyan/data/ucf_sport/training/')
    main()

    #0601 [0.94260393 0.90405098 0.52923381 3.62900686] mean so far [0.94260393 0.90405098 0.52923381 3.62900686]
    #0602 [0.91216144 0.87534007 0.50829668 2.66743018] mean so far [0.92738268 0.88969553 0.51876525 3.14821852]
