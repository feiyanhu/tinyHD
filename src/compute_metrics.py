import os
import matplotlib.pylab as plt
import numpy as np
from metrics.metrics import (AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC, SIM)
from multiprocessing import Pool
from dataset import generate_shuffleMaps as shuffleMaps
#dhf_shuffledmap = shuffleMaps.read_DHF1K_shuffledMap()
#vid_dict = shuffleMaps.get_vid_list('/data/DHF1K/annotation')

def calc_metric(paths):
    #weights = [0.3, 0.4, 0.3]
    #weights = [0.5, 0.5]
    pred_sal = paths[0]
    gt_sal = paths[1]
    fixation = plt.imread(gt_sal.replace('maps', 'fixation'))
    gt_sal = plt.imread(gt_sal)
    pred_list = [plt.imread(pred_sal)]
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

def main(data_dir, vid_list, pred_path, data_type):
    pool = Pool(6)
    if data_type == 'dhf1k':
        gt_path = '{}annotation'.format(data_dir)
    elif data_type == 'ucf':
        gt_path = data_dir

    #sub_dir = '/home/feiyan/'
    #sub_path = [sub_dir+'test_generate']

    all_metrics = []
    for vid in vid_list:
        #vid = vid + 1
        if data_type == 'dhf1k':
            vid_path = '{}/{:04d}/maps/'.format(gt_path, vid)
            #frame_list = [n.split('.')[0] for n in os.listdir(vid_path) if '.png' in n]
        elif data_type == 'ucf':
            vid_path = '{}/{}/maps/'.format(gt_path, vid)
        frame_list = [n.split('.')[0] for n in os.listdir(vid_path) if '.png' in n]
        frame_list.sort()
        
        frame_list = frame_list[:int(int(len(frame_list)/16)*16)]
        #gt_list = [os.path.join(vid_path, frame_id) for frame_id in frame_list]
        if data_type == 'dhf1k':
            frame_list = [(os.path.join(pred_path, '{:04d}'.format(vid), frame_id+'.png'), os.path.join(vid_path, frame_id+'.png')) for frame_id in frame_list]
        elif data_type == 'ucf':
            frame_list = [(os.path.join(pred_path, vid, frame_id+'.png'), os.path.join(vid_path, frame_id+'.png')) for frame_id in frame_list]
        #print(frame_list)
        #exit()
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
    #shuffleMaps.sample_DHF1K('/home/feiyan/data/DHF1K/annotation')
    #shuffleMaps.sample_UCF('/home/feiyan/data/ucf_sport/training/')
    data_type = 'dhf1k' #dhf1k or ucf
    if data_type == 'dhf1k':
        data_path = '/home/feiyan/data/DHF1K/' #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = range(601, 701)
    elif data_type == 'ucf':
        data_path = '/home/feiyan/data/ucf_sport/testing/' #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = os.listdir(data_path)

    pred_path = '/home/feiyan/runs/test_generate_d123m'
    #pred_path = '/home/feiyan/runs/test_generate_ucf_tmp'
    #pred_path = '/home/feiyan/runs/test_generate_ucf_multi'
    #pred_path = '/home/feiyan/runs/test_generate'
    #sub_path = [sub_dir+'test_generate_rc4']
    #sub_path = [sub_dir+'test_generate_rc2']
    #sub_path = [sub_dir+'test_generate_samepad']
    #sub_path = [sub_dir+'test_generate_d123m']
    #sub_path = [sub_dir+'test_generate_d123m_rc2_ta']
    main(data_path, vid_list, pred_path, data_type)

    #0601 [0.94260393 0.90405098 0.52923381 3.62900686] mean so far [0.94260393 0.90405098 0.52923381 3.62900686]
    #0602 [0.91216144 0.87534007 0.50829668 2.66743018] mean so far [0.92738268 0.88969553 0.51876525 3.14821852]
