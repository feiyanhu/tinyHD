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
    else:exit('length of prediction list can only be 1')
    
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
        
        #frame_list = frame_list[:int(int(len(frame_list)/16)*16)]
        #gt_list = [os.path.join(vid_path, frame_id) for frame_id in frame_list]
        if data_type == 'dhf1k':
            frame_list = [(os.path.join(pred_path, '{:04d}'.format(vid), frame_id+'.png'), os.path.join(vid_path, frame_id+'.png')) for frame_id in frame_list]
        elif data_type == 'ucf':
            frame_list = [(os.path.join(pred_path, vid, frame_id+'.png'), os.path.join(vid_path, frame_id+'.png')) for frame_id in frame_list]

        result_matrix = pool.map(calc_metric, frame_list)
        result_matrix = np.asarray(result_matrix)
        #np.save('../eval_results/d123s_lonly/{}.npy'.format(vid), result_matrix)
        all_metrics.append(np.mean(result_matrix, axis=0))
        print(vid, np.mean(result_matrix, axis=0), 'accumulated mean so far', np.mean(all_metrics, axis=0))
    print('----------------------------------->123s 16 frame*')
    print(np.mean(all_metrics, axis=0))
    pool.close()
    pool.join()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_path", help="path to prediction")
    parser.add_argument("GT_path", help="path to ground truth")
    parser.add_argument("data_type", help="which dataset")
    args = parser.parse_args()

    pred_path = args.prediction_path
    gt_path = args.GT_path
    data_type = args.data_type

    data_type = 'dhf1k' #dhf1k or ucf
    if data_type == 'dhf1k':
        data_path = '/data/DHF1K/' #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = range(601, 701)
    elif data_type == 'ucf':
        data_path = '/home/feiyan/data/ucf_sport/testing/' #'/data/DHF1K/' or '/home/feiyan/data/ucf_sport/testing/'
        vid_list = os.listdir(data_path)

    main(gt_path, vid_list, pred_path, data_type)