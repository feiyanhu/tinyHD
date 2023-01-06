import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset.DHF1K_kinetics400 as dhf1k_kinetic
import dataset.DHF1K_vid as dhf1k
import dataset.ucf_kinetics400 as ucf_kinetic
import dataset.UCFsport_vid as ucf
import dataset.kinetics400 as kinetic

from my_sampler import BatchSampler, UniformClipSampler, RandomClipSampler, RandomClipSampler_mix
from models.fastsal3D.model import FastSalA
from utils import AverageMeter, my_scheduler, KLDLoss1vs1
from teacher_wrapper import S3D_wrapper

def compute_teacher_label_loss_v4_kinetic(model, vgg_in, label, has_sal, decoder_config, inter_config, teacher_supervision):
    kl_loss = KLDLoss1vs1()
    loss_gt = t.zeros(1).cuda()
    inter_loss = []
    y, y_inters = model(vgg_in)
    
    y_t, y_inters_t = teacher_supervision
    func_dict = {'I': lambda x: x, 'M': lambda x:t.mean(x, dim=2, keepdim=True)}
    
    for y_inter, d, ic in zip(y_inters, decoder_config, inter_config):
        if d == 'd1' and ic is not None:
            y_inter_s3dlike = y_inter.permute(1, 0, 2, 3, 4).contiguous()
            y_inters_t = func_dict[ic](y_inters_t)
            #print(y_inter_s3dlike.shape, y_inters_t.shape)
            #print(h1.shape, h2.shape, h3.shape, h4.shape, y_t.shape)
            #print(y_inter.shape, d, ic)
            inter_loss = [kl_loss(tmp_x.view(-1, 192, 256), tmp_h.cuda(0).view(-1, 192, 256)) for tmp_x, tmp_h in zip(y_inter_s3dlike, y_inters_t)]
        elif (d == 'd2' or d == 'd3') and ic == 'GT':
            if has_sal.sum(0) > 0:
                inter_loss_tmp = kl_loss(y_inter[has_sal].view(-1, 192, 256), label.view(-1, 192, 256))
                inter_loss.append(0.5*inter_loss_tmp)


    if len(inter_loss) == 0: inter_loss = [t.zeros(1).cuda()]
    inter_loss = sum(inter_loss)
    loss_s3d = kl_loss(y.view(-1, 192, 256), y_t.cuda(0).view(-1, 192, 256)) #- 0.5*cc_loss(y, y_t.cuda(1))
    
    if has_sal.sum(0) > 0:
        loss_gt = kl_loss(y[has_sal].view(-1, 192, 256), label.view(-1, 192, 256)) #- 0.5*cc_loss(y[has_sal], label)
    return loss_gt, loss_s3d, inter_loss

def train_one(model, dataloader, optimizer, scaler, decoder_config, inter_config, x_indx, teacher_indx):
    [model, s3d_model] = model
    all_loss_gt, all_loss_s3d, all_loss_inter= AverageMeter(), AverageMeter(), AverageMeter()
    

    for i, (X, cls, has_sal, has_cls) in enumerate(dataloader):
        optimizer.zero_grad()
        vgg_in = X[0][:,:,x_indx,:,:].float().cuda(0)
        s3d_in = X[1].float().cuda(0)
        label = X[2][has_sal].float().cuda(0)
        #print(vgg_in.shape, s3d_in.shape, label.shape)
        #exit()
        t_s = s3d_model.get_supervision(s3d_in, teacher_indx)
        if scaler is None:
            loss_gt, loss_s3d, inter_loss = compute_teacher_label_loss_v4_kinetic(model, vgg_in, label, has_sal, decoder_config, inter_config, t_s)
        else:
            with t.cuda.amp.autocast():
                #t_s = s3d_model.get_supervision(s3d_in, teacher_indx)
                loss_gt, loss_s3d, inter_loss = compute_teacher_label_loss_v4_kinetic(model, vgg_in, label, has_sal, decoder_config, inter_config, t_s)
        loss = loss_gt + loss_s3d + inter_loss

        all_loss_gt.update(loss_gt.item(), t.sum(has_sal))
        all_loss_s3d.update(loss_s3d.item(), len(vgg_in))
        all_loss_inter.update(inter_loss.item(), len(vgg_in))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if i%15 == 0:
            print('{}/{} current accumulated gt loss {}, s3d loss {}, inter loss {}'.format(i, len(dataloader), 
            all_loss_gt.avg, all_loss_s3d.avg, all_loss_inter.avg))
    
    print('{} current accumulated gt loss {}, s3d {}'.format(i, all_loss_gt.avg, all_loss_s3d.avg))
    print('-------------------')
    return all_loss_gt.avg, all_loss_s3d.avg, model

def test_one(model, dataloader, decoder_config, inter_config, x_indx, teacher_indx):
    [model, s3d_model] = model
    all_loss_gt, all_loss_s3d, all_loss_inter = AverageMeter(), AverageMeter(), AverageMeter()
    
    for i, (X, _, video_idx, _, has_sal, rand_sig) in enumerate(dataloader):
        vgg_in = X[0][:,:,x_indx,:,:].float().cuda(0)
        s3d_in = X[1].float().cuda(0)
        label = X[2][has_sal].float().cuda(0)

        t_s = s3d_model.get_supervision(s3d_in, teacher_indx)
        with t.cuda.amp.autocast():
            loss_gt, loss_s3d, inter_loss = compute_teacher_label_loss_v4_kinetic(model, vgg_in, label, has_sal, decoder_config, inter_config, t_s)
        all_loss_gt.update(loss_gt.item(), len(vgg_in))
        all_loss_s3d.update(loss_s3d.item(), len(vgg_in))
        all_loss_inter.update(inter_loss.item(), len(vgg_in))
        
        if i%15 == 0:
            print('{}/{} current accumulated gt loss {}, s3d loss {}, inter loss {}'.format(i, len(dataloader), 
            all_loss_gt.avg, all_loss_s3d.avg, all_loss_inter.avg))
        
    print('{} current accumulated gt loss {}, s3d {}'.format(i, all_loss_gt.avg, all_loss_s3d.avg))
    print('-------------------')
    return all_loss_gt.avg, all_loss_s3d.avg, model

def DHF1K_kinetic_data(aux_data, batch_size, frame_num, sal_indx, data_dir):
    if aux_data is not None:
        print('with aux')
        train_snpit = dhf1k_kinetic.DHF1K_kinetics400('train', frame_num, 1, ['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                                  sal_indx=sal_indx, frame_rate=None,
                                                  data_dir=data_dir)
        train_batch_sampler = BatchSampler(RandomClipSampler_mix(train_snpit.dhf1k_dataset.clips, train_snpit.kinetics400_dataset.clips, 1, 1), batch_size, False)
    else:
        print('without aux')
        train_snpit = dhf1k.DHF1K('train', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                  sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])
        train_batch_sampler = BatchSampler(RandomClipSampler(train_snpit.clips, 2), batch_size, False)
    val_snpit = dhf1k.DHF1K('val', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(200, 355), (192, 256)],
                            sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])

    
    val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 5), batch_size*2, False)
    dataloader = {
        'train': DataLoader(train_snpit, num_workers=6, batch_sampler=train_batch_sampler),
        'val': DataLoader(val_snpit, num_workers=6, batch_sampler=val_batch_sampler)
    }
    return dataloader, train_snpit

def UCF_kinetic_data(aux_data, batch_size, frame_num, sal_indx, data_dir):
    if aux_data is not None:
        print('with aux')
        train_snpit = ucf_kinetic.ucf_kinetics400('train_', frame_num, 1, ['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                                  sal_indx=sal_indx, frame_rate=None,
                                                  data_dir=data_dir)
        train_batch_sampler = BatchSampler(RandomClipSampler_mix(train_snpit.ucf_dataset.clips, train_snpit.kinetics400_dataset.clips, 1, 10), batch_size, False)
    else:
        print('without aux')
        train_snpit = ucf.UCF('train_', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                                  sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])
        train_batch_sampler = BatchSampler(RandomClipSampler(train_snpit.clips, 11), batch_size, False)
    val_snpit = ucf.UCF('val_', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(200, 355), (192, 256)],
                            sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])

    #train_batch_sampler = BatchSampler(RandomClipSampler_mix(train_snpit.ucf_dataset.clips, train_snpit.kinetics400_dataset.clips, 1, 10), batch_size, False)
    #val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 5), batch_size*2, False)
    val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 20), batch_size*2, False)
    dataloader = {
        'train': DataLoader(train_snpit, num_workers=6, batch_sampler=train_batch_sampler),
        'val': DataLoader(val_snpit, num_workers=6, batch_sampler=val_batch_sampler)
    }
    return dataloader, train_snpit

def kinetic_data(batch_size, frame_num, sal_indx, data_dir):
    train_snpit = dhf1k_kinetic.DHF1K_kinetics400('train', frame_num, 1, ['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                                  sal_indx=sal_indx, frame_rate=None, data_dir=data_dir)
    train_batch_sampler = BatchSampler(RandomClipSampler_mix(train_snpit.dhf1k_dataset.clips, train_snpit.kinetics400_dataset.clips, 0, 2), batch_size, False)
    val_snpit = dhf1k.DHF1K('val', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(200, 355), (192, 256)],
                            sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])
    val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 5), batch_size*2, False)
    dataloader = {
        'train': DataLoader(train_snpit, num_workers=6, batch_sampler=train_batch_sampler),
        'val': DataLoader(val_snpit, num_workers=6, batch_sampler=val_batch_sampler)
    }
    return dataloader, train_snpit

def main(model, teacher_model, optimizer, dataloader, inter_config, train_config, save_path):
    [x_indx, teacher_indx] = train_config
    model.cuda(0)
    teacher_model.model.cuda(0)
    scaler = t.cuda.amp.GradScaler()
    #scaler = None

    smallest_loss_gt, smallest_loss_s3d = 100, 100
    scheduler3 = my_scheduler(optimizer, [100, 150, 180], 0.1)

    #for epoch in range(0,100, 1):
    for epoch in range(0,200, 1):
        model.train()
        loss_train_gt,loss_train_s3d, model = train_one([model, teacher_model], dataloader['train'], optimizer, scaler, decoder_config, inter_config, x_indx, teacher_indx)
        print('{} loss train gt {}, s3d {}'.format(epoch, loss_train_gt, loss_train_s3d))
        print('--------------------------------------------->>>>>>')
        model.eval()
        with t.no_grad():
            loss_val_gt,loss_val_s3d, model = test_one([model, teacher_model], dataloader['val'], decoder_config, inter_config, x_indx, teacher_indx)
        print('--------------------------------------------->>>>>>')
        print('{} loss gt val {}, s3d {}'.format(epoch, loss_val_gt, loss_val_s3d))
        if loss_val_gt < smallest_loss_gt or loss_val_s3d < smallest_loss_s3d:
            path = '{}{}/ft_{}_{:.5f}_{:.5f}.pth'.format(save_path, model_name, epoch,loss_val_s3d, loss_val_gt)
            d = {}
            d['student_model'] = model.state_dict()
            t.save(d,path)
            best_epoch = epoch
            smallest_loss_gt = loss_val_gt
            smallest_loss_s3d = loss_val_s3d

        best_weight_path = '{}{}/ft_{}_{:.5f}_{:.5f}.pth'.format(save_path, model_name, best_epoch, smallest_loss_s3d, smallest_loss_gt)
        scheduler3.step(model, best_weight_path)

def config_dataset(data_dir, aux_only, aux_data, model_input_size, model_output_size, teacher_type):
    data_select_idx = [dt is not None for dt in data_dir]
    assert sum(data_select_idx) <= 1, print('Please choose only 1 target dataset!! Mark unwanted dataset path to NULL in config file')
    if sum(data_select_idx) == 0 and aux_data is None:
        print('if no target dataset is choosen, kinetic 400 data needs to be specified!')
        exit()
    #print('model input size ', model_input_size, 'model ouput index ', model_output_index, 'teacher input size ', teacher_input_size, 'teacher output index ', teacher_output_index)
    print('model input size ', model_input_size, 'teacher_type ', teacher_type)
    if teacher_type == 'HD2S':
        if model_output_size == 1:
            print('single output')
            frame_num, sal_indx = 16, list(range(15, 16))
            x_indx, teacher_indx = range(0, 16), [range(0, 16)]
        elif model_output_size == 16:
            print('16 outputs')
            frame_num, sal_indx = 32, list(range(16, 32))
            x_indx, teacher_indx = range(16, 32), [range(i, i+16) for i in range(16)]
        elif model_output_size == 8:
            print('8 output not yet implemented'); exit()
            frame_num, sal_indx = 24, list(range(16, 24))
            x_indx, teacher_indx = range(8, 24), [range(i, i+16) for i in range(8)]
        else:
            print('{} number of output is not supported yet!'.format(model_output_size))
            exit()
    
    #if sum(data_select_idx) == 0 and aux_data is not None:
    #    print('only kinectic 400 is used! only teacher supervision is used!')
    #    dataloader, _ = kinetic_data(batch_size, frame_num, sal_indx, [data_dir[0], aux_data])
    #print(data_dir, aux_only, aux_data)
    if data_select_idx.index(True) == 0:
        if aux_only:
            print('only kinectic 400 is used! only teacher supervision is used for training! Only DHF1K validation set used!')
            dataloader, _ = kinetic_data(batch_size, frame_num, sal_indx, [data_dir[0], aux_data])
        else:
            print('DHF1K selected! Aux data selected! Kinetic400 is used! Create dataset now', data_dir[0], aux_data)
            dataloader, _ = DHF1K_kinetic_data(aux_data, batch_size, frame_num, sal_indx, [data_dir[0], aux_data])
    elif data_select_idx.index(True) == 1:
        print('UCF selected! Aux data selected! Kinetic400 is used! Create dataset now')
        dataloader, _ = UCF_kinetic_data(aux_data, batch_size, frame_num, sal_indx, [data_dir[1], aux_data])
    elif data_select_idx.index(True) == 2:
        print('Hollywood selected! Aux data selected! Kinetic400 is used! Create dataset now')
    
    return dataloader, [x_indx, teacher_indx]

if __name__ == '__main__':
    import argparse
    import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to config file")
    args = parser.parse_args()
    config = cfg.get_config(args.config_path)
    #config = cfg.get_config('config/train_config_single.yaml')
    #config = cfg.get_config('config/train_config_multi.yaml')
    #config = cfg.get_config('config/train_config_multi_slow.yaml')
    #print(config)
    #print(, config.MODEL_SETUP.DECODER)
    #exit()
    lr = config.LEARNING_SETUP.LR
    batch_size = config.LEARNING_SETUP.BATCH_SIZE#15#9#15#18 #11
    save_path = config.LEARNING_SETUP.OUTPUT_PATH
    data_dir=[config.DATASET_SETUP.DHF1K_PATH, config.DATASET_SETUP.UCF_PATH, config.DATASET_SETUP.HOLLYWOOD_PATH]
    aux_data = config.DATASET_SETUP.KINETIC400_PATH
    aux_only = config.DATASET_SETUP.AUX_ONLY
    teacher_path = config.TEACHER_SETUP.PATH

    model_input_size = config.MODEL_SETUP.INPUT_SIZE
    model_output_size = config.MODEL_SETUP.OUTPUT_SIZE
    teacher_type = config.TEACHER_SETUP.TYPE

    reduced_channel = config.MODEL_SETUP.CHANNEL_REDUCTION
    decoder_config = config.MODEL_SETUP.DECODER
    single_mode = config.MODEL_SETUP.SINGLE
    force_multi = config.MODEL_SETUP.FORCE_MULTI
    inter_config = config.MODEL_SETUP.ITERMEDIATE_TARGET
    #Single
    #single_mode = [True, True, True]
    #force_multi = False
    #inter_config = ['I', None, None]
    #multi16
    #single_mode = [True, False, False]
    #force_multi = False
    #inter_config = ['M', None, None]
    #multi16
    #single_mode = [False, False, False]
    #force_multi = False
    #inter_config = ['I', None, None]
    #multi16
    #single_mode = [True, True, True]
    #force_multi = True
    #inter_config = ['M', None, None]
    #d1 d2
    #single_mode = [True, True]
    #force_multi = False
    #inter_config = ['I', None]

    
    data_loader, train_config = config_dataset(data_dir, aux_only, aux_data, model_input_size, model_output_size, teacher_type)
    model = FastSalA(reduced_channel, decoder_config, single_mode, force_multi=force_multi, n_output=model_output_size)
    teacher_model = S3D_wrapper(teacher_path)
    optimizer = model.get_optimizer(lr)
    main(model, teacher_model, optimizer, data_loader, inter_config, train_config, save_path)
