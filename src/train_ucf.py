import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset.ucf_kinetics400 as ucf_kinetic
import dataset.UCFsport_vid as ucf

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
        if d == 'd1':
            y_inter_s3dlike = y_inter.permute(1, 0, 2, 3, 4).contiguous()
            y_inters_t = func_dict[ic](y_inters_t)
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

def DHF1K_data(batch_size, frame_num, sal_indx, data_dir):
    train_snpit = ucf_kinetic.ucf_kinetics400('train_', frame_num, 1, ['vgg_in', 's3d', 'sal'], size=[(192, 256), (192, 256)],
                                                  sal_indx=sal_indx, frame_rate=None,
                                                  data_dir=data_dir)
    size=[(200, 355), (192, 256)]
    #val_snpit = ucf.UCF('testing', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(200, 355), (192, 256)],
    #                        sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])
    val_snpit = ucf.UCF('val_', frame_num, 1, out_type=['vgg_in', 's3d', 'sal'], size=[(200, 355), (192, 256)],
                            sal_indx=sal_indx, frame_rate=None, data_dir=data_dir[0])

    train_batch_sampler = BatchSampler(RandomClipSampler_mix(train_snpit.ucf_dataset.clips, train_snpit.kinetics400_dataset.clips, 1, 10), batch_size, False)
    #val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 5), batch_size*2, False)
    val_batch_sampler = BatchSampler(UniformClipSampler(val_snpit.clips, 20), batch_size*2, False)
    dataloader = {
        'train': DataLoader(train_snpit, num_workers=6, batch_sampler=train_batch_sampler),
        'val': DataLoader(val_snpit, num_workers=6, batch_sampler=val_batch_sampler)
    }
    return dataloader, train_snpit

def create_teacher(model_path):
    from models.S3D.SalGradNet_original import SalGradNet as S3D_SGN
    model = S3D_SGN()
    model.load_state_dict(t.load(model_path, map_location='cuda:0'))
    return model

def main(model_name, reduced_channel, decoder_config, single_mode, inter_config,
         frame_num, sal_indx, x_indx, teacher_indx,
         lr, batch_size, save_path, data_dir, teacher_path, force_multi=False):
    model = FastSalA(reduced_channel, decoder_config, single_mode, force_multi=force_multi)
    teacher_model = S3D_wrapper(teacher_path)

    dataloader, ds_train = DHF1K_data(batch_size, frame_num, sal_indx, data_dir)

    model.cuda(0)
    teacher_model.model.cuda(0)
    optimizer = model.get_optimizer(lr)
    scaler = t.cuda.amp.GradScaler()
    #scaler = None

    smallest_loss_gt, smallest_loss_s3d = 100, 100
    scheduler3 = my_scheduler(optimizer, [100, 150, 180], 0.1)


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
    

if __name__ == '__main__':
    ###########################################################
    #model_name = 'dhf1k_kinetic_v4_s3d_label_oldschedule_rc_e1d3_3dd_d2d3'
    #reduced_channel = 1 #can only be 1, 2, 4
    #decoder_config = ['d1', 'd1', 'd3']
    #single_mode = [True, True, True]
    #inter_config = ['I', None, None] #temporal Mean or Identical
    #frame_num, sal_indx = 32, list(range(16, 32))
    #x_indx, teacher_indx = range(16, 32), range(16)
    ########################################################

    lr = 0.01
    batch_size = 12#15#9#15#18 #11
    save_path = '/home/feiyan/Github/Sal_dist/'
    save_path = '/home/feiyan/runs/'
    data_dir=['/home/feiyan/data/ucf_sport/', '/data/kinetics400/kinetics400/']
    teacher_path = '/home/feiyan/weight_MinLoss_UCF.pt'

    #frame_num, sal_indx = 16, list(range(15, 16))
    #x_indx, teacher_indx = range(0, 16), [range(0, 16)]
    frame_num, sal_indx = 32, list(range(16, 32))
    x_indx, teacher_indx = range(16, 32), [range(i, i+16) for i in range(16)]

    reduced_channel = 1
    decoder_config = ['d1', 'd2', 'd3']
    single_mode = [True, False, False]
    force_multi = False

    inter_config = ['M', None, None]
    model_name = 'ucf_kinetic_lt_myschedule_e1_d1s_d2m_d3m_update'
    main(model_name, reduced_channel, decoder_config, single_mode, inter_config,
         frame_num, sal_indx, x_indx, teacher_indx,
         lr, batch_size, save_path, data_dir, teacher_path, force_multi=force_multi)
    exit()
    
