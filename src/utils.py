import torch as t
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class KLDLoss1vs1(nn.Module):
    def __init__(self, dev='cpu'):
        super(KLDLoss1vs1, self).__init__()
        self.dev=dev
        self.kld = nn.KLDivLoss(reduce=False)

    def KLD(self, inp, trg):
        assert inp.size(0)==trg.size(0), "Sizes of the distributions doesn't match"
        inp = inp.view(inp.size(0), -1)
        trg = trg.view(trg.size(0), -1)
        inp = t.log(inp/t.sum(inp, dim=1, keepdim=True))
        trg = trg/t.sum(trg, dim=1, keepdim=True)
        kldloss = self.kld(inp, trg)
        kldloss = t.mean(t.sum(kldloss, dim=1))
        return kldloss

    def forward(self, inp, trg, rand_sig=None):
        return self.KLD(inp, trg)

class my_scheduler:
    def __init__(self, optimizer, milestones, gamma, reload=True):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.reload = reload
        self.step_count = 0
    
    def step(self, model, weight_path):
        self.step_count += 1
        if self.step_count in self.milestones:
            print('Reload best model and reducing learning rate by {}'.format(self.gamma))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
            state_dict = t.load(weight_path)['student_model']
            model.load_state_dict(state_dict)

def eval(net):
    from ptflops import get_model_complexity_info
    with t.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 16, 192, 256), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))