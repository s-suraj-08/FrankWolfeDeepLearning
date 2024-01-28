import torch.optim
from dfw import DFW
from dfw.baselines import BPGrad

def get_optimizer(opt,eta,regparam,momentum,parameters):

    if opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=eta, weight_decay=regparam,
                                    momentum=momentum, nesterov=bool(momentum))
    elif opt == 'dfw':
        optimizer = DFW(parameters, eta=eta, momentum=momentum, weight_decay=regparam)
    

    print("Optimizer: \t {}".format(opt.upper()))

    optimizer.gamma = 1
    optimizer.eta = eta

    return optimizer

def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        # update state
        optimizer.eta = optimizer.param_groups[0]['lr']
    else:
        raise ValueError