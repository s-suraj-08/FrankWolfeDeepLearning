import torch.nn as nn
from dfw.losses import MultiClassHingeLoss, set_smoothing_enabled


def get_loss(ls,regparam,device):
    if ls == 'svm':
        loss_fn = MultiClassHingeLoss()
    elif ls == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError

    print('Regularization param: \t {}'.format(regparam))
    print('Loss function:',loss_fn)
    
    loss_fn = loss_fn.to(device)
    return loss_fn
