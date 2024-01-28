import torch
from .conv import Convnet
from .pretrain import VggModel

def get_model(dataset, m_type, device, dropout=False, percent_dropout=0.2,opt=""):

    assert dataset in ("cifar10", "mnist")

    inchannels = 3
    size = 32
    num_classes = 10
    
    params_to_update=[]
    nparams=0
    if(m_type == "cnn"):
        print("Model: cnn")
        model = Convnet(inchannels=inchannels,
                        num_classes=num_classes,
                        size=size,
                        dropout=dropout,
                        percent_dropout = percent_dropout).to(device=device)

        nparams = sum([p.data.nelement() for p in model.parameters()])
        params_to_update = model.parameters()
        print('Number of model parameters: {}'.format(nparams))

    elif(m_type=='pretrained'):
        print("Using pretrained VGG11, finetuning is assumed to be false")
        model = VggModel()
        nparams = sum([p.data.nelement() for p in model.parameters()])
        params_to_update = model.parameters()
        print('Number of model parameters: {}'.format(nparams))

    elif(m_type=='transfer'):
        print("Loading model trained on CIFAR10, to use on MNIST")
        model = Convnet(inchannels=inchannels,
                        num_classes=num_classes,
                        size=size,
                        dropout=dropout,
                        percent_dropout = percent_dropout).to(device=device)
        
        if(opt=="sgd"):
            print('loading best_sgd')
            model.load_state_dict(torch.load('./saved_model/best_sgd.pkl'))
        else:
            print('loading best_dfw')
            model.load_state_dict(torch.load('./saved_model/best_dfw.pkl'))
        
        print("model to be finetuned on MNIST")
        for name,param in model.named_parameters():
            if(name=='out.1.weight' or name=='out.1.bias'):
                param.requires_grad=True
                params_to_update.append(param)
            else:
                param.requires_grad = False

    return model,nparams,params_to_update