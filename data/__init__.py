import sys
from data.loaders import loaders_cifar,loaders_mnist

def get_data_loaders(dataset,root,batch_size,augment=False):
    if dataset == 'mnist':
        loader_train, loader_val, loader_test = loaders_mnist(root=root,batch_size=batch_size)
    elif dataset=='cifar10':
        loader_train, loader_val, loader_test = loaders_cifar(root=root,batch_size=batch_size,augment=augment)
    
    return loader_train, loader_val, loader_test