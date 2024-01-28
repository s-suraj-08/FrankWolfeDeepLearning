import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def loaders_cifar(root,batch_size,augment):

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if augment:
        print('Using data augmentation on CIFAR data set.')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        print('Not using data augmentation on CIFAR data set.')
        transform_train = transform_test

    dataset = datasets.CIFAR10
    dataset_train = dataset(root=root, train=True,
                            transform=transform_train)
    dataset_test = dataset(root=root, train=False,
                           transform=transform_test)
    
    mask = list(range(45000))
    train_dataset = data.Subset(dataset_train, mask)
    mask = list(range(45000,45000+5000))
    val_dataset = data.Subset(dataset_train, mask)

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    test_loader = data.DataLoader(dataset=dataset_test,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_loader,val_loader,test_loader


def loaders_mnist(root,batch_size):
    print(root)
    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.Resize((32,32)),
                                    transforms.ToTensor(), 
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    normalize])

    # define two datasets in order to have different transforms
    # on training and validation
    dataset_train = datasets.MNIST(root=root, train=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, transform=transform)
    
    mask = list(range(50000))
    train_dataset = data.Subset(dataset_train, mask)
    mask = list(range(50000, 50000 + 10000))
    val_dataset = data.Subset(dataset_train, mask)

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    test_loader = data.DataLoader(dataset=dataset_test,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_loader,val_loader,test_loader