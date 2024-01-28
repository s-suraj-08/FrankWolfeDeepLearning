import torch
import torch.nn as nn
from torchvision import models

class VggModel(nn.Module):
    def __init__(self):
        super(VggModel, self).__init__()
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,10))
        
        self.cnn_vgg = models.vgg11_bn(pretrained=True)
        del self.cnn_vgg.avgpool
        del self.cnn_vgg.classifier
        self.cnn_vgg.avgpool = torch.nn.Identity()
        self.cnn_vgg.classifier = torch.nn.Sequential(*layers)        

    def forward(self, x):
        out = self.cnn_vgg(x)
        return out