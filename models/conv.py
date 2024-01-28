import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):
    def __init__(self,inchannels,num_classes,size,dropout,percent_dropout):
        super(Convnet,self).__init__()

        #building a neural netword with 3 conv layers
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=inchannels,            
                out_channels=16,            
                kernel_size=5,
                stride=1,                   
                padding=2,
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        if(dropout):
            self.conv1.add_module('drop1',nn.Dropout2d(p = percent_dropout))
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
    
        self.conv3 = nn.Sequential(         
            nn.Conv2d(32, 32, 5, 1, 2),     
            nn.ReLU(),             
        )
        if(dropout):
            self.conv1.add_module('drop2',nn.Dropout2d(p = percent_dropout))

        self.out = nn.Sequential(nn.Flatten(),nn.Linear(32 * int(size/4) * int(size/4), num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.out(x)
        return output