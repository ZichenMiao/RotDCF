import numpy as np
import torch
import torch.nn as nn

from Rot_DCF import Rot_DCF_Init, Rot_DCF, RotBN
from DCF import *

class MNIST_CNN_Net(nn.Module):
    def __init__(self, M=32):
        super(MNIST_CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, M, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=M)

        self.conv2 = nn.Conv2d(M, 2*M, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=2*M)
        
        self.conv3 = nn.Conv2d(2*M, 4*M, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=4*M)
        
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.adapt_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        num_params = np.sum([param.numel() for param in self.parameters()])

        self.fc1 = nn.Linear(3*3*4*M, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

        print('\nCreate CNN Net with num_features {}, Total Params: {}k'.\
                format(M, num_params/1000)) 

        print(self)

    def forward(self, x):
        bs, c, H, W = x.shape

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.avg_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.avg_pool(x)

        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class MNIST_DCF_Net(nn.Module):
    def __init__(self, M=32, K=5):
        super(MNIST_DCF_Net, self).__init__()
        ## num bases
        self.K = K

        self.conv1 = Conv_DCF(1, M, kernel_size=5, padding=2, num_bases=K, mode='mode0_1')
        self.bn1 = nn.BatchNorm2d(num_features=M)

        self.conv2 = Conv_DCF(M, 2*M, kernel_size=5, padding=2, num_bases=K, mode='mode0_1',)
        self.bn2 = nn.BatchNorm2d(num_features=2*M)
        
        self.conv3 = Conv_DCF(2*M, 4*M, kernel_size=5, padding=2, num_bases=K, mode='mode0_1')
        self.bn3 = nn.BatchNorm2d(num_features=4*M)
        
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.adapt_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        num_params = np.sum([param.numel() for param in self.parameters()])

        self.fc1 = nn.Linear(3*3*4*M, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

        print('\nCreate CNN Net with num_features {}, Total Params: {}k'.\
                format(M, num_params/1000)) 

        print(self)

    def forward(self, x):
        bs, c, H, W = x.shape

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.avg_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.avg_pool(x)

        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class MNIST_RotDCF_Net(nn.Module):
    def __init__(self, M=8, Ntheta=8, K=3, K_a=5):
        super(MNIST_RotDCF_Net, self).__init__()
        
        self.rot_conv1 = Rot_DCF_Init(M, 1, kernel_size=5, Ntheta=Ntheta, K=K, K_a=K_a, bias=True)
        self.bn1 = RotBN(M)

        self.rot_conv2 = Rot_DCF(2*M, M, kernel_size=5, Ntheta=Ntheta, K=K, K_a=K_a, bias=True)
        self.bn2 = RotBN(2*M)
        
        self.rot_conv3 = Rot_DCF(4*M, 2*M, kernel_size=5, Ntheta=Ntheta, K=K, K_a=K_a, bias=True)
        self.bn3 = RotBN(4*M)
        
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        num_params = np.sum([param.numel() for param in self.parameters()])

        self.fc1 = nn.Linear(3*3*Ntheta*4*M, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

        print('\nCreate Rot-DCF Net with num_features {}, Ntheta {}, K {}, K_a {}; Total Params: {}k'.\
                format(M, Ntheta, K, K_a, num_params/1000)) 
        print(self)

    def forward(self, x):
        bs, c, H, W = x.shape

        x = self.rot_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.avg_pool(x)

        x = self.rot_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.avg_pool(x)

        x = self.rot_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.avg_pool(x)

        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    def get_feature(self, x, idx):
        self.eval()
        bs, c, H, W = x.shape

        x = self.rot_conv1(x)
        if idx == 1:
            return x
        x = self.relu(x)
        x = self.bn1(x)
        x = self.avg_pool(x)

        x = self.rot_conv2(x)
        if idx == 2:
            return x
        x = self.relu(x)
        x = self.bn2(x)
        x = self.avg_pool(x)

        x = self.rot_conv3(x)
        if idx == 3:
            return x
        x = self.relu(x)
        x = self.bn3(x)
        x = self.avg_pool(x)

        


if __name__ == '__main__':
    x = torch.randn(4, 1, 28, 28)
    net = MNIST_Net(M=8, Ntheta=16)
    x = net(x)
    print(x.shape)