import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Rot_DCF import Rot_DCF_Init, Rot_DCF, RotBN
from DCF import *

class Conv_Module(nn.Module):
    def __init__(self, layer_type, feat_in, feat_out, first_rot=False, Ntheta=8,
                    K=3, K_a=5):
        super(Conv_Module, self).__init__()
        if layer_type == 'plain_conv':
            self.conv = nn.Conv2d(feat_in, feat_out, kernel_size=5, padding=2)
            self.bn = nn.BatchNorm2d(feat_out)
        elif layer_type == 'rotdcf_conv':
            if first_rot:
                self.conv = Rot_DCF_Init(feat_out, feat_in, kernel_size=5, Ntheta=Ntheta,
                            K=K, K_a=K_a)
            else:
                self.conv = Rot_DCF(feat_out, feat_in, kernel_size=5, Ntheta=Ntheta,
                            K=K, K_a=K_a)
            self.bn = RotBN(feat_out)
        else:
            self.conv = Conv_DCF(feat_in, feat_out, kernel_size=5, padding=2, num_bases=K, mode='mode0_1')
            self.bn = nn.BatchNorm2d(feat_out)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)

        return x


class MNIST_CNN_Net(nn.Module):
    def __init__(self, M=32):
        super(MNIST_CNN_Net, self).__init__()
        self.conv_layers = nn.Sequential(
            Conv_Module('plain_conv', feat_in=1, feat_out=M),
            Conv_Module('plain_conv', feat_in=M, feat_out=int(1.5*M)),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('plain_conv', feat_in=int(1.5*M), feat_out=2*M),
            Conv_Module('plain_conv', feat_in=2*M, feat_out=2*M),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('plain_conv', feat_in=2*M, feat_out=3*M),
            Conv_Module('plain_conv', feat_in=3*M, feat_out=4*M),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
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

        x = self.conv_layers(x)

        x = x.view(bs, -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class MNIST_DCF_Net(nn.Module):
    def __init__(self, M=32, K=5):
        super(MNIST_DCF_Net, self).__init__()
        self.conv_layers = nn.Sequential(
            Conv_Module('dcf_conv', feat_in=1, feat_out=M),
            Conv_Module('dcf_conv', feat_in=M, feat_out=int(1.5*M)),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('dcf_conv', feat_in=int(1.5*M), feat_out=2*M),
            Conv_Module('dcf_conv', feat_in=2*M, feat_out=2*M),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('dcf_conv', feat_in=2*M, feat_out=3*M),
            Conv_Module('dcf_conv', feat_in=3*M, feat_out=4*M),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        
        # self.adapt_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        num_params = np.sum([param.numel() for param in self.parameters()])

        self.fc1 = nn.Linear(3*3*4*M, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

        print('\nCreate DCF Net with num_features {}, K {}, Total Params: {}k'.\
                format(M, self.K, num_params/1000)) 

        print(self)

    def forward(self, x):
        bs, c, H, W = x.shape

        x = self.conv_layers(x)

        x = x.view(bs, -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x




class MNIST_RotDCF_Net(nn.Module):
    def __init__(self, M=8, Ntheta=8, K=3, K_a=5):
        super(MNIST_RotDCF_Net, self).__init__()
        
        self.conv_layers = nn.Sequential(
            Conv_Module('rotdcf_conv', 1, M, Ntheta=Ntheta, K=K, K_a=K_a, first_rot=True),
            Conv_Module('rotdcf_conv', M, int(1.5*M), Ntheta=Ntheta, K=K, K_a=K_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('rotdcf_conv', int(1.5*M), 2*M, Ntheta=Ntheta, K=K, K_a=K_a),
            Conv_Module('rotdcf_conv', 2*M, 2*M, Ntheta=Ntheta, K=K, K_a=K_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv_Module('rotdcf_conv', 2*M, 3*M, Ntheta=Ntheta, K=K, K_a=K_a),
            Conv_Module('rotdcf_conv', 3*M, 4*M, Ntheta=Ntheta, K=K, K_a=K_a),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        num_params = np.sum([param.numel() for param in self.parameters()])

        self.fc1 = nn.Linear(3*3*Ntheta*4*M, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

        print('\nCreate Rot-DCF Net with num_features {}, Ntheta {}, K {}, K_a {}; Total Params: {}k'.\
                format(M, Ntheta, K, K_a, num_params/1000)) 
        print(self)

    def forward(self, x):
        bs, c, H, W = x.shape

        x = self.conv_layers(x)

        x = x.view(bs, -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)

        return x



if __name__ == '__main__':
    x = torch.randn(4, 1, 28, 28)
    net = MNIST_Net(M=8, Ntheta=16)
    x = net(x)
    print(x.shape)