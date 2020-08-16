import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from init_bases import initialize_bases



class Rot_DCF_Init(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size, Ntheta, K, K_a,
                 stride=1, bias=True, trainable_bases=False):
        super(Rot_DCF_Init, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size_s = kernel_size
        self.Ntheta = Ntheta
        self.K = K
        self.K_a = K_a
        self.padding = int((self.kernel_size_s-1)/2)
        self.stride = stride
        
        ## init spatial FB bases and rotation F bases
        s_bases, rot_mtx_s, r_bases, rot_mtx_r = initialize_bases('FB_FOUR', 
                self.kernel_size_s, self.Ntheta, self.K, self.K_a)
        rot_mtx_s = torch.from_numpy(np.stack(rot_mtx_s)).float()
        rot_mtx_r = torch.from_numpy(np.stack(rot_mtx_r)).float()

        # with shape [k_s, k_s, K]
        self.register_buffer('s_bases', torch.from_numpy(s_bases).float())
        # list of rotation matrices(len=Ntheta) for spatial bases, each with shape [K, K]
        self.register_buffer('rot_mtx_s', rot_mtx_s)

        ## init coeff.
        self.joint_coeff = nn.Parameter(torch.Tensor(K, self.out_channels, self.in_channels))
        stdv = 1. / math.sqrt(self.joint_coeff.shape[0]*self.joint_coeff.shape[2])
        #Normal works better, working on more robust initializations
        init.normal(self.joint_coeff, mean=0, std=stdv)                                             

        ## init bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            init.normal(self.bias, mean=0, std=1./math.sqrt(self.out_channels))
        else:
            self.bias = None

    def rototion_weight(self,):
        rot_kernel = []
        bases = self.s_bases.view(-1, self.K)
        coeff = self.joint_coeff.view(self.K, -1)
        
        for i in range(self.Ntheta):
            # append [k_s, k_s, chn_out, chn_in]
            rot_kernel.append(torch.matmul(bases, 
                            torch.matmul(self.rot_mtx_s[i].transpose(1,0), coeff)).\
                            view(self.kernel_size_s, self.kernel_size_s, self.out_channels, self.in_channels))
        
        # with shape [k_s, k_s, Ntheta, chn_out, chn_in]
        rot_kernel = torch.stack(rot_kernel, dim=2)
        # with shape [k_s, k_s, Ntheta*chn_out, chn_in]
        rot_kernel = rot_kernel.view(self.kernel_size_s, self.kernel_size_s, -1, self.in_channels)
        # with shape [Ntheta*chn_out, chn_in, k_s, k_s]
        rot_kernel = rot_kernel.permute(2, 3, 0, 1).contiguous()

        return rot_kernel


    def forward(self, x):
        bs, chn_in, H, W = x.shape
        
        rot_kernel = self.rototion_weight()
        if self.bias is not None:
            bias = self.bias.repeat(self.Ntheta)
        else:
            bias = None
        x = F.conv2d(x, rot_kernel, bias, self.stride, self.padding)

        return x



class Rot_DCF(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size, Ntheta, K, K_a,
                 stride=1, bias=True, trainable_bases=False):
        super(Rot_DCF, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size_s = kernel_size
        self.Ntheta = Ntheta
        self.K = K
        self.K_a = K_a
        self.padding = int((self.kernel_size_s-1)/2)
        self.stride = stride
        
        ## init spatial FB bases and rotation F bases
        s_bases, rot_mtx_s, r_bases, rot_mtx_r = initialize_bases('FB_FOUR', 
                self.kernel_size_s, self.Ntheta, self.K, self.K_a)
        rot_mtx_s = torch.from_numpy(np.stack(rot_mtx_s)).float()
        rot_mtx_r = torch.from_numpy(np.stack(rot_mtx_r)).float()

        # with shape [k_s, k_s, K]
        self.register_buffer('s_bases', torch.from_numpy(s_bases).float())
        # list of rotation matrices(len=Ntheta) for spatial bases, each with shape [K, K]
        self.register_buffer('rot_mtx_s', rot_mtx_s)
        # with shape [K_a, Ntheta]
        self.register_buffer('r_bases', torch.from_numpy(r_bases).float())
        # list of rotation matrices(len=Ntheta) for fourier bases, each with shape [K_a, K_a]
        self.register_buffer('rot_mtx_r', rot_mtx_r)

        ## init coeff. with shape [K, chn_out, chn_in, K_a]
        self.joint_coeff = nn.Parameter(torch.Tensor(K, self.out_channels, self.in_channels,
                                                        K_a))
        stdv = 1. / math.sqrt(self.joint_coeff.shape[0]*self.joint_coeff.shape[2]*self.joint_coeff.shape[3])
        #Normal works better, working on more robust initializations
        init.normal(self.joint_coeff, mean=0, std=stdv)                                             

        ## init bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
            init.normal(self.bias, mean=0, std=1./math.sqrt(self.out_channels))
        else:
            self.bias = None

    def rotation_weight(self,):
        """
          given reconstructed weight [chn_out, chn_in, Ntheta, k_s, k_s]
          rotate in space and rotation domain to get [Ntheta*chn_out, chn_in, Ntheta, k_s, k_s]
          Actually, Only Rotate Coefficient
        """
        rot_kernel = []
        s_bases = self.s_bases.view(-1, self.K)
        r_bases = self.r_bases
        # with shape [K, chn_out*chn_in*K_a]
        jcoeff = self.joint_coeff.view(self.K, -1)

        for i in range(self.Ntheta):
            ## rotation with spatial rot mat, with shape [K_a, K, chn_out, chn_in]
            rot_jcoeff = torch.matmul(self.rot_mtx_s[i].transpose(1,0), jcoeff).\
                            view(self.K, self.out_channels, self.in_channels, self.K_a).\
                            permute(3, 0, 1, 2).contiguous()
            # with shape [K_a, K*chn_out*chn_in]
            rot_jcoeff = rot_jcoeff.view(self.K_a, -1)
            
            ## rotation with fourier rot mat, with shape [K, chn_out, chn_in, K_a]
            rot_jcoeff = torch.matmul(self.rot_mtx_r[i], rot_jcoeff).\
                            view(self.K_a, self.K, self.out_channels, self.in_channels)
            rot_jcoeff = rot_jcoeff.permute(1, 2, 3, 0).contiguous()
            
            ## reconstruct joint bases
            # with shape [k_s*k_s, chn_out*chn_in*K_a]
            rec_bases = torch.matmul(s_bases, rot_jcoeff.view(self.K, -1))
            # with shape [k_s*k_s*chn_out*chn_in, Ntheta]
            rec_bases = torch.matmul(rec_bases.view(-1, self.K_a), r_bases).contiguous()
            
            ## adjust dimensions to [chn_out, Ntheta*chn_in, k_s, k_s]
            rec_bases = rec_bases.view(self.kernel_size_s, self.kernel_size_s, self.out_channels,
                                    self.in_channels, self.Ntheta)
            rec_bases = rec_bases.permute(2, 4, 3, 0, 1).contiguous()
            rec_bases = rec_bases.view(self.out_channels, -1, self.kernel_size_s, self.kernel_size_s)
            rot_kernel.append(rec_bases)

        # with shape [Ntheta*chn_out, Ntheta*chn_in, k_s, k_s]
        rot_kernel = torch.stack(rot_kernel, dim=0)
        rot_kernel = rot_kernel.view(self.Ntheta*self.out_channels, self.Ntheta*self.in_channels,
                                        self.kernel_size_s, self.kernel_size_s).contiguous()
        
        return rot_kernel

    def forward(self, x):
        bs, C, H, W = x.shape
        rot_kernel = self.rotation_weight()
        if self.bias is not None:
            bias = self.bias.repeat(self.Ntheta)
        else:
            bias = None
        x = F.conv2d(x, rot_kernel, bias, self.stride, self.padding)

        return x


class RotBN(nn.Module):
    ## do the same batch normalization across alpha dimension
    ## [Bs, Ntheta*num_feat, L, L] -> [Bs, num_feat, Ntheta*L, L] -> BN2d(num_feat) -> 
    ## [Bs, Ntheta*num_feat, L, L]
    def __init__(self, chn):
        super(RotBN, self).__init__()
        self.chn = chn
        self.bn = nn.BatchNorm2d(self.chn)

    def forward(self, x):
        ## input shape [Bs, Ntheta*num_feat, L, L]
        bs, _, H, W = x.shape
        ## with shape [Bs, num_feat, Ntheta*L, L]
        x = x.view(bs, -1, self.chn, H, W).permute(0, 2, 1, 3, 4).contiguous().\
                    view(bs, self.chn, -1, W)

        ## do bn on feature dimension
        x = self.bn(x)
        ## with shape [Bs, Ntheta*num_feat, L, L]
        x = x.view(bs, self.chn, -1, H, W).permute(0, 2, 1, 3, 4).contiguous().\
                    view(bs, -1, H, W)

        return x


if __name__ == "__main__":
    rot_dcf_1 = Rot_DCF_Init(32, 1, kernel_size=5, Ntheta=16, K=14, K_a=16)
    rot_dcf = Rot_DCF(64, 32, kernel_size=5, Ntheta=16, K=14, K_a=9)
    
    x = torch.randn(1, 1, 28, 28)
    print(x.shape)
    x = rot_dcf_1(x)
    print(x.shape)
    x = rot_dcf(x)
    print(x.shape)