import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import time

from dataset import MNIST_rot

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='CNN', choices=['CNN', 'RotDCF', 'DCF'], help='Type of convolution layer')
parser.add_argument('--model_depth', type=int, default=3, choices=[3, 6], help='Two networks: 3conv network and 6conv network, corresponds to two experiments')
parser.add_argument('--num_feat', type=int, default=8)
parser.add_argument('--Ntheta', type=int, default=8)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--Ka', type=int, default=5)

parser.add_argument('--test_ver', default='uni_rot', help='version of test set')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

model_type = args.model_type
model_depth = args.model_depth
M = args.num_feat
Ntheta = args.Ntheta
K = args.K
K_a = args.Ka
batch_size = 100

test_version = args.test_ver


## dataset & dataloader
testset = MNIST_rot(model_depth, train=False, ver=test_version)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            drop_last=False, num_workers=12)

## load, create different models
if model_depth == 3:
    from model_3conv import MNIST_CNN_Net, MNIST_RotDCF_Net, MNIST_DCF_Net
else:
    from model_6conv import MNIST_CNN_Net, MNIST_RotDCF_Net, MNIST_DCF_Net

if model_type == 'RotDCF':
    model = MNIST_RotDCF_Net(M=M, Ntheta=Ntheta, K=K, K_a=K_a)
    path_ckpt = 'checkpoints/{}layers/{}_M{}_Ntheta{}_K{}_Ka{}.pth'.\
                        format(model_depth, model_type, M, Ntheta, K, K_a)
elif model_type == 'CNN':
    M = 32
    model = MNIST_CNN_Net(M=M)
    path_ckpt = 'checkpoints/{}layers/{}_M{}.pth'.format(model_depth, model_type, M)
else:
    M=32
    model = MNIST_DCF_Net(M=M, K=K)
    path_ckpt = 'checkpoints/{}layers/{}_M{}_K{}.pth'.format(model_depth, model_type, M, K)

model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(path_ckpt))


Loss = nn.CrossEntropyLoss()

model.eval()
loss_epoch = []
num_correct_epoch = 0
for i, (images, labels) in enumerate(test_loader):
    images, labels = images.cuda(), labels.cuda()
    logits = model(images)
    loss = Loss(logits, labels)
    loss_epoch.append(loss.cpu().detach().item())
    num_correct_epoch += torch.sum(torch.argmax(logits, dim=1) == labels).item()

test_loss = np.mean(loss_epoch)
test_acc = 100.*num_correct_epoch/len(testset)
print('Test: Avg. Loss {:.3f}; Acc {:.2f}'.format(test_loss, test_acc))