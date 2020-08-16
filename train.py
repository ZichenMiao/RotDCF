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
parser.add_argument('model_type', default='CNN', choices=['CNN', 'RotDCF', 'DCF'], help='Type of convolution layer')
parser.add_argument('model_depth', type=int, default=3, choices=[3, 6], help='Two networks: 3conv network and 6conv network, corresponds to two experiments')
parser.add_argument('--num_feat', type=int, default=8)
parser.add_argument('--Ntheta', type=int, default=8)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--Ka', type=int, default=5)

parser.add_argument('--trn_ver', default='uni_rot', help='version of trianing set, \
                                default is random rotation with angles sampled uniformally from [0, 360]')
parser.add_argument('--test_ver', default='uni_rot', help='version of test set')
parser.add_argument('--bs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

## import different models

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

model_type = args.model_type
model_depth = args.model_depth
M = args.num_feat
Ntheta = args.Ntheta
K = args.K
K_a = args.Ka

train_version = args.trn_ver
test_version = args.test_ver
batch_size = args.bs
lr = args.lr
epochs = args.epochs


## dataset & dataloader
trainset = MNIST_rot(model_depth, train=True, ver=train_version)
testset = MNIST_rot(model_depth, train=False, ver=test_version)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                            drop_last=True, num_workers=12)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            drop_last=False, num_workers=12)

## load and create different models
if model_depth == 3:
    from model_3conv import *
else:
    from model_6conv import *

if model_type == 'RotDCF':
    weight_decay = 5e-3
    model = MNIST_RotDCF_Net(M=M, Ntheta=Ntheta, K=K, K_a=K_a)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
## regular CNN
elif model_type == 'CNN':
    M = 32
    weight_decay = 1e-5
    model = MNIST_CNN_Net(M=M)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
## DCF
else:
    M = 32
    weight_decay = 1e-5
    model = MNIST_DCF_Net(M=M, K=K)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.cuda()
model = nn.DataParallel(model)

Loss = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, 
                                patience=10, min_lr=1e-5) 


best_acc = 0.0
start_time = time.time()
print('\n Use {} samples for training, {} samples for testing.'.format(
    trainset.images.shape[0], testset.images.shape[0]))
for epoch in range(1, epochs+1):
    model.train()
    loss_epoch = []
    num_correct_epoch = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        logits = model(images)
        loss = Loss(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.cpu().detach().item())
        num_correct_epoch += torch.sum(torch.argmax(logits, dim=1) == labels).item()
    print('Train: Epoch {}/{}; Avg. Loss {:.3f}; Acc {:.2f}; lr: {}, w_decay: {}'.format(epoch, epochs, 
                        np.mean(loss_epoch), 100.*num_correct_epoch/((i+1)*batch_size),
                        optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay']))

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
    print('Test: Epoch {}/{}; Avg. Loss {:.3f}; Acc {:.2f}'.format(epoch, epochs,
                        test_loss, test_acc))

    if best_acc < test_acc:
        best_acc = test_acc
        ## save best checkpoints
        if model_type == 'RotDCF':
            torch.save(model.state_dict(), 'checkpoints/{}layers/{}_M{}_Ntheta{}_K{}_Ka{}.pth'.\
                        format(model_depth, model_type, M, Ntheta, K, K_a))
        elif model_type == 'CNN':
            torch.save(model.state_dict(), 'checkpoints/{}layers/{}_M{}.pth'.\
                        format(model_depth, model_type, M))
        else:
            torch.save(model.state_dict(), 'checkpoints/{}layers/{}_M{}_K{}.pth'.\
                        format(model_depth, model_type, M, K))
        print('save best model')
    print('Best Acc: {:.2f}'.format(best_acc))

    ## learning rate adjustment
    scheduler.step(best_acc)                     

end_time = time.time()
total_time = end_time - start_time
print('Model: {}, time for total {} epoches train/test, use {:.2f} secs'.format(
                args.model_type, args.epochs, total_time))
