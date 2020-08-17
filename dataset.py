## custom dataset for rotation MNIST
import torch
import os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import PIL

data_path = '/home/zichen/Projects/data/MNIST/processed/'
save_path = '/home/zichen/Projects/data/MNIST/rotate'

TOTAL_LENGTH = 70000
def rotate_MNIST_permute(model_depth):
    assert model_depth in [3, 6]
    if model_depth == 3:
        ntr, nval, ntest = 10000, 5000, 50000
    else:
        ntr, nval, ntest = 12000, 2000, 50000

    ## reconstruct original total dataset
    ori_trn_set = torch.load(os.path.join(data_path, 'training.pt'))
    ori_tst_set = torch.load(os.path.join(data_path, 'test.pt'))
    imgs_trn_ori, labels_trn_ori = ori_trn_set
    imgs_test_ori, labels_test_ori = ori_tst_set
    total_imgs = torch.cat([imgs_trn_ori, imgs_test_ori])
    total_labels = torch.cat([labels_trn_ori, labels_test_ori])

    ## random permute total dataset, and partition training/validating/testing set
    rand_indices = torch.randperm(TOTAL_LENGTH)
    total_imgs = total_imgs[rand_indices]
    total_labels = total_labels[rand_indices]
    trn_imgs, val_imgs, test_imgs = total_imgs[:ntr], total_imgs[ntr:ntr+nval], total_imgs[ntr+nval: ntr+nval+ntest]
    trn_labels, val_labels, test_labels = total_labels[:ntr], total_labels[ntr:ntr+nval], total_labels[ntr+nval: ntr+nval+ntest]

    ## using ori testing set as training set, save both upright version and rotated version
    torch.save((trn_imgs, trn_labels), os.path.join(save_path, '{}layers/trainig_upright.pt'.format(model_depth)))
    new_images = []
    for image in trn_imgs:
        # transfer tensor to PIL image
        image = transforms.functional.to_pil_image(image)
        angle = np.random.randint(0, 360)
        new_image = transforms.functional.rotate(image, angle, resample=PIL.Image.BILINEAR)
        new_images.append(transforms.functional.to_tensor(new_image))
    new_images = torch.stack(new_images, dim=0)
    torch.save((new_images, trn_labels), os.path.join(save_path, '{}layers/training_uni_rot.pt'.format(model_depth)))
    
    # save random rotated versions with sample intervals of [0, 360], [-30, 30], [-60, 60]
    new_images_uni_rot = []
    new_images_30_rot = []
    new_images_60_rot = []
    for image in test_imgs:
        image = transforms.functional.to_pil_image(image)
        angle = np.random.randint(0, 360)
        new_images_uni_rot.append(transforms.functional.to_tensor(transforms.functional.rotate(image, angle, resample=PIL.Image.BILINEAR)))
        angle = np.random.randint(-30, 30)
        new_images_30_rot.append(transforms.functional.to_tensor(transforms.functional.rotate(image, angle, resample=PIL.Image.BILINEAR)))
        angle = np.random.randint(-60, 60)
        new_images_60_rot.append(transforms.functional.to_tensor(transforms.functional.rotate(image, angle, resample=PIL.Image.BILINEAR)))
    new_images_uni_rot = torch.stack(new_images_uni_rot)
    new_images_30_rot = torch.stack(new_images_30_rot)
    new_images_60_rot = torch.stack(new_images_60_rot)

    torch.save((new_images_uni_rot, test_labels), os.path.join(save_path, '{}layers/test_uni_rot.pt'.format(model_depth)))
    torch.save((new_images_30_rot, test_labels), os.path.join(save_path, '{}layers/test_30_rot.pt'.format(model_depth)))
    torch.save((new_images_60_rot, test_labels), os.path.join(save_path, '{}layers/test_60_rot.pt'.format(model_depth)))


class MNIST_rot(torch.utils.data.Dataset):
    def __init__(self, model_depth, data_path=save_path, train=False, ver='uni_rot', transform=None):
        super(MNIST_rot, self).__init__()
        self.transform = transform
        
        if train:
            data_file = '{}layers/training_{}.pt'.format(model_depth, ver)
        else:
            data_file = '{}layers/test_{}.pt'.format(model_depth, ver)
        self.images, self.labels = torch.load(os.path.join(data_path, data_file))

    def __len__(self, ):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.images[idx]

        return img, label
    
