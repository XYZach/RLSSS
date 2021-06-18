# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source UAMT_code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def standardized_seg(seg, label_list):
    """
    standardized seg_tensor with label list to generate a tensor,
    which can be put into nn.CrossEntropy(input, target) as "target"
    :param seg:
    :param label_list:
    :return:
    """
    result = torch.zeros(seg.shape, dtype=torch.long).cuda()
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(seg.shape), fill_value=label_class).cuda()
        label_seg = (label_mask == seg.float()).long()
        label_seg = label_seg * (index+1)
        result = torch.add(result, label_seg)
    return result


def transform_for_noise(input, sigma=0.05):
    gaussian = np.clip(sigma * np.random.randn(input.shape[0], input.shape[1], input.shape[2],
                                               input.shape[3], input.shape[4]),
                       -2*sigma, 2*sigma)
    gaussian = torch.from_numpy(gaussian).float().cuda()
    input_noise = input + gaussian
    return input_noise


def transform_for_rot(input):
    rot_mask = np.random.randint(0, 4, input.shape[0])
    flip_mask = np.random.randint(0, 2, input.shape[0])
    flip_dim = np.random.randint(1, 4, input.shape[0])

    for idx in range(input.shape[0]):
        if flip_mask[idx] == 1:
            input[idx] = torch.flip(input[idx], [flip_dim[idx]])
        input[idx] = torch.rot90(input[idx], int(rot_mask[idx]), dims=[1, 2])

    return input, rot_mask, flip_mask, flip_dim


def transforms_back_rot(output, rot_mask, flip_mask, flip_dim):
    for idx in range(output.shape[0]):
        output[idx] = torch.rot90(output[idx], int(4 - rot_mask[idx]), dims=[1, 2])
        if flip_mask[idx] == 1:
            output[idx] = torch.flip(output[idx], [flip_dim[idx]])
    return output


def makedir(path):
    """
    create directory if path to the directory is not existed
    :param path:(str) path to the directory
    :return:If create successfully then return True, otherwise return False
    """
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' create directory successfully!')
        return True
    else:
        value = input(f"\n{path} is already existed.Do you want to continue?[y/n]")
        if value is "Y" or value is "y":
            return False
        else:
            sys.exit()


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

