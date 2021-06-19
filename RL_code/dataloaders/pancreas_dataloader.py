from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os
import nibabel as nib
import numpy as np
import torch.nn as nn
import torch
import random
import utils
from scipy.ndimage import rotate
import itertools


class Pancreas(Dataset):
    def __init__(self, vol_path, seg_path, transform=None):
        self.vol_path = vol_path
        self.seg_path = seg_path
        self.transform = transform

    def __getitem__(self, index):
        volume_path = self.vol_path[index]
        label_path = self.seg_path[index]

        volume = nib.load(volume_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        name = volume_path.split('/')[-1].split('.')[0]
        sample = {'name': name, 'volume': volume, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.vol_path)


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume = np.pad(volume, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = volume.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume = volume[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume = np.pad(volume, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = volume.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume = volume[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        k = np.random.randint(0, 4)
        volume = np.rot90(volume, k)
        label = np.rot90(label, k)

        axis = np.random.randint(0, 2)
        volume = np.flip(volume, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomRotate(object):
    def __init__(self, rotate_rate):
        self.rotate_rate = rotate_rate

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        if np.random.random() < self.rotate_rate:
            rand_angle = [90, 180, 270]
            np.random.shuffle(rand_angle)
            # volume = rotate(volume, angle=rand_angle[0], axes=(1, 0), reshape=False, order=1)
            # label = rotate(label, angle=rand_angle[0], axes=(1, 0), reshape=False, order=0)
            k = rand_angle[0] // 90
            volume = np.rot90(volume, k)
            label = np.rot90(label, k)
        sample = {'name': name, 'volume': volume, 'label': label}

        return sample


class Rotate(object):
    """
    rotate the volume and label by rotate_angle
    """
    def __init__(self, rotate_axes):
        self.rotate_axes = rotate_axes
        # (轴0-x,轴1-y,轴2-z)
        # rotate_axes
        # (0,2) x轴向z轴转
        # (1,2) y轴向z轴转

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        if self.rotate_axes == (0, 0):
            sample = {'name': name, 'volume': volume, 'label': label}
            return sample
        # volume = rotate(volume, angle=self.rotate_angle, axes=(1, 0), reshape=False, order=1)
        # label = rotate(label, angle=self.rotate_angle, axes=(1, 0), reshape=False, order=0)
        volume = np.rot90(volume, k=1, axes=self.rotate_axes)
        label = np.rot90(label, k=1, axes=self.rotate_axes)
        sample = {'name': name, 'volume': volume, 'label': label}

        return sample


class Flip(object):
    def __init__(self, flip_rate):
        self.flip_rate = flip_rate

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        if np.random.random() < self.flip_rate:
            volume = volume[:, :, ::-1]
            label = label[:, :, ::-1]
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.05):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(volume.shape[0], volume.shape[1], volume.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        volume = volume + noise
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class RandomAxisFlip(object):
    """
    random axis mirror flip (for all 3 axes) with a probability 0.5.
    """
    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        axis_random = np.random.randn(3)
        for index, flag in enumerate(axis_random):
            if flag >= 0:
                volume = np.flip(volume, axis=index)
                label = np.flip(label, axis=index)
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class ToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        name, volume, label = sample['name'], sample['volume'], sample['label']
        volume = torch.Tensor(np.expand_dims(volume, axis=0).copy())
        label = torch.Tensor(np.expand_dims(label, axis=0).copy())
        sample = {'name': name, 'volume': volume, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    # 12 * 50
    # def __iter__(self):
    #     primary_iter = iterate_once(self.primary_indices)
    #     secondary_iter = iterate_eternally(self.secondary_indices)
    #     return (
    #         primary_batch + secondary_batch
    #         for (primary_batch, secondary_batch)
    #         in zip(grouper(primary_iter, self.primary_batch_size),
    #                 grouper(secondary_iter, self.secondary_batch_size))
    #     )
    # def __len__(self):
    #     return len(self.primary_indices) // self.primary_batch_size

    # 50 * 12
    def __iter__(self):
        secondary_iter = iterate_once(self.secondary_indices)
        primary_iter = iterate_eternally(self.primary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.secondary_indices) // self.secondary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    " Collect data into fixed-length chunks or blocks "
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



if __name__ == "__main__":
    from torchvision import transforms
    from utils import util
    from torch.utils.data import DataLoader
    batch_size = 3
    labeled_bs = 2
    data_path = '/data1/hra/dataset/Pancreas/'

    total_volume_path, total_label_path = util.gen_Pancreas_data_path(data_path)

    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 60, 1124)

    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 60))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    dataset = Pancreas(train_volume_path, train_volume_path,
                       transform=transforms.Compose([RandomNoise(), Rotate(0.5), Flip(0.5), ToTensor()]))
    def worker_init_fn(worker_id):
        random.seed(worker_id)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn(1))

    for epoch_num in range(3):
        for i_batch, sampled_batch in enumerate(dataloader):
            name_batch = sampled_batch['name']
            print('d')

