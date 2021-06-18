import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import json
import cv2
import SimpleITK as sitk


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, list_folder=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.list_folder = list_folder
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open(os.path.join(self.list_folder, 'train.list'), 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[:num]
        elif split == 'val':
            with open(os.path.join(self.list_folder, 'train.list'), 'r') as f:
                self.image_list = f.readlines()
                if num is not None:
                    self.image_list = self.image_list[num:]
        elif split == 'test':
            with open(os.path.join(self.list_folder, 'test.list'), 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]

        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, image_name, "mri_norm2.h5"), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'name': image_name, 'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'name': name, 'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'name': name, 'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        name, image, label = sample['name'], sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'name': name, 'image': image, 'label': label}


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
        image, label = sample['image'], sample['label']
        if self.rotate_axes == (0, 0):
            sample = {'image': image, 'label': label}
            return sample
        # volume = rotate(volume, angle=self.rotate_angle, axes=(1, 0), reshape=False, order=1)
        # label = rotate(label, angle=self.rotate_angle, axes=(1, 0), reshape=False, order=0)
        image = np.rot90(image, k=1, axes=self.rotate_axes)
        label = np.rot90(label, k=1, axes=self.rotate_axes)
        sample = {'volume': image, 'label': label}

        return sample


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


def find_win(volume):  # https://github.com/pyushkevich/itksnap/search?q=accum_goal
    my_bins = 1000
    hist, bins = np.histogram(volume, bins=my_bins, range=(np.amin(volume), np.amax(volume)), density=False)
    totalSample = np.sum(hist)
    accum_goal = totalSample / 1000
    accum = 0
    ilow = bins[0]
    for i in range(my_bins):
        if accum + hist[i] < accum_goal:
            accum += hist[i]
            ilow = bins[i]
        else:
            break
    accum = 0
    ihigh = bins[my_bins - 1]
    for i in range(my_bins):
        j = my_bins - 1 - i
        if accum + hist[j] < accum_goal:
            accum += hist[j]
            ihigh = bins[j]
        else:
            break
    return ilow, ihigh


class Clip(object):
    def __init__(self, clip_min=0, clip_max=1400):
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        self.clip_min, self.clip_max = find_win(image)
        image = np.clip(image, self.clip_min, self.clip_max)
        sample = {'image': image, 'label': label}
        return sample


def hist_equal(data):
    l, w, h = data.shape
    for i in range(h):
        temp = data[:, :, i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        data[:, :, i] = clahe.apply(temp)
    return data


class Normalize(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image_max = np.amax(image)  # 等同于np.max
        image_min = np.amin(image)
        image = ((image-image_min)/(image_max-image_min))*255
        image = image.astype('uint8')
        image = hist_equal(image)
        # image = (image - image_max) / (image_max - image_min)
        image = image.astype(np.float)
        label = label.astype(np.uint8)
        sample = {'image': image, 'label': label}
        return sample


from scipy.ndimage import gaussian_filter, map_coordinates


class ElasticDeformation(object):
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order!
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """
    def __init__(self, random_state, spline_order, alpha=15, sigma=3):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        assert image.ndim == 3
        dx = gaussian_filter(self.random_state.randn(*image.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(self.random_state.randn(*image.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dz = gaussian_filter(self.random_state.randn(*image.shape), self.sigma, mode="constant", cval=0) * self.alpha

        x_dim, y_dim, z_dim = image.shape
        x, y, z = np.meshgrid(np.arange(x_dim), np.arange(y_dim), np.arange(z_dim), indexing='ij')
        indices = x + dx, y + dy, z + dz
        image = map_coordinates(image, indices, order=self.spline_order, mode='reflect')
        label = map_coordinates(label, indices, order=0, mode='reflect')
        return {'image': image, 'label': label}


class produceRandomlyDeformedImage(object):
    def __init__(self, numcontrolpoints=2, stdDef=15):
        self.numcontrolpoints = numcontrolpoints
        self.stdDef = stdDef

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        sitkImage = sitk.GetImageFromArray(image, isVector=False)
        sitklabel = sitk.GetImageFromArray(label, isVector=False)

        transfromDomainMeshSize = [self.numcontrolpoints]*sitkImage.GetDimension()  # [2,2,2]

        tx = sitk.BSplineTransformInitializer(sitkImage, transfromDomainMeshSize)

        params = tx.GetParameters()

        paramsNp = np.asarray(params, dtype=float)
        paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*self.stdDef

        paramsNp[0:int(len(params)/3)] = 0  # remove z deformations! The resolution in z is too bad  # ?

        params = tuple(paramsNp)
        tx.SetParameters(params)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)

        resampler.SetDefaultPixelValue(0)
        outimgsitk = resampler.Execute(sitkImage)
        outlabsitk = resampler.Execute(sitklabel)

        outimg = sitk.GetArrayFromImage(outimgsitk)
        outimg = outimg.astype(dtype=np.float32)

        outlbl = sitk.GetArrayFromImage(outlabsitk)
        outlbl = (outlbl > 0.5).astype(dtype=np.float32)

        return {'image': outimg, 'label': outlbl}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        name, image = sample['name'], sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'name': name, 'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size  # 4
        self.primary_batch_size = batch_size - secondary_batch_size  # 8-4

        assert len(self.primary_indices) >= self.primary_batch_size > 0  # 16>=4
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0  # 64>=4

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size  # 10//1


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    a = iterate_eternally(list(range(20, 83)))
    print(next(a))
    pass
