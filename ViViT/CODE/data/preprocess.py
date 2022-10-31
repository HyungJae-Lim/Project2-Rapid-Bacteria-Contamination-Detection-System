import random

import torch

import numpy as np

from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt


def mat2npy(mat, **kwargs):
    img = mat['data']
    img = img.astype('float')
    #target_regression = mat['target_regression']
    return img#, target_regression


def crop_2d(img, target_shape=[96, 96], **kwargs):
    origin_size = img.shape
    rand_1 = random.randint(0, origin_size[0] - target_shape[0])
    rand_2 = random.randint(0, origin_size[1] - target_shape[1])
    img = img[rand_1:rand_1 + target_shape[0], rand_2:rand_2 + target_shape[1]]
    return img


def center_crop_2d(img, target_shape=[96, 96], **kwargs):
    origin_size = img.shape
    middle = origin_size[0] // 2
    half = target_shape[0] // 2
    img = img[middle - half:middle + half, middle - half:middle + half]
    return img


def gaussian_2d(img, **kwargs):
    size = img.shape
    sigma = random.uniform(0.01, 0.05)
    noise = np.random.normal(0, sigma, size=size)
    img = img + noise
    return img


def calibration(img, **kwargs):
    ri = img["ri"]
    img_calibrated = (img - ri) * 9
    img_calibrated[img_calibrated < 0] = 0
    img = img_calibrated
    return img


def no_calibration(img, **kwargs):
    return img


def flipud_2d(img, **kwargs):
    rand = random.randint(0, 1)
    if rand == 0:
        return img[::-1, :]
    else:
        return img


def fliplr_2d(img, **kwargs):
    rand = random.randint(0, 1)
    if rand == 0:
        return img[:, ::-1]
    else:
        return img


# https://github.com/scipy/scipy/issues/5925
def rotate_2d(img, **kwargs):
    # rand = random.randint(1,360)
    rand = random.randrange(0, 360, 90)
    return ndimage.interpolation.rotate(img, rand, reshape=False, order=0, mode='reflect')


def to_tensor(img, **kwargs):
    img = torch.from_numpy(img)
    t, h, w = img.shape
    return img.reshape(t, -1, h, w).float()
#    return img.permute(0,3,1,2).float()


# (1, 1), (5, 2), (1, 0.5), (1, 3)
def elastic_transform(img, alpha=0, sigma=0, random_state=None, **kwargs):
    param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]
    rand = random.randint(0, 3)
    alpha, sigma = param_list[rand]

    # alpah = [1,5], sigma =[0.5,3]
    # alpha = random.uniform(1,1)
    # sigma = random.uniform(1,3)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(img, indices, order=1).reshape(shape)


TRAIN_AUGS_2D = [
#    flipud_2d,
#    fliplr_2d,
#    rotate_2d,
    to_tensor
]

TEST_AUGS_2D = [
    to_tensor
]


