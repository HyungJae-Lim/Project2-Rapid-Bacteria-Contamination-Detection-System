import os
import random

import numpy as np
import scipy.io as io
from imageio import imread

import torch
from torch.utils import data

from data.preprocess import TRAIN_AUGS_2D
from data.preprocess import TEST_AUGS_2D
from data.preprocess import mat2npy

def find_classes(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(path, class_to_idx, frames,
                 valid='Bacometer4', test='Bacometer5'):
    clips = []
    images = []
    clips_train = []
    clips_valid = []
    clips_test = []

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
                if len(images) != 0 and len(images) % frames == 0:
                    label_baco = root.split('/')[-2]
                    if label_baco == valid:
                        clips_valid += images
                        images = []

                    else:
                        clips_train += images
                        images = []

    return clips_train, clips_valid, clips_test


def make_dataset_v2(path, class_to_idx, frames, fold=None):
    num_fold = 5
    set_size = 500

    clips = []
    images = []
    clips_train = []
    clips_valid = []
    clips_test = []

    quantity = set_size // num_fold
    cross_val_idx = quantity * fold

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)

                if len(images) != 0 and len(images) % set_size == 0:
                    if fold is not None:
                        clips_train += images[:set_size-cross_val_idx]
                        clips_valid += images[set_size-cross_val_idx: set_size + quantity - cross_val_idx]
                        clips_train += images[set_size + quantity -cross_val_idx:]
                        images = []

                    else:
                        clips_train += images[:set_size//5*4]
                        clips_valid += images[set_size//5*4:]
                        images = []

    return clips_train, clips_valid


def make_dataset_v3(path, class_to_idx, frames, val_sample):
    clips = []
    images = []
    clips_train = []
    clips_valid = []
    clips_test = []
    val_sample = [val_sample, str(int(val_sample) + 10)]

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
                if len(images) != 0 and len(images) % frames == 0:
                    sample_num = root.split('/')[-2]

                    for val in val_sample:
                        if sample_num == val:
                            clips_valid += images
                            images = []

                    clips_train += images
                    images = []

    return clips_train, clips_valid, clips_test


def SpeckleDataset(root, frames, valid, test, fold, val_sample):
    classes, class_to_idx = find_classes(root)
#    tr_imgs, val_imgs = make_dataset_v2(root, class_to_idx, frames, fold=fold)
#    tr_imgs, val_imgs, test_imgs = make_dataset(root, class_to_idx, frames, valid, test)
    tr_imgs, val_imgs, test_imgs = make_dataset_v3(root, class_to_idx, frames, val_sample)
    test_imgs = val_imgs

    origin_imgs = len(tr_imgs) + len(val_imgs) + len(test_imgs)
    print("{} origin : {}, aug: (Tr {}, Val {}, Test {})".format(
            root, origin_imgs, len(tr_imgs), len(val_imgs), len(test_imgs))
    )

    return tr_imgs, val_imgs, test_imgs


def test_SpeckleDataset(root, frames, blind=False):
    classes, class_to_idx = find_classes(root)
    test_imgs = real_test_make_dataset(root, class_to_idx, frames)
    return test_imgs

class _Dataset(data.Dataset):
    def __init__(self, img, frames, aug_rate=0, transform=None):
        self.imgs = img
        self.frames = frames
        self.origin_imgs = len(img)
        self.augs = [] if transform is None else transform

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

    def __getitem__(self, index):
        num_imgs = len(self.imgs[index])
        if num_imgs < self.frames:
            print('Error: T is too large | Num img:{} < T:{}'.format(num_imgs, self.frames))
            sys.exit()

        # Extract Sequential T Images
        imgs = np.array(self.imgs[index])
        path = imgs[0]
        target = imgs[1]
        img = np.array(imread(path))

        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img)
        else:
            for t in TEST_AUGS_2D:
                img = t(img)

        return img, target, path

    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images, classes):
    nclasses = len(classes)
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    print(classes)
    print(count)

    N = float(sum(count))
    assert N == len(images)

    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler


def SpeckleLoader(image_path, frames, batch_size, test_batch_size, sampler=False,
                 tr_transform=None, val_transform=None, test_transform=None,
                 tr_aug_rate=0, val_aug_rate=0, test_aug_rate=0, fold=None,
                 num_workers=1, shuffle=False, drop_last=False, test=False, val_sample=None,
                 baco_val='Bacometer4', baco_test='Bacometer5'):

    tr_imgs, val_imgs, test_imgs = SpeckleDataset(image_path, frames, baco_val, baco_test, fold, val_sample)
    tr_dataset = _Dataset(tr_imgs, frames, aug_rate=tr_aug_rate, transform=tr_transform)
    val_dataset = _Dataset(val_imgs, frames, aug_rate=val_aug_rate, transform=val_transform)
    test_dataset = _Dataset(val_imgs, frames, aug_rate=val_aug_rate, transform=val_transform)
#    test_dataset = _Dataset(test_imgs, frames, aug_rate=test_aug_rate, transform=test_transform)
    print('Data len: {} Training, {} Valid, {} Test'.format(len(tr_imgs), len(val_imgs), len(test_imgs)))

    if sampler:
        print("Sampler : ", image_path[-5:])
        if not test:
            tr_sampler = _make_weighted_sampler(tr_dataset.imgs)
            val_sampler = _make_weighted_sampler(val_dataset.imgs)
            tr_dataset = data.DataLoader(tr_dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
            val_dataset = data.DataLoader(val_dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)

        test_sampler = _make_weighted_sampler(test_dataset.imgs)
        test_dataset = data.DataLoader(test_dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        return tr_dataset, val_dataset, test_dataset

    else:
        test_dataset = data.DataLoader(test_dataset, test_batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        if not test:
            tr_dataset = data.DataLoader(tr_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
            val_dataset = data.DataLoader(val_dataset, test_batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
            return tr_dataset, val_dataset, test_dataset
        return test_dataset
