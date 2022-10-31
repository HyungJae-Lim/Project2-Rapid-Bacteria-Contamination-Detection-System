# Bacteria Classification Preprocess
# Author : Gunho Choi, Daewoong Ahn
# 2.5D Image(Matlab Format) Loader

import os
import random
from glob import glob

import scipy.io as io

import torch
from torch.utils import data

from datas.BacLoader import find_classes, find_classes_multi, class_meta_data, task_meta
from datas.preprocess2d import TRAIN_AUGS_2D, TEST_AUGS_2D, mat2npy


def make_dataset(dir, class_to_idx, task):
    images = []
    max_len = 0
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        target = class_meta_data[target][task_meta[task]]
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                z_shape = io.loadmat(path)["data"].shape[-1]

                # for z_dim in range(10, 30):
                for z_dim in range(z_shape):
                    item = (path, class_to_idx[target], z_dim)
                    images.append(item)
    return images


class BacDataset_2d(data.Dataset):
    def __init__(self, root, aug_rate=0, transform=None, task="bac"):
        classes, class_to_idx = find_classes(root, task)
        self.imgs = make_dataset(root, class_to_idx, task)
        
        if len(self.imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root))

        self.origin_imgs = len(self.imgs)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root ))

        
        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))
        print(root, "origin : ", self.origin_imgs, ", aug : ", len(self.imgs))

        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.augs = [] if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, z_dim = self.imgs[index]
        mat = io.loadmat(path)
        img, ri = mat2npy(mat)
        img = img[:, :, z_dim]

        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_2D:
                img = t(img, ri=ri)
        return img, target, path

    def __len__(self):
        return len(self.imgs)


def _make_weighted_sampler(images):                        
    nclasses = len(class_meta_data.keys())
    count = [0] * nclasses                                                      
    for item in images:
        count[item[1]] += 1
    print(count)
    N = float(sum(count))
    assert N == len(images)
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):        
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler       


def bacLoader_2d(image_path, batch_size, task="bac", sampler=False,
                 transform=None, aug_rate=0,
                 num_workers=1, shuffle=False, drop_last=False):
    dataset = BacDataset_2d(image_path, task=task, transform=transform, aug_rate=aug_rate)
    if sampler:
        print("Sampler : ", image_path[-5:])
        sampler = _make_weighted_sampler(dataset.imgs)
        return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


if __name__ == "__main__":
    data_path = "/data2/UTI/200930_bac/Bacteria/train"

    import preprocess_25d as preprocess
    train_preprocess = preprocess.get_preprocess("train")
    train_loader = bacLoader_25d(data_path, 2, "bac",
                             transform=train_preprocess,
                             num_workers=1, infer=False, shuffle=True, drop_last=True)

    for input, target_, _ in train_loader:
        print("input :", input.shape)
