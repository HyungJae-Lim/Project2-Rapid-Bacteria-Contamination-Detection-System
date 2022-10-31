import os
import copy
import random

import numpy as np
import scipy.io as io
#from scipy.misc import imread
from imageio import imread

import torch
from torch.utils import data

from data.preprocess import TRAIN_AUGS_2D
from data.preprocess import TEST_AUGS_2D
from data.preprocess import mat2npy

def find_classes(path, task="bac"):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(path, class_to_idx, frames, task="bac"):
    if task == "multi" or task == "cascade":
        task = "bac"

    clips = []
    con_clips_train = []
    con_clips_valid = []
    con_clips_test = []
    uncon_clips_train = []
    uncon_clips_valid = []
    uncon_clips_test = []
    images = []

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(list(item))
                if len(images) != 0 and len(images) % frames == 0:
                    clips.append(images)
                    images = []

        num = len(clips)
        num_tr = num // 10 * 8
        num_val = (num - num_tr) // 2
        clips = np.array(clips)
        if target == sorted(os.listdir(path))[0]: # Contaminated Samples
            con_clips_train += clips[:num_tr].tolist()
            con_clips_valid += clips[num_tr : num_tr + num_val].tolist()
            con_clips_test += clips[num_tr + num_val : ].tolist()
            num_prev = copy.copy(num)
        else: # Uncontaminated Samples
            uncon_clips_train += clips[:num_tr].tolist()
            uncon_clips_valid += clips[num_tr : num_tr + num_val].tolist()
            uncon_clips_test += clips[num_tr + num_val : num_tr + num_val + num_val].tolist()
        clips = []

    samples = []
    contaminated_samples = {'train': con_clips_train, 'valid': con_clips_valid, 'test': con_clips_test}
    uncontaminated_samples = {'train': uncon_clips_train, 'valid': uncon_clips_valid, 'test': uncon_clips_test}
    samples.append(contaminated_samples)
    samples.append(uncontaminated_samples)
    return samples

def make_dataset_ver2(path, class_to_idx, frames, task="bac"):
    if task == "multi" or task == "cascade":
        task = "bac"

    con_clips = []
    con_clips_train = []
    con_clips_valid = []
    con_clips_test = []

    uncon_clips = []
    uncon_clips_train = []
    uncon_clips_valid = []
    uncon_clips_test = []
    images = []
    clip_per_cam = 500 // frames

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(list(item))

                # Extract Uncontaminated Samples for each camera folder
                if len(images) != 0 and len(images) % frames == 0 and target == sorted(os.listdir(path))[1]:
                    uncon_clips.append(images)
                    images = []

                    # Allocate 50 stacked clips
                    if len(uncon_clips) % clip_per_cam == 0:
                        uncon_num = len(uncon_clips)
#                        uncon_num_tr = uncon_num // 10 * 8
                        uncon_num_tr = uncon_num // 10 * 9
                        uncon_num_val = (uncon_num - uncon_num_tr) // 2

                        uncon_clips = np.array(uncon_clips)
                        np.random.seed(0)
                        np.random.shuffle(uncon_clips)
                        uncon_clips_train += uncon_clips[:uncon_num_tr].tolist()
                        uncon_clips_valid += uncon_clips[uncon_num_tr:].tolist()
#                        uncon_clips_valid += uncon_clips[uncon_num_tr : uncon_num_tr + uncon_num_val].tolist()
#                        uncon_clips_test += uncon_clips[uncon_num_tr + uncon_num_val : uncon_num_tr + uncon_num_val * 2].tolist()
                        uncon_clips = []
                        break;

                # Extract Contaminated Samples for each camera folder
                elif len(images) != 0 and len(images) % frames == 0 and target == sorted(os.listdir(path))[0]:
                    con_clips.append(images)
                    images = []

                    # Allocate 25 stacked clips
                    if len(con_clips) % clip_per_cam == 0:
                        num = len(con_clips)
#                        num_tr = num // 10 * 8
                        num_tr = num // 10 * 9
                        num_val = (num - num_tr) // 2

                        con_clips = np.array(con_clips)
                        np.random.seed(0)
                        np.random.shuffle(con_clips)
                        con_clips_train += con_clips[:num_tr].tolist()
                        con_clips_valid += con_clips[num_tr:].tolist()
#                        con_clips_valid += con_clips[num_tr : num_tr + num_val].tolist()
#                        con_clips_test += con_clips[num_tr + num_val : num_tr + num_val * 2].tolist()
                        con_clips = []
            if len(uncon_clips_train) + len(uncon_clips_valid) + len(uncon_clips_test) == 400:
                break;

    num1 = len(con_clips_train) + len(con_clips_valid) + len(con_clips_test)
    num2 = len(uncon_clips_train) + len(uncon_clips_valid) + len(uncon_clips_test)
    print('contaminated samples: {}, uncontaminated samples: {}'.format(num1, num2))

    samples = []
    contaminated_samples = {'train': con_clips_train, 'valid': con_clips_valid, 'test': con_clips_test}
    uncontaminated_samples = {'train': uncon_clips_train, 'valid': uncon_clips_valid, 'test': uncon_clips_test}
    samples.append(contaminated_samples)
    samples.append(uncontaminated_samples)
    return samples

def make_dataset_ver2_test_only(path, class_to_idx, frames, task="bac"):
    if task == "multi" or task == "cascade":
        task = "bac"

    con_clips = []
    con_clips_train = []
    con_clips_valid = []
    con_clips_test = []

    uncon_clips = []
    uncon_clips_train = []
    uncon_clips_valid = []
    uncon_clips_test = []

    images = []
    clip_per_cam = 500 // frames

    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                if target[:3] == 'Cgl' or 'Sam':
                    item = (mat_path, '1')
                else:
                    item = (mat_path, '0')
                images.append(list(item))

                # Extract Uncontaminated Samples for each camera folder
                if len(images) != 0 and len(images) % frames == 0 and target[:3] == 'Cgl' or 'Sam':
                    uncon_clips.append(images)
                    images = []

                    # Allocate 50 stacked clips
                    if len(uncon_clips) % clip_per_cam == 0:
                        uncon_clips = np.array(uncon_clips)
                        np.random.seed(0)
                        np.random.shuffle(uncon_clips)
                        uncon_clips_test += uncon_clips[:].tolist()
                        uncon_clips = []
                        break;

                # Extract Contaminated Samples for each camera folder
                elif len(images) != 0 and len(images) % frames == 0:
                    con_clips.append(images)
                    images = []

                    # Allocate 25 stacked clips
                    if len(con_clips) % clip_per_cam == 0:
                        con_clips = np.array(con_clips)
                        np.random.seed(0)
                        np.random.shuffle(con_clips)
                        con_clips_test += con_clips[:].tolist()
                        con_clips = []

    num1 = len(con_clips_test)
    num2 = len(uncon_clips_test)
    print('contaminated samples: {}, uncontaminated samples: {}'.format(num1, num2))

    samples = []
    contaminated_samples = {'train': con_clips_train, 'valid': con_clips_valid, 'test': con_clips_test}
    uncontaminated_samples = {'train': uncon_clips_train, 'valid': uncon_clips_valid, 'test': uncon_clips_test}
    samples.append(contaminated_samples)
    samples.append(uncontaminated_samples)
    return samples

def SpeckleDataset(root, frames, test=False, task="bac"):
    # Convert Species into labels
    classes, class_to_idx = find_classes(root, task)

    # Extract Images (T X H X W)
#    samples = make_dataset(root, class_to_idx, frames, task)

    if test:
        samples_ = make_dataset_ver2_test_only(root, class_to_idx, frames, task)
    else:
        samples_ = make_dataset_ver2(root, class_to_idx, frames, task)

    tr_samples = []
    tr_samples.append(samples_[0]['train'])
    tr_samples.append(samples_[1]['train'])
    val_samples = []
    val_samples.append(samples_[0]['valid'])
    val_samples.append(samples_[1]['valid'])
    test_samples = []
    test_samples.append(samples_[0]['test'])
    test_samples.append(samples_[1]['test'])
    ensembled_samples = {'train': tr_samples, 'valid': val_samples, 'test': test_samples}

    num_tr_imgs = len(ensembled_samples['train'][0]) + len(ensembled_samples['train'][1])
    num_val_imgs = len(ensembled_samples['valid'][0]) + len(ensembled_samples['valid'][1])
    num_test_imgs = len(ensembled_samples['test'][0]) + len(ensembled_samples['test'][1])
    origin_imgs = num_tr_imgs + num_val_imgs + num_test_imgs

    print("{} origin : {}, aug: (Tr {}, Val {}, Test {})".format(
            root, origin_imgs, num_tr_imgs, num_val_imgs, num_test_imgs)
    )
    return ensembled_samples

class _Dataset(data.Dataset):
    def __init__(self, con_imgs, uncon_imgs, frames, aug_rate=0, transform=None):
        self.imgs = con_imgs + uncon_imgs
        self.frames = frames
        self.origin_imgs = len(self.imgs)
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
        path = imgs[:, 0].tolist()

        # List of the same numbers -> one representative number
        target = imgs[:, 1][0]

        img = []
        for i in range(len(path)):
            img.append(imread(path[i]))
        img = np.array(img)

        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img)
        else:
            for t in TEST_AUGS_2D:
                img = t(img)

        return img, target, path

    def __len__(self):
        return len(self.imgs)

class _Dataset_contrastive(data.Dataset):
    def __init__(self, con_imgs, uncon_imgs, frames, aug_rate=0, transform=None):
        self.con_imgs = con_imgs
        self.uncon_imgs = uncon_imgs

        self.frames = frames
        self.origin_imgs = len(con_imgs) + len(uncon_imgs)
        self.augs = [] if transform is None else transform

        if aug_rate != 0:
            self.con_imgs += random.sample(self.con_imgs, int(len(self.con_imgs) * aug_rate))
            self.uncon_imgs += random.sample(self.uncon_imgs, int(len(self.uncon_imgs) * aug_rate))

        import itertools
        self.dissimilar = []
        # combinations X two clips X fps (2891520 X 2 X 20)
        self.similar = list(itertools.combinations(self.con_imgs, r=2)) + list(itertools.combinations(self.uncon_imgs, r=2))
        self.dissimilar = [(con_img, uncon_img) for uncon_img in uncon_imgs for con_img in con_imgs]

        self.num_similar = len(self.similar)
        self.num_dissimilar = len(self.dissimilar)
        self.imgs = self.dissimilar + self.similar
        self.imgs = np.array(self.imgs)

    def __getitem__(self, index):
        imgs = self.imgs[index]
        num_imgs = imgs.shape[1]
        if num_imgs < self.frames:
            print('Error: T is too large | Num img:{} < T:{}'.format(num_imgs, self.frames))
            sys.exit()

        # Extract Sequential T Images
        path1 = imgs[0, :, 0].tolist()
        path2 = imgs[1, :, 0].tolist()
        cls_target1 = imgs[0, :, 1][0]
        cls_target2 = imgs[1, :, 1][0]

        if np.all(imgs[:, :, 1] == imgs[0, 0, 1]): # Similar
            target = 1
        else: # Dissimilar
            target = 0

        img1 = imread(path1[0])
        img2 = imread(path2[0])
        for i in range(1, len(path1)):
            img1 = np.append(img1, imread(path1[i]))
            img2 = np.append(img2, imread(path2[i]))
        img1 = img1.reshape(-1, 256, 256)
        img2 = img2.reshape(-1, 256, 256)

        if index > self.origin_imgs:
            for t in self.augs:
                img1 = t(img1)
                img2 = t(img2)
        else:
            for t in TEST_AUGS_2D:
                img1 = t(img1)
                img2 = t(img2)

        img = (img1, img2)
        path = (path1, path2)
        return img, target, cls_target1, cls_target2, path

    def __len__(self):
        return len(self.imgs)

class _Dataset_contrastive_on_the_fly(data.Dataset):
    def __init__(self, con_imgs, uncon_imgs, frames, aug_rate=0, transform=None):
        self.con_imgs = con_imgs
        self.uncon_imgs = uncon_imgs

        self.frames = frames
        self.origin_imgs = len(con_imgs) + len(uncon_imgs)
        self.augs = [] if transform is None else transform

#        if aug_rate != 0:
#            self.con_imgs += random.sample(self.con_imgs, int(len(self.con_imgs) * aug_rate))
#            self.uncon_imgs += random.sample(self.uncon_imgs, int(len(self.uncon_imgs) * aug_rate))

        self.imgs_ = self.con_imgs + self.uncon_imgs
        self.num_contam = len(self.con_imgs)
        self.num_uncontam = len(self.uncon_imgs)
        self.num_imgs = len(self.imgs_)

        self.imgs = self.con_imgs
        self.imgs = np.array(self.imgs)
        self.imgs_ = np.array(self.imgs_)

    def __getitem__(self, index):
        con_imgs = self.imgs_[index]
        random_imgs = self.imgs_[np.random.choice(self.num_imgs)]
        num_imgs = self.num_contam + self.num_uncontam
        if num_imgs < self.frames:
            print('Error: T is too large | Num img:{} < T:{}'.format(num_imgs, self.frames))
            sys.exit()

        # Extract Sequential T Images
        con_path = con_imgs[:, 0].tolist()
        random_path = random_imgs[:, 0].tolist()
        con_cls_target = con_imgs[:, 1][0]
        random_cls_target = random_imgs[:, 1][0]

        if con_cls_target == random_cls_target:
            target = 1
        else:
            target = 0

        img1 = imread(con_path[0])
        img2 = imread(random_path[0])
        for i in range(1, len(con_path)):
            img1 = np.append(img1, imread(con_path[i]))
            img2 = np.append(img2, imread(random_path[i]))
        img1 = img1.reshape(-1, 256, 256)
        img2 = img2.reshape(-1, 256, 256)

        if index > self.origin_imgs:
            for t in self.augs:
                img1 = t(img1)
                img2 = t(img2)
        else:
            for t in TEST_AUGS_2D:
                img1 = t(img1)
                img2 = t(img2)

        img = (img1, img2)
        path = (con_path, random_path)
        return img, target, con_cls_target, random_cls_target, path

    def __len__(self):
        return len(self.imgs_)

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

def SpeckleLoader(image_path, frames, batch_size, test_batch_size, task="bac", sampler=False,
                 tr_transform=None, val_transform=None, test_transform=None, tr_aug_rate=0,
                 val_aug_rate=0, test_aug_rate=0, num_workers=1, shuffle=False, drop_last=False,
                 test=False, train_backbone=True):

    samples = SpeckleDataset(image_path, frames, test=test, task=task)
    if train_backbone:
        tr_dataset = _Dataset(samples['train'][0], samples['train'][1], frames, aug_rate=tr_aug_rate, transform=tr_transform)
        val_dataset = _Dataset(samples['valid'][0], samples['valid'][1], frames, aug_rate=val_aug_rate, transform=val_transform)
        test_dataset = _Dataset(samples['test'][0], samples['test'][1], frames, aug_rate=test_aug_rate, transform=test_transform)
#        tr_dataset = _Dataset_contrastive_on_the_fly(samples['train'][0], samples['train'][1], frames, aug_rate=tr_aug_rate, transform=tr_transform)
#        val_dataset = _Dataset_contrastive_on_the_fly(samples['valid'][0], samples['valid'][1], frames, aug_rate=val_aug_rate, transform=val_transform)
#        test_dataset = _Dataset_contrastive_on_the_fly(samples['test'][0], samples['test'][1], frames, aug_rate=test_aug_rate, transform=test_transform)
#        tr_dataset = _Dataset_contrastive(samples['train'][0], samples['train'][1], frames, aug_rate=tr_aug_rate, transform=tr_transform)
#        val_dataset = _Dataset_contrastive(samples['valid'][0], samples['valid'][1], frames, aug_rate=val_aug_rate, transform=val_transform)
#        test_dataset = _Dataset_contrastive(samples['test'][0], samples['test'][1], frames, aug_rate=test_aug_rate, transform=test_transform)

    else:
        tr_dataset = _Dataset(samples['train'][0], samples['train'][1], frames, aug_rate=tr_aug_rate, transform=tr_transform)
        val_dataset = _Dataset(samples['valid'][0], samples['valid'][1], frames, aug_rate=val_aug_rate, transform=test_transform)
        test_dataset = _Dataset(samples['test'][0], samples['test'][1], frames, aug_rate=test_aug_rate, transform=test_transform)

    if sampler:
        print("Sampler : ", image_path[-5:])
        tr_sampler = _make_weighted_sampler(tr_dataset.imgs)
        val_sampler = _make_weighted_sampler(val_dataset.imgs)
        tr_dataset = data.DataLoader(tr_dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        val_dataset = data.DataLoader(val_dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        return tr_dataset, val_dataset, test_dataset

    else:
        if test:
            test_dataset = data.DataLoader(test_dataset, test_batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
            return None, None, test_dataset

        else:
            tr_dataset = data.DataLoader(tr_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
            val_dataset = data.DataLoader(val_dataset, test_batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
            test_dataset = None
            return tr_dataset, val_dataset, test_dataset
