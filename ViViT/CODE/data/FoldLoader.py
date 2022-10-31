from datas.BacLoader import BacDataset
from datas.BacLoader import find_classes
from datas.BacLoader import make_dataset
from datas.BacLoader import  _make_weighted_sampler

from datas.preprocess3d import TRAIN_AUGS_3D
from datas.preprocess3d import TEST_AUGS_3D

import random

from torch.utils import data



class FoldSet(BacDataset):
    def __init__(self, cls, cls_to_idx, imgs, transform=None, aug_rate=0, task="bac"):
        self.imgs = imgs
        self.origin_imgs = len(self.imgs)

        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))

        self.augs = [] if transform is None else transform
        self.classes = cls
        self.class_to_idx = cls_to_idx
        self.task = task

def FoldGenerator(path, batch_size, cpus, aug_rate, task="bac"):
    classes, class_to_idx = find_classes(path, task=task)
    imgs = make_dataset(path, class_to_idx, task=task)

    def _generator():
        for i in range(3):
            random.shuffle(imgs)
            valid_imgs = imgs[:40]
            train_imgs = imgs[40:]

            trainset = FoldSet(classes, class_to_idx, train_imgs[:],
                            transform=TRAIN_AUGS_3D, aug_rate=aug_rate, task=task)
            validset = FoldSet(classes, class_to_idx, valid_imgs[:],
                            transform=TEST_AUGS_3D, aug_rate=0, task=task)

            sampler = _make_weighted_sampler(trainset.imgs)

            trainloader = data.DataLoader(trainset, batch_size,
                                        sampler=sampler, num_workers=cpus, drop_last=True)
            validloader = data.DataLoader(validset, batch_size,
                                        num_workers=cpus, drop_last=False)                                                                            
            yield trainloader, validloader
    return _generator()