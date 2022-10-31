import os 
import random

import numpy as np 
import scipy.io as io

import torch
from torch.utils import data

from datas.preprocess3d import TEST_AUGS_3D
from datas.preprocess3d import mat2npy


class_meta_data = {                                                                        # Gram - 3   Morphology       Motility     Metabolism
    "Acinetobacter_baumannii":["Acinetobacter_baumannii", "rod", "Acinetobacter",           "negative", "coccobacillus", "nonmotile", "aerobic"],   # Acinetobacter_baumannii
    "Bacillus_subtilis":["Bacillus_subtilis", "rod", "Bacillus",                            "positive", "bacillus",      "motile",    "aerobic"],   # Bacillus_subtilis
    "Enterobacter_cloacae":["Enterobacter_cloacae", "rod", "Enterobacter",                  "negative", "bacillus",      "motile",    "anaerobic"], # Enterobacter_cloacae
    "Enterococcus_faecalis":["Enterococcus_faecalis", "round", "Enterococcus",              "positive", "coccus",        "nonmotile", "anaerobic"], # Enterococcus_faecalis
    "Escherichia_coli":["Escherichia_coli", "rod", "Escherichia",                           "negative", "bacillus",      "motile",    "anaerobic"], # Escherichia_coli
    "Klebsiella_pneumoniae":["Klebsiella_pneumoniae", "rod", "Klebsiella",                  "negative", "bacillus",      "nonmotile", "anaerobic"], # Klebsiella_pneumoniae
    "Listeria_monocytogenes":["Listeria_monocytogenes", "rod", "Listeria",                  "positive", "bacillus",      "motile",    "anaerobic"], # Listeria_monocytogenes
    "Micrococcus_luteus":["Micrococcus_luteus", "round", "Micrococcus",                     "positive", "coccus",        "nonmotile", "aerobic"],   # Micrococcus_luteus
    "Pseudomonas_aeruginosa":["Pseudomonas_aeruginosa", "rod", "Pseudomonas",               "negative", "bacillus",      "motile",    "anaerobic"], # Pseudomonas_aeruginosa
    "Proteus_mirabilis":["Proteus_mirabilis", "rod", "Proteus",                             "negative", "bacillus",      "motile",    "anaerobic"], # Proteus_mirabilis
    "Smarcescens":["Smarcescens", "rod", "Serratia",                                        "negative", "bacillus",      "motile",    "anaerobic"], # Smarcescens
    "Staphylococcus_aureus":["Staphylococcus_aureus", "round", "Staphylococcus",            "positive", "coccus",        "nonmotile", "anaerobic"], # Staphylococcus_aureus
    "Staphylococcus_epidermidis":["Staphylococcus_epidermidis", "round", "Staphylococcus",  "positive", "coccus",        "nonmotile", "anaerobic"], # Staphylococcus_epidermidis
    "Streptococcus_anginosus":["Streptococcus_anginosus", "round", "Streptococcus",         "positive", "coccus",        "nonmotile", "anaerobic"], # Streptococcus_anginosus
    "Streptococcus_pyogenes":["Streptococcus_pyogenes", "round", "Streptococcus",           "positive", "coccus",        "nonmotile", "anaerobic"], # Streptococcus_pyogenes
    "Streptococcus_agalactiae":["Streptococcus_agalactiae", "round", "Streptococcus",       "positive", "coccus",        "nonmotile", "anaerobic"], # Streptococcus_agalactiae
    "Streptococcus_pneumoniae":["Streptococcus_pneumoniae", "round", "Streptococcus",       "positive", "coccus",        "nonmotile", "anaerobic"], # Streptococcus_pneumoniae
    "Smaltophilia":["Smaltophilia", "rod", "Stenotrophomonas",                              "negative", "bacillus",      "motile",    "aerobic"],   # Smaltophilia
    "Haemophilus_influenzae": ["Haemophilus_influenzae", "rod", "Haemophilus",              "negative", "coccobacillus", "nonmotile", "anaerobic"]  # Haemophilus_influenzae
}

task_meta = {"bac":0, "shape":1, "gen":2, "gram":3, "morphology":4, "motility":5, "metabolism":6}


def find_classes(path, task="bac"):
    if task == "multi":
        return find_classes_multi(path)
    if task == "cascade":
        task = "bac"

    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    task = task_meta[task]
    classes = sorted(list(set(class_meta_data[c][task] for c in classes)))

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def find_classes_multi(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    classes_np = np.array([class_meta_data[c] for c in classes])
    classes_inverse = [list(set(classes_np[:, i])) for i in range(5)]

    class_to_idx = [{c[i]:i for i in range(len(c))} for c in classes_inverse]
    class_to_idx = {k:v for d in class_to_idx for k, v in d.items()}

    return classes, {k:[class_to_idx[i] for i in v] for k, v in class_meta_data.items()}


def make_dataset(path, class_to_idx, task="bac"):
    if task == "multi" or task == "cascade":
        task = "bac"
    images = []
    path = os.path.expanduser(path)
    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        target = class_meta_data[target][task_meta[task]]
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                item = (mat_path, class_to_idx[target])
                images.append(item)
    return images


class BacDataset(data.Dataset):
    def __init__(self, dataset_path, transform=None, aug_rate=0,
                 task="bac"):
        classes, class_to_idx = find_classes(dataset_path, task=task)
        print(class_to_idx)
        self.imgs = make_dataset(dataset_path, class_to_idx, task=task)

        self.origin_imgs = len(self.imgs)
        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + dataset_path ))

        
        if aug_rate != 0:
            self.imgs += random.sample(self.imgs, int(len(self.imgs) * aug_rate))
        print("Dataset Dir : ", dataset_path, "origin : ", self.origin_imgs, ", aug : ", len(self.imgs))

        self.augs = [] if transform is None else transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
        self.task = task

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        mat = io.loadmat(path)
        img, ri = mat2npy(mat)

        if index > self.origin_imgs:
            for t in self.augs:
                img = t(img, ri=ri)
        else:
            for t in TEST_AUGS_3D:
                img = t(img, ri=ri)

        if self.task == "cascade":
            crop_s = max(img.shape) // 4
            crop_z = img.shape[-1] // 4
            crop_x = img[:, crop_s:-crop_s, crop_s:-crop_s, crop_z:-crop_z]
            return img, crop_x, target, path
        else:
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
        weight_per_class[i] = N/float(count[i])    

    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    return sampler       


def bacLoader(image_path, batch_size, task="bac", sampler=False,
              transform=None, aug_rate=0,
              num_workers=1, shuffle=False, drop_last=False):
    dataset = BacDataset(image_path, task=task, transform=transform, aug_rate=aug_rate)
    if sampler:
        print("Sampler : ", image_path, len(list(dataset.class_to_idx.keys())))
        sampler = _make_weighted_sampler(dataset.imgs, list(dataset.class_to_idx.keys()))
        return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


if __name__ == "__main__":
    import torch
    data_path = "/data2/UTI/200930_bac/Bacteria/"
    data_path = "/data2/UTI/200930_bac/181017_Bacteria/"

    import preprocess3d as preprocess
    pp = preprocess.get_preprocess("test")
    loader = bacLoader(data_path + "valid", 4, task="bac",
                             transform=pp, aug_rate=0,
                             num_workers=4, shuffle=False, drop_last=False)
    p1 = []
    for input, target, path in loader:
        p1 += list(path)

    p2 = []
    for input, target, path in loader:
        p2 += list(path)

    p3 = []
    for input, target, path in loader:
        p3 += list(path)
    
    cc = len("/data2/UTI/200930_bac/201017_Bacteria/valid/") + 1
    for z1, z2, z3 in zip(p1, p2, p3):
        if z1 != z2 or z1 != z3 or z2 != z3:
            print(z1[cc:], z2[cc:] ,z3[cc:])

