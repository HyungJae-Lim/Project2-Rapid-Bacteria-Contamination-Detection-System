import os
import copy
import random

import numpy as np
import scipy.io as io
#from scipy.misc import imread
from imageio import imread
from PIL import Image
import torch

path = os.path.expanduser('/data_cuda_ssd/minsu/TEST_hyungjae_CFU/test')
if __name__ == '__main__':
    clips = []
    images = []
    f_names = []

    for target in sorted(os.listdir(path)):
        d = os.path.join(path, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): # One cycle = One species
            for fname in sorted(fnames):
                mat_path = os.path.join(root, fname)
                images.append(torch.Tensor(imread(mat_path)))
                f_names.append(fname)

    images = torch.stack(images, dim=0).reshape(5, -1, 500, 256, 256)
    intensities = torch.mean(images, axis=2, keepdim=True)
    temp = (images - intensities).view(-1, 256, 256)

    print(temp[0])

#    for i in range(temp.shape[0]):
#        img = Image.fromarray(temp[i].numpy())
#        img.save('./tf_test/{}'.format(f_names[i]))
