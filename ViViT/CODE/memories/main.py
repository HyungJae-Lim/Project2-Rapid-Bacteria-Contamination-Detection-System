from multiprocessing import Process
import os
import json
import argparse
import utils

import torch
import torch.nn as nn

# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False

# example for mnist
#from data.SpeckleLoader import *
#from data.FermentationLoader import *
#from data.FermentationLoader_hyungjae_CFU import *
from data.preprocess import TRAIN_AUGS_2D, TEST_AUGS_2D

from Logger import Logger

from models.vivit_2 import ViViT
from models.tCNN_3 import tCNN_3
from models.tCNN_3_regression import tCNN_3_regression
from models.tCNN_modified import tCNN_modified
from models.tCNN_auxcl import tCNN_auxcl

from runners.BaseRunner import BaseRunner
#from runners.SpeckleRunner import SpeckleRunner
#from runners.SpeckleRunner_hyungjae import SpeckleRunner
from runners.BlindTest import SpeckleRunner as BlindTest
from runners.SP2_removal import SpeckleRunner as removalTest
from runners.SpeckleRunner_auxcl import SpeckleRunner_auxcl

def arg_parse():
    # projects description
    desc = "Bacteria Speckle Analyzer"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7",
                        help="Select GPU Numbering | 0,1,2,3,4,5,6,7 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")

    parser.add_argument('--aug', type=float, default=0.5, help='The number of Augmentation Rate')

    parser.add_argument('--norm', type=bool, default=False,
                        choices=[True, False])
    parser.add_argument('--act', type=str, default='lrelu',
                        choices=["relu", "lrelu", "prelu"])

    parser.add_argument('--task', type=str, default='bac',
                        choices=["bac"])

    parser.add_argument('--model', type=str, default='tCNN_3',
                        choices=["tCNN_3", "tCNN_3_regression", "tCNN_modified", "tCNN_auxcl", "vivit_2"],
                        help='The type of Models | tCNN_3 | tCNN_3_regression | tCNN_modified |')

    parser.add_argument('--data_dir', type=str, default='',
                        help='Directory name to load the data')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--batch_size_test', type=int, default=1, help='The size of batch')
    parser.add_argument('--test', action="store_true", help='Only Test')
    parser.add_argument('--sampler', action="store_true", default=False, help='Weighted Sampler work')

    parser.add_argument('--optim', type=str, default='adam', choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=1.0)
    # Adam Optimizer
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))
    # SGD Optimizer
    parser.add_argument('--momentum', type=float, default=0.9)#0.9)
    parser.add_argument('--decay', type=float, default=0.0)#1e-4)
    parser.add_argument('--load_fname', type=str, default=None)
    parser.add_argument('--mode', type=int, default=0, help='0/1: Species/3labels')

    # Added by Minsu
    parser.add_argument('--image_T', type=int, default=2, help='T Frames per batch')
    parser.add_argument('--clip_frames', type=int, default=300, help='Clip Frames per batch')
    parser.add_argument('--shuffle', type=bool, default=True, choices=[True, False])
    parser.add_argument('--resume', type=bool, default=False, choices=[True, False])
    parser.add_argument('--blind_test', type=bool, default=False, choices=[True, False])
    parser.add_argument('--reverse_axis', type=bool, default=False, choices=[True, False])
    parser.add_argument('--flow', type=bool, default=False, choices=[True, False])
    parser.add_argument('--description', type=str, default='', help='Optional Ongoing Thread information')
    parser.add_argument('--patch_size', type=int, default=64, help='model patch size')

    return parser.parse_args()

def get_params_and_GPU_memory(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn

    mem = torch.cuda.memory_allocated(0) // 1024 ** 2
    return pp, mem

if __name__ == "__main__":
    arg = arg_parse()
    arg.task='bac'
    if arg.mode == 0 and arg.flow:
        labels = 2
        in_channel=2
        from data.FermentationLoader_hyungjae import *
        from runners.SpeckleRunner import SpeckleRunner
    elif arg.mode == 0 and not arg.flow:
        labels = 2
        in_channel=1
        from data.FermentationLoader import *
#        from data.FermentationLoader_hyungjae import *
        from runners.SpeckleRunner import SpeckleRunner
    elif arg.mode == 1 and arg.flow:
        labels = 5
        in_channel=2
        from data.FermentationLoader_hyungjae_CFU import *
        from runners.SpeckleRunner_hyungjae import SpeckleRunner
    elif arg.mode == 1 and not arg.flow:
        labels = 5
        in_channel=1
        from data.FermentationLoader_hyungjae_CFU import *
        from runners.SpeckleRunner_hyungjae import SpeckleRunner
    elif arg.mode == 2: # Histogram classification
        labels = 2
        in_channel=1
        from data.histogram_loader import *
        from runners.HistogramSVMRunner import SpeckleRunner

    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    logger = Logger(arg.save_dir)
    logger.will_write(str(arg) + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    data_path = arg.data_dir
    print("Data Path : ", data_path)

    if arg.blind_test:
        '''
        If test goes on with blind dataset,
        We can simply know the files' prediction results with no shuffling

        If shuffling is applied, clips are tangled up in complicated orders
        And comparing the predictions and labels may become more difficult.

        So we didn't apply the shuffling work when the blind test was done.
        '''
        arg.shuffle = False

    loaders = SpeckleLoader(
            data_path, arg.clip_frames, arg.batch_size, arg.batch_size_test, task=arg.task, sampler=arg.sampler,
            tr_transform=TRAIN_AUGS_2D, val_transform=TEST_AUGS_2D, test_transform=TEST_AUGS_2D,
            tr_aug_rate=arg.aug, val_aug_rate=0, test_aug_rate=0, num_workers=arg.cpus, shuffle=arg.shuffle, drop_last=True,
            test=arg.test, flow=arg.flow
    )

    # Changed Backbone network; Model of ViViT Paper
    # Initialized Parameters: (image_size, patch_size, num_classes, arg.image_T)
    net = ViViT(arg.clip_frames, 256, arg.patch_size, labels, arg.image_T, in_channels=in_channel, reverse_axis=arg.reverse_axis).to(torch_device)
    net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()

    optim = {
        "adam": torch.optim.Adam(net.parameters(), lr=arg.lr, betas=arg.beta, weight_decay=arg.decay),
        "sgd": torch.optim.SGD(net.parameters(),
                               lr=arg.lr, momentum=arg.momentum,
                               weight_decay=arg.decay, nesterov=False)
    }["adam"]

    model = SpeckleRunner(arg, net, optim, torch_device, loss, logger)
    if arg.test is False:
        train_loader, val_loader, test_loader = loaders
        model.train(train_loader, val_loader, test_loader)
#        model._get_acc(test_loader, test=True)

    else:
        _, _, test_loader = loaders
        if arg.blind_test:
            model = BlindTest(arg, net, optim, torch_device, loss, logger)
            model.no_label_test_get_acc(test_loader)

        else:
            model._get_acc(test_loader, test=True)
#            model = removalTest(arg, net, optim, torch_device, loss, logger)
#            model.no_label_test_get_acc(test_loader)

