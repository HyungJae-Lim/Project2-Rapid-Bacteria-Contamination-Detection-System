import os
import time
import numpy as np
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from glob import glob
from utils import get_confusion
from .BaseRunner import BaseRunner
from collections import defaultdict

from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

def get_params_and_GPU_memory(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn

    mem = torch.cuda.max_memory_allocated(0)
    return pp, mem

class SpeckleRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)
        self.net = net
        self.loss = loss
        self.optim = optim
        self.image_T = arg.image_T
        self.description = arg.description

        if arg.resume:
            self.load(arg.load_fname)

#        if arg.optim == "sgd":
#            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 100)

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.save_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.save_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File" % (self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" % (
            ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def no_label_test_get_acc(self, loader, confusion=False):
        self.load()
        species_dict = {
            "E.coli": 0, "E.fascal": 0, "K.pneum": 0, "P.aerug": 0,
            "P.mirab": 0, "S.agalac": 0, "S.aureus": 0, "S.epider": 0, "S.sapro": 0
        }

        preds = []
        labels = []
        results = []
        results.append(species_dict.copy())

        prev = 0
        correct = 0

        i = 0
        loader_iter = iter(loader)
        pbar = tqdm(range(len(loader.dataset)//self.arg.batch_size_test), smoothing=0.9)
        for eval_b_id in pbar:
            input_, path = next(loader_iter)
            input_ = input_.to(self.torch_device)

            output_ = self.net(input_)
            prediction = torch.max(output_, dim=-1)[1]

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

            if prediction == 0:
                label = "E. coli"
                results[i]["E.coli"] += 1

            elif prediction == 1:
                label = "E. fascalis"
                results[i]["E.fascal"] += 1

            elif prediction == 2:
                label = "K. pneumoniae"
                results[i]["K.pneum"] += 1

            elif prediction == 3:
                label = "P. aeruginosa"
                results[i]["P.aerug"] += 1

            elif prediction == 4:
                label = "P. mirabilis"
                results[i]["P.mirab"] += 1

            elif prediction == 5:
                label = "S. agalactiae"
                results[i]["S.agalac"] += 1

            elif prediction == 6:
                label = "S. aureus"
                results[i]["S.aureus"] += 1

            elif prediction == 7:
                label = "S. epidermidis"
                results[i]["S.epider"] += 1

            elif prediction == 8:
                label = "S. saprohyticus"
                results[i]["S.sapro"] += 1

            if eval_b_id != 0 and (eval_b_id + 1) % (300//self.image_T) == 0:
                print('Tested Folder: {} | Tested Number of Clips: {}\nResults: {}'.format(i+1, eval_b_id + 1 - prev, results[i]))
                print('=' * 145, end="\n\n")
                results.append(species_dict.copy())

                prev = eval_b_id + 1
                i += 1

            pbar.set_description_str(desc="[TEST ]", refresh=True)

        if confusion:
            confusion = get_confusion(preds, labels)

#        return correct / len(loader.dataset), confusion
        return None


    def valid(self, epoch, val_loader, test_loader, train_acc):
        self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc(val_loader)
            test_acc, *_ = self._get_acc(test_loader, test=True)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)
