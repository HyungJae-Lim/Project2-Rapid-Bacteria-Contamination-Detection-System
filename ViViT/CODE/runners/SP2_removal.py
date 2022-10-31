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
            "gram_negative": 0, "gram_positive": 0, "non-infection": 0,
        }

        preds = []
        labels = []
        results = []
        results_ = []
        results.append(species_dict.copy())
        results_.append(species_dict.copy())

        prev = 0
        correct = 0

        i = 0
        j = 0
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
                label = "gram_negative"
                results[i]["gram_negative"] += 1

            elif prediction == 1:
                label = "gram_positive"
                results[i]["gram_positive"] += 1

            elif prediction == 2:
                label = "non-infection"
                results[i]["non-infection"] += 1

            if (eval_b_id + 1) % 150 == 0:
                print('Number of Vote: {} | Tested Number of Clips: {}\nResults: {}'.format(i+1, eval_b_id + 1 - prev, results[i]))
                predicted_label = max(results[i], key=results[i].get)

                if i+1 % 140 == 0:
                    print('Midterm Label Report: {}'.format(results_[j]))
                    results_.append(species_dict.copy())
                    j += 1

                if predicted_label == "gram_negative":
                    results_[j]["gram_negative"] += 1
                elif predicted_label == "gram_positive":
                    results_[j]["gram_positive"] += 1
                elif predicted_label == "non-infection":
                    results_[j]["non-infection"] += 1

                print('path: {}'.format(path))
                print('labels: {}'.format(results[0]))
                print('voted labels: {}'.format(results_[0]))
                print('=' * 145, end="\n\n")
                results.append(species_dict.copy())
                prev = eval_b_id + 1
                i += 1


#            if (eval_b_id + 1) == 9000:
#                print('Tested Folder: {} | Tested Number of Clips: {}\nResults: {}'.format(i+1, eval_b_id + 1 - prev, results[i]))
#                print('path: {}'.format(path))
#                print('=' * 145, end="\n\n")
#                results.append(species_dict.copy())
#                prev = eval_b_id + 1
#                i += 1

#            if (eval_b_id + 1) == 24000:
#                print('Tested Folder: {} | Tested Number of Clips: {}\nResults: {}'.format(i+1, eval_b_id + 1 - prev, results[i]))
#                print('=' * 145, end="\n\n")

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
