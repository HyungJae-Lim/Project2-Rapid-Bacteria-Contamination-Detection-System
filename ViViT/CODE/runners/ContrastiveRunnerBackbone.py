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

def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn

    return pp

def get_GPU_memory(model):
    return torch.cuda.max_memory_allocated(0)

class SpeckleRunnerBackbone(BaseRunner):
    def __init__(self, arg, backbone, optim, torch_device, loss, logger, margin=1.00):
        super().__init__(arg, torch_device, logger)
        self.backbone = backbone
        self.loss = loss
        self.optim = optim
        self.margin = margin
        self.description = arg.description

        self.best_metric = 1
        self.start_time = time.time()

        if arg.resume:
            self.load(arg.load_fname)
#        if arg.optim == "sgd":
#            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 100)

    def save(self, epoch, filename):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch
            filename : model save file name
        """
        if epoch < 0:
            return

        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.backbone.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("[%dE] Saved Best Model, Mean loss=%.05f" % (epoch, self.best_metric))

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

            self.backbone.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f" % (
            ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader, test_loader):
        avg_loss = []
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            if epoch != 0 and epoch % 50 == 0 and epoch < 350:
                self.arg.lr /= 2
                for p in self.optim.param_groups:
                    p['lr'] = self.arg.lr
                self.logger.will_write("lr decay : %f" % (self.arg.lr))

            print('='*150)
            self.backbone.train()
            train_loader_iter = iter(train_loader)
            if len(self.description) > 0:
                print('[Ongoing Thread] \"{}\" Task'.format(self.description))

            print('[Scheduler] Learning Rate: {}'.format(self.optim.param_groups[0]['lr']))
            pbar = tqdm(range(len(train_loader.dataset)//self.arg.batch_size), smoothing=0.9)
            for train_b_id in pbar:
                input, target, _, _, path = next(train_loader_iter)
                self.optim.zero_grad()
                target = torch.Tensor(list(map(int, target))).type(torch.LongTensor)
                target = Variable(target).to(self.torch_device)
                input1 = Variable(torch.Tensor(input[0])).to(self.torch_device)
                input2 = Variable(torch.Tensor(input[1])).to(self.torch_device)
                distance, _, _ = self.backbone(input1, x2=input2)

                loss = torch.zeros_like(distance).to(self.torch_device)
                # target==0: dissimilar, target==1: similar
                loss[target==0] = torch.clamp((self.margin - distance[target==0]) ** 2, min=0).clone()
                loss[target==1] = (distance[target==1] ** 2).clone()
                loss = torch.mean(loss)

                loss.backward()
                self.optim.step()
                avg_loss.append(loss.item())
                pbar.set_description_str(
                    desc="[Train] Epoch {}/{}, loss:{:.6f}".format(
                        epoch, self.epoch-self.start_epoch-1, loss.item()
                    ), refresh=True
                )

#                if loss.item() < 0.00001:
#                    self.save(epoch, "epoch[%05d]_distance[%.4f]" % (epoch, loss.item()))
#                    import sys; sys.exit()

            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, sum(avg_loss) / len(avg_loss))
            else:
                self.save(epoch)

    def valid(self, epoch, val_loader, test_loader, avg_train_loss):
        self.backbone.eval()
        with torch.no_grad():
            correct = 0
            avg_loss = []

            loader_iter = iter(val_loader)
            pbar = tqdm(range(len(val_loader.dataset)//self.arg.batch_size_test), smoothing=0.9)
            for eval_b_id in pbar:
                input, target, _, _, path = next(loader_iter)
                target = torch.Tensor(list(map(int, target))).type(torch.LongTensor)
                target = target.to(self.torch_device)

                input1 = Variable(torch.Tensor(input[0])).to(self.torch_device)
                input2 = Variable(torch.Tensor(input[1])).to(self.torch_device)
                distance, _, _ = self.backbone(input1, x2=input2)

                loss = torch.zeros_like(distance).to(self.torch_device)
                loss[target==0] = torch.clamp((self.margin - distance[target==0]) ** 2, min=0).clone()
                loss[target==1] = (distance[target==1] ** 2).clone()
                loss = torch.mean(loss)
                pbar.set_description_str(desc="[Valid] loss: {:.6f}".format(loss), refresh=True)
                avg_loss.append(loss.item())

            val_avg_loss = sum(avg_loss) / len(avg_loss)
            test_avg_loss = self.test(epoch, test_loader)
            print("Train loss:{:6f} | Validation loss:{:.6f} | Test loss:{:.6f}".format(avg_train_loss, val_avg_loss, test_avg_loss))
            self.logger.log_write("valid", epoch=epoch, valid_avg_loss=val_avg_loss, test_avg_loss=test_avg_loss)
            if val_avg_loss < self.best_metric:
                self.best_metric = val_avg_loss
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, val_avg_loss, test_avg_loss))

            if val_avg_loss < 0.0001:
                import sys; sys.exit()

    def test(self, epoch, test_loader):
        self.backbone.eval()
        with torch.no_grad():
            correct = 0
            avg_loss = []

            loader_iter = iter(test_loader)
            pbar = tqdm(range(len(test_loader.dataset)//self.arg.batch_size_test), smoothing=0.9)
            for eval_b_id in pbar:
                input, target, _, _, path = next(loader_iter)
                target = torch.Tensor(list(map(int, target))).type(torch.LongTensor)
                target = target.to(self.torch_device)

                input1 = Variable(torch.Tensor(input[0])).to(self.torch_device)
                input2 = Variable(torch.Tensor(input[1])).to(self.torch_device)
                distance, _, _ = self.backbone(input1, x2=input2)

                loss = torch.zeros_like(distance).to(self.torch_device)
                loss[target==0] = torch.clamp((self.margin - distance[target==0]) ** 2, min=0).clone()
                loss[target==1] = (distance[target==1] ** 2).clone()
                loss = torch.mean(loss)
                pbar.set_description_str(desc="[Test ] average loss: {}".format(loss), refresh=True)
                avg_loss.append(loss.item())

            avg_loss = sum(avg_loss) / len(avg_loss)
            return avg_loss

