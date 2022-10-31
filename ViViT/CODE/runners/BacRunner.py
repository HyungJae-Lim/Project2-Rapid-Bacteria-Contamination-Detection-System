import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import time
from utils import get_confusion

class BacRunner(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)

        self.net = net
        self.loss = loss
        self.optim = optim

        self.best_metric = -1
        self.start_time = time.time()

        self.load() 
        if arg.optim == "sgd":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 100)

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
        if epoch < 50:
            return

        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network" : self.net.state_dict(),
                    "optimizer" : self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar"%(filename))
        print("Model saved %d epoch"%(epoch))

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
            print("Load %s to %s File"%(self.save_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            self.net.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f"%(ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            """
            if epoch == 299:
                self.optim = torch.optim.SGD(self.net.parameters(), 
                                             lr=self.arg.lr, momentum=self.arg.momentum,
                                             weight_decay=self.arg.decay, nesterov=True)
            """
            
            if epoch % 50 == 0 and epoch < 350:
                self.arg.lr /= 2
                for p in self.optim.param_groups:
                    p['lr'] = self.arg.lr
                self.logger.will_write("lr decay : %f"%(self.arg.lr))

            # self.scheduler.step()
            self.net.train()
            for i, (input_, target_, path) in enumerate(train_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_, *_ = self.net(input_)
                loss = self.loss(output_, target_)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if (i % 50) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=loss.item())

            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader)
            else:
                self.save(epoch)

    def _get_acc(self, loader, confusion=False):
        correct = 0
        preds, labels = [], []
        for input_, target_, _ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            correct += torch.sum(target_ == idx).float().cpu().item()

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

        if confusion:
            confusion = get_confusion(preds, labels)
            
        return correct / len(loader.dataset), confusion


    def valid(self, epoch, val_loader, test_loader):
        self.net.eval()
        with torch.no_grad():
            if self.arg.dim == "25d" or self.arg.dim == "2d":
                acc, *_ = self._get_acc_25d(val_loader)
                test_acc, *_ = self._get_acc_25d(test_loader)
            else:
                acc, *_ = self._get_acc(val_loader)
                test_acc, *_ = self._get_acc(test_loader)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)

            if acc > self.best_metric:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]"%(epoch, acc, test_acc))


    def test(self, train_loader, val_loader, test_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, _ = self._get_acc(train_loader)
            valid_acc, _ = self._get_acc(val_loader)
            test_acc, test_confusion  = self._get_acc(test_loader, confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            np.save(self.save_dir+"/test_confusion.npy", test_confusion)
            print(test_confusion)
        return train_acc, valid_acc, test_acc
