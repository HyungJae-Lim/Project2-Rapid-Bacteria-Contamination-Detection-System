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

class SpeckleRunner(BaseRunner):
    def __init__(self, arg, net, backbone, optim, torch_device, loss, logger, margin=1e-2):
        super().__init__(arg, torch_device, logger)
        self.net = net
        self.backbone = backbone

        self.loss = loss
        self.optim = optim
        self.margin = margin
        self.description = arg.description

        self.best_metric = -1
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
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("[%dE] Saved Best Model, Accuracy=%.05f" % (epoch, self.best_metric))

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

    def train(self, train_loader, val_loader=None, test_loader=None):
        print("\nStart Train len :", len(train_loader.dataset))
        for epoch in range(self.start_epoch, self.epoch):
            if epoch != 0 and epoch % 50 == 0 and epoch < 350:
                self.arg.lr /= 2
                for p in self.optim.param_groups:
                    p['lr'] = self.arg.lr
                self.logger.will_write("lr decay : %f" % (self.arg.lr))

            train_acc = []
            print('='*150)
            self.net.train()
            train_loader_iter = iter(train_loader)
            if len(self.description) > 0:
                print('[Ongoing Thread] \"{}\" Task'.format(self.description))

            print('[Scheduler] Learning Rate: {}'.format(self.optim.param_groups[0]['lr']))
            pbar = tqdm(range(len(train_loader.dataset)//self.arg.batch_size), smoothing=0.9)
            for train_b_id in pbar:
                input, target, path = next(train_loader_iter)
                self.optim.zero_grad()
                target = torch.Tensor(list(map(int, target))).type(torch.LongTensor)
                target = Variable(target).to(self.torch_device)

                with torch.no_grad():
                    feature = self.backbone(input, training=False)

                output = self.net(feature)
                _, prediction = output.max(dim=-1)

                correct = torch.sum(target == prediction).float().cpu().item()
                accuracy = correct / self.arg.batch_size
                train_acc.append(accuracy)

                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optim.step()

                pbar.set_description_str(
                    desc="[Train] Epoch {}/{}, loss:{:.4f}, Accuracy:{:.4f}".format(
                        epoch, self.epoch-self.start_epoch-1, loss.item(), accuracy
                    ), refresh=True
                )

            avg_train_acc = sum(train_acc) / len(train_acc)
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, avg_train_acc)
            else:
                self.save(epoch)

    def _get_acc(self, loader, confusion=False, test=False):
        correct = 0
        con_correct = 0
        uncon_correct = 0
        con_count = 0
        uncon_count = 0

        labels = []
        preds = []
        preds_con = []
        preds_uncon = []

        loader_iter = iter(loader)
        num_iter = len(loader.dataset) // self.arg.batch_size_test
        num_used_imgs = num_iter * self.arg.batch_size_test
        pbar = tqdm(range(num_iter), smoothing=0.9)

        for eval_b_id in pbar:
            input, target, path = next(loader_iter)
            input = input.to(self.torch_device)
            target = torch.Tensor(list(map(int, target))).type(torch.LongTensor)
            target = target.to(self.torch_device)

            feature = self.backbone(input, training=False)
            output = self.net(feature)
            _, prediction = output.max(dim=-1)

            if prediction == 0:
                con_count += 1
                if prediction == target:
                    con_correct += 1

            elif prediction == 1:
                uncon_count += 1
                if prediction == target:
                    uncon_correct += 1

            correct += torch.sum(target == prediction).float().cpu().item()
            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

            if test:
                num_samples = self.arg.batch_size_test * (eval_b_id + 1)
                pbar.set_description_str(desc="[TEST ] Accuracy: {}".format(correct / num_samples), refresh=True)
            else:
                num_samples = self.arg.batch_size_test * (eval_b_id + 1)
                pbar.set_description_str(desc="[Valid] Accuracy: {}".format(correct / num_samples), refresh=True)

        if con_count == 0:
            print("Uncontaminated acc: {}".format(uncon_correct / uncon_count))
        elif uncon_count == 0:
            print("Contaminated acc: {}".format(con_correct / con_count))
        else:
            print("Contaminated acc: {}, Uncontaminated acc: {}".format(con_correct / con_count, uncon_correct / uncon_count))

        if confusion:
            confusion = get_confusion(preds, labels)

        return correct / num_used_imgs, confusion

    def _get_acc_path(self, loader, fname = 'individual_results.mat', confusion=False):
        correct = 0
        preds = []
        labels = []

        paths = ()
        targets = np.array([])
        outputs = np.array([])

        for input_, target_, path in loader:
            input_= input_.to(self.torch_device)
            target_ = torch.Tensor(list(map(int, target_))).type(torch.LongTensor)
            target_ = target_.to(self.torch_device)

            paths = paths + path
            output_ = self.net(input_)
            maxval, idx = output_.max(dim=1)

            targets = np.append(targets,target_.cpu().data.numpy())
            outputs = np.append(outputs,idx.cpu().data.numpy())

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

        if confusion:
            confusion = get_confusion(preds, labels)

        savestruct = {}
        savestruct['paths'] = paths
        savestruct['targets'] = targets
        savestruct['outputs'] = outputs
        io.savemat(self.save_dir +'/'+ fname, savestruct)
        return correct / len(loader.dataset), confusion


    def valid(self, epoch, val_loader, test_loader, train_acc):
        self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc(val_loader)
            test_acc, *_ = self._get_acc(test_loader, test=True)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)

            print('[Metric] Train | Valid | Test Accuracy: {:.2f}% | {:.2f}% | {:.2f}%'.format(train_acc*100, acc*100, test_acc*100))

            if acc > self.best_metric:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]" % (epoch, acc, test_acc))

    def test(self, train_loader, val_loader, test_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, _, __ = self._get_acc_path(train_loader, fname = 'individual_results_train.mat')
            valid_acc, _, __ = self._get_acc_path(val_loader, fname = 'individual_results_valid.mat')
            test_acc, test_confusion, = self._get_acc_path(test_loader, fname = 'individual_results_test.mat', confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(test_confusion)
        return train_acc, valid_acc, test_acc
