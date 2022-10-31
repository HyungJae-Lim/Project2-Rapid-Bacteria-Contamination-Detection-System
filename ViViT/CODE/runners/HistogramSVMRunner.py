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
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)
        self.net = net
        self.loss = loss
        self.optim = optim
        self.description = arg.description
        self.arg = arg

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
                input_, target_, path = next(train_loader_iter)
                target_ = torch.Tensor(list(map(int, target_))).type(torch.LongTensor)
                target_ = Variable(target_).to(self.torch_device)
                histogram = Variable(input_).to(self.torch_device)

                target_ = 2 * (target_.float() - 0.5)
                prediction, loss = self.net(histogram, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                prediction[prediction < 0] = -1
                prediction[prediction >= 0] = 1

                correct = torch.sum(target_ == prediction).float().cpu().item()
                accuracy = correct / self.arg.batch_size
                train_acc.append(accuracy)

                pbar.set_description_str(
                    desc="[Train] Epoch {}/{}, CE loss:{:.4f}, Accuracy:{:.3f}".format(
                        epoch, self.epoch-self.start_epoch-1, loss.item(), accuracy
                    ), refresh=True
                )

            avg_train_acc = sum(train_acc) / len(train_acc)
            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader, avg_train_acc)
            else:
                self.save(epoch)

    def _get_acc(self, loader, confusion=False, test=False):
        con_count = 0
        con_correct = 0
        uncon_count = 0
        uncon_correct = 0
        correct = 0
        preds = []
        labels = []

        con_stack = []
        uncon_stack = []

        loader_iter = iter(loader)
        pbar = tqdm(range(len(loader.dataset)//self.arg.batch_size_test), smoothing=0.9)

        for eval_b_id in pbar:
            input_, target_, path = next(loader_iter)
            histogram = input_.to(self.torch_device)
            target_ = torch.Tensor(list(map(int, target_))).type(torch.LongTensor)
            target_ = target_.to(self.torch_device)

            target_ = 2 * (target_ - 0.5)
            prediction, loss = self.net(histogram, target_)

            prediction[prediction < 0] = -1
            prediction[prediction >= 0] = 1
            correct += torch.sum(target_ == prediction).float().cpu().item()

            # Contaminated samples
            con_tmp = prediction[prediction == -1]
            target_tmp = target_[prediction == -1]
            con_correct += torch.sum((con_tmp == target_tmp).int())
            con_count += torch.sum((target_ == -1).int())

            # Uncontaminated samples
            uncon_tmp = prediction[prediction == 1]
            target_tmp = target_[prediction == 1]
            uncon_correct += torch.sum((uncon_tmp == target_tmp).int())
            uncon_count += torch.sum((target_ == 1).int())

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

            con = con_correct / (con_count + 1e-4)
            uncon = uncon_correct / (uncon_count + 1e-4)
            if self.arg.test:
                pbar.set_description_str(desc="[TEST] Accuracy: {:.3f} | Contam: {:.3f} | Uncontam: {:.3f}".format(
                    correct / self.arg.batch_size_test / (eval_b_id+1), con, uncon)
                    , refresh=True
                )

            else:
                pbar.set_description_str(desc="[Valid] Accuracy: {:.3f} | Contam: {:.3f} | Uncontam: {:.3f}".format(
                    correct / self.arg.batch_size_test / (eval_b_id+1), con, uncon)
                    , refresh=True
                )

            num=499
            if self.arg.test:
                if con_count == 0 and (eval_b_id + 1) % num == 0:
                    print("Uncontaminated acc: {:.3f}, Vote: {}/{}".format(uncon, uncon_correct, uncon_count))
                    uncon_stack.append(uncon_correct)
                    uncon_correct = 0; uncon_count = 0

                elif uncon_count == 0 and (eval_b_id + 1) % num == 0:
                    print("Contaminated acc: {:.3f}, Vote: {}/{}".format(con, con_correct, con_count))
                    con_stack.append(con_correct)
                    con_correct = 0; con_count = 0

                elif (eval_b_id + 1) % num == 0:
                    print("Contam acc: {:.3f}, Vote:{}/{}\nUncontam acc: {}, Vote:{}/{}".format(
                                con_correct / con_count, con_correct, con_count,
                                uncon_correct / uncon_count, uncon_correct, uncon_count
                            )
                    )
                    uncon_stack.append(uncon_correct)
                    con_stack.append(con_correct)
                    con_correct = 0; con_count = 0
                    uncon_correct = 0; uncon_count = 0

        if self.arg.test:
            print("Total contam: {} | uncontam: {}".format(sum(con_stack), sum(uncon_stack)))
        else:
            self.logger.log_write_("[Valid] Contam: {:.4f} | Uncontam: {:.4f}".format(con, uncon))

        if confusion:
            confusion = get_confusion(preds, labels)

        return correct / len(loader.dataset), confusion

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
#            test_acc, *_ = self._get_acc(test_loader, test=True)
            test_acc = 0
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)
            print('[Metric] Train | Valid | Test Accuracy: {:.2f}% | {:.2f}% | {:.2f}%'.format(train_acc*100, acc*100, test_acc*100))

            if acc > self.best_metric or (epoch + 1) % 5 == 0:
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
