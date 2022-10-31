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
import scipy.io as io

class SpeckleRunner_auxcl(BaseRunner):
    def __init__(self, arg, net, optim, torch_device, loss, logger):
        super().__init__(arg, torch_device, logger)

        self.net = net
        self.loss = loss
        self.optim = optim

        self.best_metric = 10000
        self.start_time = time.time()

        self.load(arg.load_fname)
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
        if epoch < 0:
            return

        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.net.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_dir + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

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
                self.logger.will_write("lr decay : %f" % (self.arg.lr))

            # self.scheduler.step()
            self.net.train()
            for input_, target_, path in train_loader:
            #for input_, target_, target_regression_, path in train_loader:#enumerate(train_loader):
                #######print('input_ type:' + '{}'.format(input_.dtype))
                #print('batch: '+'{}'.format(i))
                #######print(path)                
                input_= input_.to(self.torch_device)
                #######print(input_.shape)
                target_ = target_.long().to(self.torch_device)
                output_,y1,y2,y3,y4,y5,y6,y7,y8,feats_,x1,x2,x3,x4,x5,x6,x7,x8 = self.net(input_)
                #print(target_.shape)
                #print(output_.shape)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,output_,y2,y3,y4,y5,y6,y7,y8,x0,feats_,x2,x3,x4,x5,x6,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,output_,y3,y4,y5,y6,y7,y8,x0,x1,feats_,x3,x4,x5,x6,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,output_,y4,y5,y6,y7,y8,x0,x1,x2,feats_,x4,x5,x6,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,y3,output_,y5,y6,y7,y8,x0,x1,x2,x3,feats_,x5,x6,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,y3,y4,output_,y6,y7,y8,x0,x1,x2,x3,x4,feats_,x6,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,y3,y4,y5,output_,y7,y8,x0,x1,x2,x3,x4,x5,feats_,x7,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,y3,y4,y5,y6,output_,y8,x0,x1,x2,x3,x4,x5,x6,feats_,x8 = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                y0,y1,y2,y3,y4,y5,y6,y7,output_,x0,x1,x2,x3,x4,x5,x6,x7,feats_ = self.net(input_)
                loss = self.loss(output_, target_)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                #print('ground truth: ', target_.cpu().data[0])
                #print('prediction: ', output_.cpu().data[0])

                #if (i % 50) == 0:
                #    self.logger.log_write("train", epoch=epoch, loss=loss.item())
                self.logger.log_write("train", epoch=epoch, loss=loss.item())

            if val_loader is not None:
                self.valid(epoch, val_loader, test_loader)
            else:
                self.save(epoch)

    def _get_acc(self, loader, confusion=False):
        correct = 0
        preds, labels = [], []
        features = []
        for input_, target_, path in loader:
        # for input_, target_, target_regression_, path in loader:
            #print('input_ type:' + '{}'.format(input_.dtype))
            #######print('batch: '+'{}'.format(i))
            #######print(path)                
            input_= input_.to(self.torch_device)
            #######print(input_.shape)
            target_ = target_.to(self.torch_device)
            output_,y1,y2,y3,y4,y5,y6,y7,y8,feats_,x1,x2,x3,x4,x5,x6,x7,x8 = self.net(input_)
            #######print(target_.shape)
            #######print(output_.shape)
            maxval, idx = output_.max(dim=1)
            #print('ground truth: ', target_.cpu().data[0])
            #print('prediction: ', output_.cpu().data[0])
            #print('correct?: ', torch.sum(target_ == idx).float().cpu().item())
            correct += torch.sum(target_ == idx).float().cpu().item()

            features.append(feats_.cpu().data[0])

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

        if confusion:
            confusion = get_confusion(preds, labels)
            
        

        return correct / len(loader.dataset), confusion, features

    def _get_acc_path(self, loader, fname = 'individual_results.mat', confusion=False):
        correct = 0
        paths = ()
        targets = np.array([])
        outputs = np.array([])
        features = np.array([])
        
        #######preds, labels = [], []
        for input_, target_, path in loader:
        # for input_, target_, target_regression_, path in loader:
            #print('input_ type:' + '{}'.format(input_.dtype))
            #######print('batch: '+'{}'.format(i))
            #######print(path)                
            input_= input_.to(self.torch_device)
            #######print(input_.shape)
            target_ = target_.to(self.torch_device)
            output_,y1,y2,y3,y4,y5,y6,y7,y8,feats_,x1,x2,x3,x4,x5,x6,x7,x8 = self.net(input_)
            #######print(target_.shape)
            #######print(output_.shape)
            maxval, idx = output_.max(dim=1)
            #######print('ground truth: ', target_.cpu().data[0])
            #######print('prediction: ', output_.cpu().data[0])
            #######print('correct?: ', torch.sum(target_ == idx).float().cpu().item())
            #print(path, '- target: ', target_.cpu().data.numpy(), ' | output: ', idx.cpu().data.numpy())
            #######correct += torch.sum(target_ == idx).float().cpu().item()
            
            paths = paths + path
            targets = np.append(targets,target_.cpu().data.numpy())
            outputs = np.append(outputs,idx.cpu().data.numpy())
            features = np.append(features,np.expand_dims(feats.cpu().data.numpy(), axis = 0))
            
            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()

        if confusion:
            confusion = get_confusion(preds, labels)
      
        #######print(paths)
        #######print(targets)
        #######print(outputs)
        savestruct = {}
        savestruct['paths'] = paths
        savestruct['targets'] = targets
        savestruct['outputs'] = outputs
        savestruct['features'] = features
        io.savemat(self.save_dir +'/'+ fname, savestruct)
        return correct / len(loader.dataset), confusion, features


    def valid(self, epoch, val_loader, test_loader):
        self.net.eval()
        with torch.no_grad():
            acc, *_ = self._get_acc(val_loader)
            test_acc, *_ = self._get_acc(test_loader)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc)

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
            test_acc, test_confusion, features = self._get_acc_path(test_loader, fname = 'individual_results_test.mat')#, confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, time=run_time)

            np.save(self.save_dir + "/test_confusion.npy", test_confusion)
            print(test_confusion)
        return train_acc, valid_acc, test_acc
