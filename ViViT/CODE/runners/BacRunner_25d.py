import os
import time
import torch
import numpy as np

from .BacRunner import BacRunner
from Logger import Logger

from sklearn.metrics import confusion_matrix

from collections import defaultdict
from utils import get_confusion


class BacRunner_25d(BacRunner):
    
    def _get_acc(self, loader, confusion=False):
        patch_correct_sum = 0
        preds, labels = [], []

        cell_target = {}
        cell_correct = defaultdict(lambda : 0)
        for input_, target_, path in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            _, idx = output_.max(dim=1)
            patch_correct = torch.sum(target_ == idx).float().cpu().item()
            patch_correct_sum += patch_correct

            for b in range(len(path)):
                cell_correct[path[b]] += output_[b]
                cell_target[path[b]] = target_[b]

        correct = 0
        for k, v in cell_correct.items():
            target_ = cell_target[k]
            _, idx = v.max(dim=0)
            correct += (target_ == idx).float().cpu().item()

            preds += idx.view(-1).tolist()
            labels += target_.view(-1).tolist()

        acc = correct / len(cell_correct.keys())

        if confusion:
            idx_to_cls = {v:k for k,v in loader.dataset.class_to_idx.items()}
            preds = [idx_to_cls[i] for i in preds]
            labels = [idx_to_cls[i] for i in labels]
            confusion = confusion_matrix(labels, preds, labels=loader.dataset.classes)
        return acc, patch_correct_sum / len(loader.dataset), confusion

    def valid(self, epoch, val_loader, test_loader):
        self.net.eval()
        with torch.no_grad():
            acc, valid_patch, *_ = self._get_acc(val_loader)
            test_acc, test_patch, *_ = self._get_acc(test_loader)
            self.logger.log_write("valid", epoch=epoch, acc=acc, test_acc=test_acc,
                                  valid_patch=valid_patch, test_patch=test_patch)

            if acc > self.best_metric:
                self.best_metric = acc
                self.save(epoch, "epoch[%05d]_acc[%.4f]_test[%.4f]_patchacc[%.4f]_test_patch[%.4f]"%(epoch, acc, test_acc, valid_patch, test_patch))


    def test(self, train_loader, val_loader, test_loader):
        print("\n Start Test")
        self.load()
        self.net.eval()
        with torch.no_grad():
            train_acc, train_patch, _ = self._get_acc(train_loader)
            valid_acc, valid_patch, _ = self._get_acc(val_loader)
            test_acc, test_patch, test_confusion  = self._get_acc(test_loader, confusion=True)

            end_time = time.time()
            run_time = end_time - self.start_time
            self.logger.log_write("test", fname="test",
                                  train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc, 
                                  train_patch=train_patch, valid_patch=valid_patch, test_patch=test_patch,
                                  time=run_time)

            np.save(self.save_dir+"/test_confusion.npy", test_confusion)
            print(test_confusion)
        return train_acc, valid_acc, test_acc
