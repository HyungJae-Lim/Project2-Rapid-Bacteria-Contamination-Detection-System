import os
from .BacRunner import BacRunner
from Logger import Logger
import torch

class FoldRunner(BacRunner):
    def train(self, fold_gen, test_loader):
        base_dir = self.save_dir
        for i, (train_loader, val_loader) in enumerate(fold_gen):
            self.net.module.init_weight()
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.arg.lr, betas=self.arg.beta)
            self.save_dir = base_dir + "/fold%d"%(i)
            self.best_metric = -1 
            if os.path.exists(self.save_dir) is False:
                os.mkdir(self.save_dir)
            self.logger = Logger(self.save_dir)
            super().train(train_loader, val_loader, test_loader)
        self.save_dir = base_dir

    def test(self, fold_gen, test_loader):
        base_dir = self.save_dir
        test_acc = []
        for i in range(3):
            self.save_dir = base_dir + "/fold%d"%(i)
            self.logger = Logger(self.save_dir)
            super().load()
            test_acc.append(super().test(test_loader, test_loader, test_loader)[2])
        self.save_dir = base_dir
        print("-------------")
        print("Fold 3 Result")
        print("1 : %.4f\n2 : %.4f\n3: %.4f\n"%(test_acc[0], test_acc[1], test_acc[2]))
        print("Avg Test Acc : %.4f\n"%(sum(test_acc) / len(test_acc)))
        print("-------------")
        
    
