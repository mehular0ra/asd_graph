from source.utils import accuracy, TotalMeter, count_params, isfloat, BCEWithLogitsLossL2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig
from typing import List, Tuple
import torch.utils.data as utils
from source.components import lr_scheduler_factory, LRScheduler
import logging

from scipy.special import expit

from .Train import Train

from typing import Tuple

from ..components import optimizer_factory


class CVTrain(Train):
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[Tuple[utils.DataLoader]]) -> None:
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.init_model = model
        self.init_optimizers = optimizers
        self.init_lr_schedulers = lr_schedulers
        super().__init__(cfg, model, optimizers, lr_schedulers, dataloaders[0])
        self.fold_results = []


    def print_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)


    def reset_model(self):
        self.model = type(self.init_model)(self.cfg).to(self.device)
        self.optimizers = [optimizer_factory(self.model, opt_cfg) for opt, opt_cfg in zip(
            self.init_optimizers, self.cfg.optimizer)]




    def train(self):
        # Loop over each fold
        for fold, (train_dataloader, test_dataloader) in enumerate(self.dataloaders):
            self.reset_model()
            # print weights at the start of the fold
            print(f"Weights at start of fold {fold}")
            # self.print_weights()

            # Set the train and test dataloaders for this fold
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader

            # Training process for this fold
            fold_process = []
            self.current_step = 0

            # print weights at the end of the fold
            print(f"Weights at start of fold {fold}")
            self.print_weights()

            self.logger.info(f"\nFold[{fold+1}/{len(self.dataloaders)}]")



            for epoch in range(self.epochs):
                self.reset_meters()

                # print train and test accuracy 
                self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
                test_result = self.test_per_epoch(self.test_dataloader,
                                                    self.test_loss, self.test_accuracy)

                self.logger.info(" | ".join([
                    f'Epoch[{epoch+1}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',

                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Test AUC:{test_result[0]:.4f}',
                    f'Test Sen:{test_result[-1]:.4f}',
                    f'Test Spe:{test_result[-2]:.4f}',
                    f'Test F1:{test_result[-4]:.4f}',
                    f'Test Recall:{test_result[-5]:.4f}',
                    f'Test Precision:{test_result[-6]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.7f}'
                ]))

                if self.cfg.is_wandb:
                    wandb.log({
                        "Train Loss": self.train_loss.avg,
                        "Train Accuracy": self.train_accuracy.avg,

                        "Test Loss": self.test_loss.avg,
                        "Test Accuracy": self.test_accuracy.avg,
                        "Test AUC": test_result[0],
                        "Test Sensitivity": test_result[-1],
                        "Test Specificity": test_result[-2],
                        "Test F1": test_result[-4],
                        "Test Recall": test_result[-5],
                        "Test Precision": test_result[-6],
                    })

                # Store result of this epoch
                fold_process.append({
                    'Train Loss': self.train_loss.avg,
                    'Train Accuracy': self.train_accuracy.avg,
                    'Test Loss': self.test_loss.avg,
                    'Test Accuracy': self.test_accuracy.avg,
                    'Test AUC': test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'Test F1': test_result[-4],
                    'Test Recall': test_result[-5],
                    'Test Precision': test_result[-6]
                })

               

            # Store the results of this fold
            self.fold_results.append(fold_process)

            # print weights at the end of the fold
            print(f"Weights at end of fold {fold}")
            self.print_weights()

            

            # You may want to calculate and print the average results over all folds here
            # ...
