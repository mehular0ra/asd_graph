from source.utils import accuracy, TotalMeter, count_params, isfloat, BCEWithLogitsLossL2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import lr_scheduler_factory, LRScheduler
import logging

from scipy.special import expit

import ipdb


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader]) -> None:

                

        self.cfg = cfg
        self.device = self.cfg.device

        self.logger = logging.getLogger()
        self.model = model.to(self.device)
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.BCEWithLogitsLoss()


        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        # ipdb.set_trace()

        self.model.train()

        for data in self.train_dataloader:
            optimizer.zero_grad()
            # label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer,
                                step=self.current_step) 
            
            data = data.to(self.device)
            predict = self.model(data).squeeze()

            label = data.y.to(self.device)

            loss = self.loss_fn(predict, label)
            loss.backward()

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.step()
            acc = accuracy(predict, label) 
            self.train_accuracy.update_with_weight(acc, label.shape[0])

            if self.cfg.is_wandb:
                # WANDB LOGGING
                wandb.log({"LR": lr_scheduler.lr,
                       "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        logits = []

        self.model.eval()

        for data in dataloader:

            data = data.to(self.device)
            output = self.model(data).squeeze()

            label = data.y.to(self.device)

            loss = self.loss_fn(output, label)

            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            acc = accuracy(output, label)
            acc_meter.update_with_weight(acc, label.shape[0])
            # result += F.softmax(output, dim=1)[:, 1].tolist()
            logits += output.squeeze().tolist()
            labels += label.tolist()

        # convert logits to probabilities and predictions
        probabilities = expit(np.array(logits))
        predictions = np.round(probabilities)

        labels = np.array(labels)

        auc = roc_auc_score(labels, probabilities)

        metric = precision_recall_fscore_support(
            labels, predictions, average='micro')

        report = classification_report(
            labels, predictions, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def train(self):
        print("\nStarting training...")
        training_process = []
        self.current_step = 0

        ## LOG THE CONFIG IN WANDB AND LOGGING HERE
        # wandb.init(project="graph-ml", con/fig=self.cfg)
        # wandb.cfg.update(self.cfg)
        
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader,    
                                            self.val_loss, self.val_accuracy)


            self.logger.info(" | ".join([
                f'Epoch[{epoch+1}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',

                f'Val Loss:{self.val_loss.avg: .3f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Val Sen:{val_result[-1]:.4f}',
                f'Val Spe:{val_result[-2]:.4f}',
                f'Val F1:{val_result[-4]:.4f}',
                f'Val Recall:{val_result[-5]:.4f}',
                f'Val Precision:{val_result[-6]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.7f}'

            ]))

            if self.cfg.is_wandb:
                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Best Train Accuracy": self.train_accuracy.best,
                    "Train AUC": self.train_auc.avg,
                    "Train Sensitivity": self.train_sen.avg,
                    "Train Specificity": self.train_spe.avg,

                    "Val Loss": self.val_loss.avg,
                    "Val Accuracy": self.val_accuracy.avg,
                    "Best Val Accuracy": self.val_accuracy.best,
                    "Val AUC": val_result[0],
                    "Val Sensitivity": val_result[-1],
                    "Val Specificity": val_result[-2],
                    "Val F1": val_result[-4],
                    "Val Recall": val_result[-5],
                    "Val Precision": val_result[-6],

                })



