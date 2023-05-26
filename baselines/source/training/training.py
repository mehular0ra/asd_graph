from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader]) -> None:

        self.cfg = cfg
        self.logger = logging.getLogger()
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

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
        self.model.train()

        for node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer,
                                step=self.current_step)  # DOUBT

            node_feature, label = node_feature.to(
                self.cfg.device), label.to(self.cfg.device)

            if self.cfg.preprocess.continus:
                node_feature, label = continus_mixup_data(
                    node_feature, y=label)

            predict = self.model(node_feature)

            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(acc, label.shape[0])

            if self.cfg.is_wandb:
                # WANDB LOGGING
                wandb.log({"LR": lr_scheduler.lr,
                       "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for node_feature, label in dataloader:
            node_feature, label = node_feature.to(
                self.cfg.device), label.to(self.cfg.device)
            output = self.model(node_feature)

            label = label.float()  # DOUBT: why does loss_fn need float label

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def train(self):
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

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',

                f'Val Loss:{self.val_loss.avg: .3f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',

                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.7f}'
            ]))

            if self.cfg.is_wandb:
                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,

                    "Val Loss": self.val_loss.avg,
                    "Val Accuracy": self.val_accuracy.avg,
                    "Val AUC": val_result[0],
                    'Val Sensitivity': val_result[-1],
                    'Val Specificity': val_result[-2],

                    "Test Accuracy": self.test_accuracy.avg,
                    "Test AUC": test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'micro F1': test_result[-4],
                    'micro recall': test_result[-5],
                    'micro precision': test_result[-6],

                    # f'LR': self.lr_schedulers[0].lr,
                })



