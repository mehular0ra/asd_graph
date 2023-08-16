from source.utils import accuracy, TotalMeter, count_params, isfloat, BCEWithLogitsLossL2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb
from omegaconf import DictConfig, open_dict
from typing import List
import torch.utils.data as utils
from source.components import lr_scheduler_factory, LRScheduler
import logging

from scipy.special import expit


from ..dataset import create_mini_batch

import ipdb


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 data_list: list) -> None:
        
        # add total_steps and steps_per_epoch to cfg
        with open_dict(cfg):
            # total_steps, steps_per_epoch for lr schedular
            cfg.steps_per_epoch = (len(data_list[0]) - 1) // cfg.training.batch_size + 1
            cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

        self.cfg = cfg
        self.device = self.cfg.device

        self.logger = logging.getLogger()
        self.model = model.to(self.device)
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_list, self.test_list = data_list
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = BCEWithLogitsLossL2(self.model, cfg.training.l2)

        # self.best_test_accuracy = 0.0
        self.best_test_metrics = {
            "accuracy": 0.0,
            "auc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.test_loss, self.train_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(4)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.test_accuracy, self.train_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        # ipdb.set_trace()

        self.model.train()
        BATCH_SIZE = self.cfg.training.batch_size
        num_iter = int(len(self.train_list)/BATCH_SIZE)
        
        for i in range(num_iter):
            optimizer.zero_grad()
            self.current_step += 1
            lr_scheduler.update(optimizer=optimizer,
                                step=self.current_step)
            
            batch_list = self.train_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch = create_mini_batch(batch_list)
            batch = batch.to(self.device)
            predict = self.model(batch).squeeze()
            label = batch.y.to(self.device)

            loss = self.loss_fn(predict, label)
            loss.backward()

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.step()
            acc = accuracy(predict, label)
            self.train_accuracy.update_with_weight(acc, label.shape[0])

    def test_per_epoch(self, loss_meter, acc_meter):
        labels = []
        logits = []
        self.model.eval()

        BATCH_SIZE = self.cfg.training.batch_size
        num_iter = int(len(self.test_list)/BATCH_SIZE)
        for i in range(num_iter):
            batch_list = self.test_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch = create_mini_batch(batch_list)
            batch = batch.to(self.device)
            output = self.model(batch).squeeze()
            label = batch.y.to(self.device)
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
        # self.current_step = 0

        for epoch in range(self.epochs):
            self.current_step = 0

            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            test_result = self.test_per_epoch(self.test_loss, self.test_accuracy)

            if self.test_accuracy.avg > self.best_test_metrics["accuracy"]:
                self.best_test_metrics["accuracy"] = self.test_accuracy.avg
                auc = test_result[0]
                sensitivity = test_result[-1]
                specificity = test_result[-2]
                self.best_test_metrics["auc"] = auc
                self.best_test_metrics["sensitivity"] = sensitivity
                self.best_test_metrics["specificity"] = specificity


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

                f'Best Test Accuracy:{self.best_test_metrics["accuracy"]: .3f}%',
                f'Best Test AUC:{self.best_test_metrics["auc"]:.4f}',
                f'Best Test Sensitivity:{self.best_test_metrics["sensitivity"]:.4f}',
                f'Best Test Specificity:{self.best_test_metrics["specificity"]:.4f}',
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

                    "Best Test Accuracy": self.best_test_metrics["accuracy"],
                    "Best Test AUC": self.best_test_metrics["auc"],
                    "Best Test Sensitivity": self.best_test_metrics["sensitivity"],
                    "Best Test Specificity": self.best_test_metrics["specificity"],
                })
