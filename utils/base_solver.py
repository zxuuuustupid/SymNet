import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint

from utils import config as cfg


class BaseSolver(object):
    root_logger = logging.getLogger('solver')

    def logger(self, suffix):
        return self.root_logger.getChild(suffix)

    def clear_folder(self):
        """Clear weight and log directory"""
        logger = self.logger('clear_folder')

        for f in os.listdir(self.log_dir):
            logger.warning('Deleted log file ' + f)
            os.remove(os.path.join(self.log_dir, f))

        for f in os.listdir(self.weight_dir):
            logger.warning('Deleted weight file ' + f)
            os.remove(os.path.join(self.weight_dir, f))

    def snapshot(self, model, iter, filenames=None):
        """Save checkpoint"""
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        if filenames is None:
            filename = f'snapshot_epoch_{iter}.pth'
        else:
            filename = filenames

        pth = os.path.join(self.weight_dir, filename)
        torch.save(model.state_dict(), pth)
        self.logger('snapshot').info(f'Wrote snapshot to: {filename}')

    def initialize(self, model):
        """Load pretrained weights or initialize model"""
        logger = self.logger('initialize')

        if self.trained_weight is None:
            logger.info('Training from scratch')
            return  # PyTorch models are initialized by default
        else:
            logger.info(f'Restoring model weights from {self.trained_weight}')
            model.load_state_dict(torch.load(self.trained_weight))

    def set_lr_decay(self, optimizer, current_epoch):
        """Setup learning rate scheduler"""
        logger = self.logger('lr_scheduler')

        if self.args.lr_decay_type == 'no':
            return None  # No scheduler

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.args.lr_decay_step

        if self.args.lr_decay_type == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.args.lr_decay_rate
            )
            logger.info('Using exponential decay')
        elif self.args.lr_decay_type == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=total_steps,
                T_mult=2,
                eta_min=self.args.lr * 0.1
            )
            logger.info('Using cosine decay restarts')
        else:
            raise NotImplementedError(f"Unsupported decay type: {self.args.lr_decay_type}")

        return scheduler

    def set_optimizer(self, model):
        logger = self.logger('optimizer')
        parameters = model.parameters()

        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(parameters, lr=self.args.lr)
        elif self.args.optimizer == 'momentum':
            optimizer = optim.SGD(parameters, lr=self.args.lr, momentum=0.9)
            logger.info('Using momentum optimizer')
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(parameters, lr=self.args.lr)
            logger.info('Using Adam optimizer')
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(parameters, lr=self.args.lr, weight_decay=5e-5)
            logger.info('Using AdamW optimizer')
        elif self.args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=self.args.lr)
            logger.info('Using RMSProp optimizer')
        else:
            raise NotImplementedError(f"Optimizer '{self.args.optimizer}' is not supported")

        return optimizer
